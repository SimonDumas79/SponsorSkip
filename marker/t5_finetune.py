"""Fine-tune the community SponsorBlock T5 on OUR labels, warm-started from its weights. (2026-09-22)

On holdout 4 the community model (xenova/sponsorblock-ml) skipped the most ad time (75.4% on videos it
cannot have trained on) but lost 8.0 s of show per video and broke the 60 s cap; ours lost less but
found less. This keeps its breadth (years of SponsorBlock data) and adds our labels: our caption
format, self-promotion, and our read boundaries. Training data is built exactly in its own format,
with its own chunking code (segment.generate_segments): each ~500-token chunk becomes
"EXTRACT_SEGMENTS: <cleaned words>" -> "START_SPONSOR_TOKEN <words> END_SPONSOR_TOKEN ..." or
"NO_SEGMENT_TOKEN", positives and negatives balanced 50/50 as it was trained.

Leakage: only the 160 pooled videos whose SponsorBlock labels postdate its training (sb_dates.json),
with the SAME channel-grouped 5 folds the stack uses there (stack_check.oof_seeded on that subset), so
every video is scored by a copy that never saw its labels. Predictions go to
data/t5ft_predictions.jsonl in sponsorblock_ml.py's format (its classifier still scores each find).

    python -u marker/t5_finetune.py
"""

import ctypes
import datetime
import json
import math
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

from build_production import pooled
from experiments import runs
from sponsorblock_ml import CLASSIFIER, T5, game_running, import_their_code, words_of
from train import DATA, LAST_LINE_SECONDS, Rows

sys.stdout.reconfigure(encoding="utf-8")
OUT = DATA / "t5ft_predictions.jsonl"
EPOCHS, BATCH, LR = 2, 8, 5e-5
CAPS = {"features.npz": "captions", "features_tail.npz": "captions_tail", "features_holdout2.npz": "captions"}


def main() -> int:
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    torch.set_num_threads(4)
    their_predict, their_preprocess, their_segment = import_their_code()
    from shared import CustomTokens, END_SEGMENT_TEMPLATE, START_SEGMENT_TEMPLATE
    device = "cuda"
    rows, _, _ = pooled()
    d = json.load(open(DATA / "sb_dates.json"))
    cutoff = datetime.datetime(2022, 4, 1).timestamp() * 1000
    unseen = np.array([not (d.get(str(v), {}).get("first") and d[str(v)]["first"] < cutoff) for v in rows.video])
    S = Rows(rows.X[unseen], rows.y[unseen], rows.video[unseen], rows.channel[unseen], rows.start_seconds[unseen],
             rows.split[unseen], rows.feature_names)
    cat_all = np.concatenate([np.load(DATA / f)["category"] for f in CAPS])[unseen]
    cap_dir = {}
    for f, cd in CAPS.items():
        for v in np.unique(np.load(DATA / f)["video"]):
            cap_dir[str(v)] = cd

    tok = AutoTokenizer.from_pretrained(T5)
    videos = [str(v) for v in np.unique(S.video)]
    chunks, words_by_video = {}, {}
    for v in videos:
        r = np.flatnonzero(S.video == v)
        s = S.start_seconds[r]
        segs = []
        for a, b in runs(S.y[r]):
            cats = set(cat_all[r][a:b].tolist()) - {""}
            segs.append({"start": float(s[a]), "end": float(s[b]) if b < len(s) else float(s[-1] + LAST_LINE_SECONDS),
                         "category": "sponsor" if "sponsor" in cats else "selfpromo"})
        caps = json.loads((DATA / cap_dir[v] / f"{v}.json").read_text(encoding="utf-8"))
        words = words_of(caps)
        words_by_video[v] = words
        labelled = their_segment.generate_labelled_segments([dict(w) for w in words], tok,
                                                            their_segment.SegmentationArguments(), segs)
        ex = []
        for seg in labelled:
            text = " ".join(w["cleaned"] for w in seg)
            found = their_preprocess.extract_sponsors(seg)
            if found:
                target = f" {CustomTokens.BETWEEN_SEGMENTS.value} ".join(
                    f"{START_SEGMENT_TEMPLATE.format(x['category'].upper())} {' '.join(w['cleaned'] for w in x['words'])} "
                    f"{END_SEGMENT_TEMPLATE.format(x['category'].upper())}" for x in found)
            else:
                target = CustomTokens.NO_SEGMENT.value
            ex.append((text, target, bool(found)))
        chunks[v] = ex
    n_pos = sum(e[2] for ex in chunks.values() for e in ex)
    print(f"{len(videos)} unseen pooled videos, {S.reads} reads; {sum(len(e) for e in chunks.values())} chunks, "
          f"{n_pos} with a read", flush=True)

    channels = np.array(sorted(set(S.channel)))
    np.random.default_rng(0).shuffle(channels)   # the folds stack_check.oof_seeded uses on this subset
    chan_of = {v: S.channel[np.flatnonzero(S.video == v)[0]] for v in videos}
    ctok = AutoTokenizer.from_pretrained(CLASSIFIER)
    clf = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER).to(device).eval()
    labels = [clf.config.id2label[i].lower() for i in range(clf.config.num_labels)]
    prefix = CustomTokens.EXTRACT_SEGMENTS_PREFIX.value
    OUT.write_text("", encoding="utf-8")
    started = time.time()
    for f, fold in enumerate(np.array_split(channels, 5), 1):
        test_v = [v for v in videos if chan_of[v] in set(fold)]
        train_ex = [e for v in videos if chan_of[v] not in set(fold) for e in chunks[v]]
        rng = np.random.default_rng(f)
        pos = [e for e in train_ex if e[2]]
        neg = [e for e in train_ex if not e[2]]
        neg = [neg[i] for i in rng.permutation(len(neg))[:len(pos)]]
        data = pos + neg
        torch.manual_seed(0)
        model = AutoModelForSeq2SeqLM.from_pretrained(T5).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        steps = EPOCHS * math.ceil(len(data) / BATCH)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
        model.train()
        for epoch in range(EPOCHS):
            order = rng.permutation(len(data))
            for i in range(0, len(order), BATCH):
                while game_running():
                    print("  paused: a game is running", flush=True)
                    time.sleep(120)
                b = [data[k] for k in order[i:i + BATCH]]
                enc = tok([prefix + t for t, _, _ in b], truncation=True, max_length=512, padding=True, return_tensors="pt").to(device)
                lab = tok([t for _, t, _ in b], truncation=True, max_length=512, padding=True, return_tensors="pt").input_ids.to(device)
                lab[lab == tok.pad_token_id] = -100
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**enc, labels=lab).loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
        model.eval()
        with OUT.open("a", encoding="utf-8") as out, torch.no_grad():
            for v in test_v:
                preds = their_predict.predict(v, model, tok, their_segment.SegmentationArguments(),
                                              words=[dict(w) for w in words_by_video[v]])
                texts = [their_preprocess.clean_text(" ".join(w["text"] for w in p["words"])) for p in preds]
                probs = []
                for i in range(0, len(texts), 32):
                    enc = ctok(texts[i:i + 32], truncation=True, padding=True, return_tensors="pt").to(device)
                    probs += torch.softmax(clf(**enc).logits.float(), -1).cpu().tolist()
                for p, pr in zip(preds, probs):
                    out.write(json.dumps({"video": v, "fold": f, "start": p["start"], "end": p["end"],
                                          "t5_category": p["category"], "probs": dict(zip(labels, pr))}) + "\n")
                if not preds:
                    out.write(json.dumps({"video": v, "fold": f, "start": None, "end": None}) + "\n")
        del model, opt
        torch.cuda.empty_cache()
        print(f"  fold {f}/5: trained on {len(data)} chunks ({len(pos)} with a read), scored {len(test_v)} videos; "
              f"{(time.time() - started) / 60:.0f} min so far", flush=True)
    del clf
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
