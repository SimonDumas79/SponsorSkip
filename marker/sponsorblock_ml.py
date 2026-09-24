"""Run the community SponsorBlock model (xenova/sponsorblock-ml) on our videos, for comparison. (2026-09-22)

Simon asked to compare our scores with the SponsorBlock ML model. That project (GPL-3.0, trained on the
SponsorBlock database in early 2022; weights on Hugging Face) extracts sponsor text with a T5 model
(Xenova/sponsorblock-small) and filters each find with a BERT classifier (EColi/SB_Classifier, its
default). Its own chunking, output parsing and merging code is used unchanged, from a clone kept in
data/sponsorblock-ml (data/ is never committed, so none of their code enters this repo).

Two differences from their live tool, both forced: the words come from OUR caption files (so both
systems read the same text), and each caption line's words are spread evenly over the line's time,
because our files keep line times, not word times.

LEAKAGE: it was trained on SponsorBlock labels up to early 2022, the same source as our labels, so it
may have seen some pooled videos. The 169-video channel set is recent uploads and is a clean test.

Every find is recorded with the classifier's probabilities to data/sbml_predictions.jsonl, so the
grading can sweep the probability threshold without running the model again.

    python -u marker/sponsorblock_ml.py              # pooled 205 videos and the channel set
"""

import ctypes
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).parent
DATA = HERE / "data"
SRC = DATA / "sponsorblock-ml" / "src"
OUT = DATA / "sbml_predictions.jsonl"
T5, CLASSIFIER = "Xenova/sponsorblock-small", "EColi/SB_Classifier"
SETS = [("features.npz", "captions"), ("features_tail.npz", "captions_tail"),
        ("features_holdout2.npz", "captions"), ("features_channels.npz", "captions_channels"),
        ("features_holdout4.npz", "captions"), ("features_negatives.npz", "captions_negatives"),
        ("features_holdout3.npz", "captions"), ("features_holdout5.npz", "captions")]
LAST_LINE_SECONDS = 2.2


def import_their_code():
    """Their modules import packages only their training needs; stand-ins let the inference path load."""
    for name, attrs in (("datasets", ["load_dataset"]),
                        ("youtube_transcript_api", ["YouTubeTranscriptApi", "CouldNotRetrieveTranscript",
                                                    "YouTubeRequestFailed", "TooManyRequests"]),
                        ("pandas", [])):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            for a in attrs:
                setattr(mod, a, type(a, (Exception,), {}))
            sys.modules[name] = mod
    # Their src has modules named like ours (predict, train...). Load theirs with their names in place,
    # then put ours back, so whichever was imported first cannot be picked up by the other.
    clashing = ("predict", "train", "model", "segment", "preprocess", "shared", "utils", "classify", "errors")
    ours = {n: sys.modules.pop(n) for n in clashing if n in sys.modules}
    sys.path.insert(0, str(SRC))
    try:
        import predict as their_predict
        import preprocess as their_preprocess
        import segment as their_segment
        import shared as their_shared
    finally:
        sys.path.remove(str(SRC))
        theirs = {n: sys.modules.pop(n) for n in clashing if n in sys.modules}
        sys.modules.update(ours)
        sys.modules["sbml_shared"] = their_shared
    return their_predict, their_preprocess, their_segment


def words_of(caps: dict) -> list[dict]:
    """Our caption lines as their word list: each line's words spread evenly over the line's time."""
    lines = caps["lines"]
    words = []
    for k, line in enumerate(lines):
        start = float(line["start"])
        end = float(lines[k + 1]["start"]) if k + 1 < len(lines) else start + LAST_LINE_SECONDS
        toks = [t for t in line["text"].split() if t not in ("[Music]", "[Applause]", "[Laughter]")]
        step = max(end - start, 0.01) / max(len(toks), 1)
        for j, t in enumerate(toks):
            words.append({"text": t, "start": start + j * step, "end": start + (j + 1) * step})
    return words


class LivePredictor:
    """The community model on ONE video, as serving calls it: candidate v3's seventh detector.

    Loads both models once. CPU by default, like the rest of the serving path (the extension must not
    fight a game for the card). Returns the same records main() writes to sbml_predictions.jsonl, so
    stack_sbml.sbml_stream turns them into a line score exactly as in the graded runs.
    """

    def __init__(self, device: str = "cpu"):
        self.predict, self.preprocess, self.segment = import_their_code()
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(T5)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(T5).to(device).eval()
        self.ctok = AutoTokenizer.from_pretrained(CLASSIFIER)
        self.clf = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER).to(device).eval()
        self.labels = [self.clf.config.id2label[i].lower() for i in range(self.clf.config.num_labels)]

    def __call__(self, caps: dict) -> list[dict]:
        vid = str(caps["videoID"])
        with torch.no_grad():
            preds = self.predict.predict(vid, self.model, self.tok, self.segment.SegmentationArguments(),
                                         words=words_of(caps))
            texts = [self.preprocess.clean_text(" ".join(w["text"] for w in p["words"])) for p in preds]
            probs = []
            for i in range(0, len(texts), 32):
                enc = self.ctok(texts[i:i + 32], truncation=True, padding=True, return_tensors="pt").to(self.device)
                probs += torch.softmax(self.clf(**enc).logits.float(), -1).cpu().tolist()
        return [{"video": vid, "start": p["start"], "end": p["end"], "probs": dict(zip(self.labels, pr))}
                for p, pr in zip(preds, probs)]


def game_running() -> bool:
    import subprocess
    try:
        tasks = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True,
                               creationflags=0x08000000).stdout.lower()
    except Exception:
        return False
    for ln in tasks.splitlines():
        proc = ln.split('","')[0].strip('"')
        if proc.startswith(("eosoverlayrenderer", "epicwebhelper", "unrealcefsubprocess", "crashreportclient")):
            continue
        if any(g in proc for g in ("valorant", "-win64-shipping", "cs2.exe", "fortnite", "overwatch", "eldenring")):
            return True
    return False


def main() -> int:
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    torch.set_num_threads(4)
    their_predict, their_preprocess, their_segment = import_their_code()
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(T5)
    model = AutoModelForSeq2SeqLM.from_pretrained(T5).to(device).eval()
    ctok = AutoTokenizer.from_pretrained(CLASSIFIER)
    clf = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER).to(device).eval()
    labels = [clf.config.id2label[i].lower() for i in range(clf.config.num_labels)]
    print(f"T5 {T5} (max {tok.model_max_length} tokens), classifier {CLASSIFIER} labels {labels}", flush=True)

    done = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as f:
            done = {json.loads(l)["video"] for l in f if l.strip()}
    todo = []
    for feats, caps_dir in SETS:
        path = DATA / feats
        if path.exists():
            todo += [(str(v), caps_dir, feats) for v in np.unique(np.load(path)["video"]) if str(v) not in done]
    print(f"{len(done)} videos done, {len(todo)} to run", flush=True)

    started = time.time()
    with OUT.open("a", encoding="utf-8") as out:
        for n, (vid, caps_dir, feats) in enumerate(todo, 1):
            while game_running():
                model.to("cpu"), clf.to("cpu")
                torch.cuda.empty_cache()
                print("  paused: a game is running; checking again in 2 min", flush=True)
                time.sleep(120)
                model.to(device), clf.to(device)
            caps = json.loads((DATA / caps_dir / f"{vid}.json").read_text(encoding="utf-8"))
            words = words_of(caps)
            with torch.no_grad():
                preds = their_predict.predict(vid, model, tok, their_segment.SegmentationArguments(), words=words)
                texts = [their_preprocess.clean_text(" ".join(w["text"] for w in p["words"])) for p in preds]
                probs = []
                for i in range(0, len(texts), 32):
                    enc = ctok(texts[i:i + 32], truncation=True, padding=True, return_tensors="pt").to(device)
                    probs += torch.softmax(clf(**enc).logits.float(), -1).cpu().tolist()
            for p, pr in zip(preds, probs):
                out.write(json.dumps({"video": vid, "set": feats, "start": p["start"], "end": p["end"],
                                      "t5_category": p["category"], "probs": dict(zip(labels, pr))}) + "\n")
            if not preds:
                out.write(json.dumps({"video": vid, "set": feats, "start": None, "end": None}) + "\n")
            out.flush()
            if n % 25 == 0 or n == len(todo):
                rate = (time.time() - started) / n
                print(f"  [{n}/{len(todo)}] {rate:.1f} s per video, ~{rate * (len(todo) - n) / 60:.0f} min left", flush=True)
    del model, clf
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
