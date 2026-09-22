"""Fine-tune the MiniLM encoder itself to score caption lines, out of fold. (2026-09-22)

The shipped marker uses MiniLM FROZEN: the encoder was trained for general sentence meaning, and
only a thin linear layer on top learns what an ad is. It has no reason to separate "this host is
reviewing a phone" from "this host is selling you a VPN", and those product-talk false alarms are
what pin the free tier's threshold (error_budget.py). This trains the whole encoder on our labels.

Setup, fixed before any result was seen:
  input     the line (first segment) and the WINDOW lines either side of it (second segment),
            so the encoder reads a stretch of the video, not one line
  model     all-MiniLM-L6-v2 + mean pooling + one linear output, every weight trained
  training  2 epochs, AdamW, lr 3e-5 for the encoder and 1e-3 for the output layer, 6% warmup then
            linear decay, batch 64, bf16, the same pos_weight rule as the marker (negatives / positives)
  folds     the pooled 205 videos, the SAME five channel folds (seed 0) as every other stage: each
            fold's lines are scored by a model that never saw that channel

Output: data/finetune_oof_seed<seed>.npy, one probability per pooled row. finetune_eval.py
compares it with the frozen marker through the same context model, edge heads and rule.

Gentle: when a game is running, the model and optimiser move off the GPU until it closes.

    python -u marker/finetune_minilm.py              # seed 0, about 30 min on the 3080
    python -u marker/finetune_minilm.py --seed 1
"""

import argparse
import ctypes
import json
import math
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from build_production import pooled
from features import MODEL_NAME
from train import DATA

WINDOW = 8          # lines either side of the scored line
MAX_TOKENS = 192
EPOCHS, BATCH, LR_ENCODER, LR_HEAD, WARMUP = 2, 64, 3e-5, 1e-3, 0.06
SETS = [("examples.jsonl", "features.npz"), ("examples_tail.jsonl", "features_tail.npz"),
        ("examples_holdout2.jsonl", "features_holdout2.npz")]


def game_running() -> str | None:
    try:
        tasks = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True,
                               creationflags=0x08000000).stdout.lower()
    except Exception:
        return None
    for ln in tasks.splitlines():
        proc = ln.split('","')[0].strip('"')
        if proc.startswith(("eosoverlayrenderer", "epicwebhelper", "unrealcefsubprocess", "crashreportclient")):
            continue
        if any(g in proc for g in ("valorant", "-win64-shipping", "cs2.exe", "fortnite", "overwatch", "eldenring")):
            return proc
    return None


def line_texts(rows) -> list[str]:
    """Every pooled row's caption text, in the pooled order (checked against it)."""
    out, videos = [], []
    for examples, features in SETS:
        text = {}
        with (DATA / examples).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    text[(r["videoID"], r["i"])] = r["text"]
        d = np.load(DATA / features)
        out += [text[(str(v), int(i))] for v, i in zip(d["video"], d["line"])]
        videos.append(d["video"])
    assert (np.concatenate(videos) == rows.video).all(), "examples are not in the pooled order"
    return out


def pairs(rows, texts: list[str]) -> list[tuple[str, str]]:
    """(the line, the WINDOW lines either side of it) for every row, never crossing into another video."""
    out = [None] * len(rows)
    for vid in np.unique(rows.video):
        r = np.flatnonzero(rows.video == vid)
        t = [texts[k] for k in r]
        for j, k in enumerate(r):
            out[k] = (t[j], " ".join(t[max(0, j - WINDOW):j + WINDOW + 1]))
    return out


class Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, ids, mask, types):
        h = self.encoder(input_ids=ids, attention_mask=mask, token_type_ids=types).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)


def encode(tok, batch: list[tuple[str, str]], device):
    enc = tok([a for a, _ in batch], [b for _, b in batch], truncation="longest_first", max_length=MAX_TOKENS,
              padding=True, return_tensors="pt")
    return (enc["input_ids"].to(device), enc["attention_mask"].to(device),
            enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])).to(device))


def move_optimizer(opt, device) -> None:
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def wait_for_games(model, opt, device) -> None:
    game = game_running()
    if not game:
        return
    model.to("cpu")
    move_optimizer(opt, "cpu")
    torch.cuda.empty_cache()
    while game:
        print(f"  paused: {game} is running; the GPU is free for it. Checking again in 2 min", flush=True)
        time.sleep(120)
        game = game_running()
    model.to(device)
    move_optimizer(opt, device)
    print("  resumed", flush=True)


def train_fold(train_idx, test_idx, data, y, tok, device, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Scorer().to(device)
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": LR_ENCODER},
                             {"params": model.head.parameters(), "lr": LR_HEAD}], weight_decay=0.01)
    steps = EPOCHS * math.ceil(len(train_idx) / BATCH)
    warm = int(WARMUP * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    pos = y[train_idx].sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train_idx) - pos) / pos, device=device))
    step, started = 0, time.time()
    model.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(train_idx)
        for i in range(0, len(order), BATCH):
            if step % 200 == 0:
                wait_for_games(model, opt, device)
            b = order[i:i + BATCH]
            ids, mask, types = encode(tok, [data[k] for k in b], device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, mask, types)
            loss = loss_fn(logits.float(), torch.from_numpy(y[b]).float().to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 500 == 0:
                rate = step / (time.time() - started)
                print(f"    epoch {epoch + 1} step {step}/{steps}  loss {loss.item():.3f}  "
                      f"{rate:.1f} steps/s, ~{(steps - step) / rate / 60:.0f} min left in this fold", flush=True)
    model.eval()
    out = np.zeros(len(test_idx), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(test_idx), 256):
            b = test_idx[i:i + 256]
            ids, mask, types = encode(tok, [data[k] for k in b], device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out[i:i + len(b)] = torch.sigmoid(model(ids, mask, types).float()).cpu().numpy()
    del model, opt
    torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    torch.set_num_threads(4)
    device = "cuda"
    rows, _, _ = pooled()
    data = pairs(rows, line_texts(rows))
    y = rows.y.astype(np.float32)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"pooled: {rows.videos} videos, {len(rows):,} lines, {int(y.sum()):,} inside a read; "
          f"window +-{WINDOW} lines, {MAX_TOKENS} tokens, {EPOCHS} epochs, seed {args.seed}", flush=True)

    channels = np.array(sorted(set(rows.channel)))
    np.random.default_rng(0).shuffle(channels)   # the same folds as experiments.cross_validate(seed=0)
    out_path = DATA / f"finetune_oof_seed{args.seed}.npy"
    partial = DATA / f"finetune_oof_seed{args.seed}.partial.npy"
    oof = np.load(partial) if partial.exists() else np.full(len(rows), np.nan, dtype=np.float32)
    started = time.time()
    for f, fold in enumerate(np.array_split(channels, 5), 1):
        test = np.isin(rows.channel, fold)
        if not np.isnan(oof[test]).any():
            print(f"  fold {f}/5: already scored (resumed)", flush=True)
            continue
        print(f"  fold {f}/5: training on {int((~test).sum()):,} lines, scoring {int(test.sum()):,}", flush=True)
        oof[test] = train_fold(np.flatnonzero(~test), np.flatnonzero(test), data, y, tok, device, args.seed)
        np.save(partial, oof)
    assert not np.isnan(oof).any()
    np.save(out_path, oof)
    print(f"done in {(time.time() - started) / 60:.1f} min -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
