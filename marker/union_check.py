"""Simon's question, 2026-09-21: do two detectors miss DIFFERENT reads?

If they do, their union covers more than either alone, and the marker should be
a union rather than a single best detector. If they miss the same reads, a
second detector buys nothing and only costs.

Measured here on the low-review holdout, where the cue patterns get 62.9%:
  cue patterns   a phrase matcher -- "brought to you by", "use code", a bare URL
  qwen3:8b       a semantic detector, asked per window "is there an ad here?"

These two fail for different reasons in principle: a phrase matcher cannot see
a read that opens "EXO has asked me to share with you about their anniversary
sale", and a language model does not care whether stock phrasing is present.
Whether that principle holds in the data is what this measures.

False positives are measured too, on windows containing no read at all. A
second detector that catches the missing reads by flagging everything has not
helped -- in the default tier a false positive skips real show content.

Run: python marker/union_check.py     (uses the GPU; unloads the model after)
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OLLAMA = "http://localhost:11434"
WINDOW = 45  # caption lines shown to the model, about 2 minutes

ASK = """You are given numbered caption lines from part of a YouTube video.

Is there a SPONSOR READ in these lines -- an advertisement the host reads out for a company that paid them, interrupting the video's actual subject?

A review, unboxing or demonstration of a product is NOT a sponsor read: if the product is the video's subject, nothing is being interrupted. The host's own merch, Patreon or other videos are not sponsor reads either.

Answer ONLY with JSON: {"sponsor": true} or {"sponsor": false}"""

SCHEMA = {"type": "object", "properties": {"sponsor": {"type": "boolean"}}, "required": ["sponsor"]}


def ask(model: str, lines: list[str], timeout: float = 90.0) -> bool | None:
    rendered = "\n".join(f"{i:3d}: {t}" for i, t in enumerate(lines))
    body = json.dumps({
        "model": model, "stream": False, "think": False, "format": SCHEMA,
        "keep_alive": "120s", "options": {"num_ctx": 4096, "temperature": 0},
        "messages": [{"role": "user", "content": f"{ASK}\n\n{rendered}"}],
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return bool(json.loads(json.load(r)["message"]["content"])["sponsor"])
    except (urllib.error.URLError, KeyError, ValueError, TimeoutError):
        return None


def unload(model: str) -> None:
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"{OLLAMA}/api/generate", data=json.dumps({"model": model, "keep_alive": 0}).encode(),
            headers={"Content-Type": "application/json"}), timeout=20).read()
    except Exception:
        pass


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=HERE / "data" / "features_tail.npz")
    ap.add_argument("--captions", type=Path, default=HERE / "data" / "captions_tail")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--negatives", type=int, default=40, help="ad-free windows, to price the false positives")
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=False)
    names = list(d["feature_names"])
    cue_cols = [i for i, n in enumerate(names) if n.startswith("cue_")]
    X, video, line, y = d["X"], d["video"], d["line"], d["y"]

    reads, negatives = [], []
    for vid in np.unique(video):
        rows = np.flatnonzero(video == vid)
        order = rows[np.argsort(line[rows])]
        flags = (X[order][:, cue_cols].sum(axis=1) > 0).astype(np.int8)
        labels = y[order]
        text = [l["text"] for l in json.loads(
            (args.captions / f"{vid}.json").read_text(encoding="utf-8"))["lines"]]
        idx = [int(line[i]) for i in order]

        for s in range(len(labels)):
            if labels[s] and (s == 0 or not labels[s - 1]):
                e = s
                while e < len(labels) and labels[e]:
                    e += 1
                lo = max(0, s - 8)
                reads.append({
                    "video": str(vid),
                    "cue": bool(flags[max(0, s - 3):e + 3].any()),
                    "lines": [text[i] for i in idx[lo:lo + WINDOW]],
                })
        # ad-free stretches, for the false-positive price
        for s in range(0, len(labels) - WINDOW, WINDOW * 3):
            if not labels[s:s + WINDOW].any():
                negatives.append({"video": str(vid), "lines": [text[i] for i in idx[s:s + WINDOW]]})

    negatives = negatives[: args.negatives]
    print(f"{len(reads)} real reads on the holdout ({sum(r['cue'] for r in reads)} caught by cue patterns), "
          f"{len(negatives)} ad-free windows\n")

    started = time.time()
    for r in reads:
        r["qwen"] = ask(args.model, r["lines"])
    false_pos = sum(1 for n in negatives if ask(args.model, n["lines"]))
    unload(args.model)

    cue_hits = sum(1 for r in reads if r["cue"])
    qwen_hits = sum(1 for r in reads if r["qwen"])
    union = sum(1 for r in reads if r["cue"] or r["qwen"])
    both = sum(1 for r in reads if r["cue"] and r["qwen"])
    only_qwen = sum(1 for r in reads if r["qwen"] and not r["cue"])
    only_cue = sum(1 for r in reads if r["cue"] and not r["qwen"])
    neither = len(reads) - union
    n = len(reads)

    print(f"=== recall on {n} reads, {(time.time() - started) / 60:.1f} min ===")
    print(f"  cue patterns alone   {cue_hits:3d}/{n}  ({100 * cue_hits / n:.1f}%)")
    print(f"  qwen3 alone          {qwen_hits:3d}/{n}  ({100 * qwen_hits / n:.1f}%)")
    print(f"  UNION of the two     {union:3d}/{n}  ({100 * union / n:.1f}%)   <- the gate is 90%")
    print()
    print(f"  both found           {both:3d}")
    print(f"  only qwen found      {only_qwen:3d}   <- what the union BUYS")
    print(f"  only cue found       {only_cue:3d}   <- what a qwen-only marker would lose")
    print(f"  neither found        {neither:3d}   <- left for the trained model")
    print()
    print(f"  qwen on ad-free windows: {false_pos}/{len(negatives)} flagged "
          f"({100 * false_pos / max(len(negatives), 1):.0f}% false positive)")
    print("\nA union only helps if the two miss DIFFERENT reads. 'only qwen' is that number;")
    print("if it is near zero the second detector is redundant and costs GPU for nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
