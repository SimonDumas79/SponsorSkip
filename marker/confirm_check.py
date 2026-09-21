"""Measure the "local model confirms" tier: qwen3 checks each window the marker flags.

The tier in one sentence: the marker (free, offline) flags candidate regions at a
threshold loose enough to catch 90% of reads; a local language model then reads
a short caption window around each flag and says whether a promotional read is
really there. False alarms that it rejects cost the viewer nothing; the show is
skipped only where both agree.

What this script does
  1. Scores every row of data/features.npz out-of-fold (five channel-grouped
     folds of the linear marker), so each flag comes from a model that never
     saw the video's channel.
  2. Picks the checker threshold by the usual rule (highest threshold keeping
     recall >= 90%), takes every flagged region, and cuts a window of caption
     lines around it: the region plus CONTEXT lines either side, capped at
     MAX_LINES. The window is centred on what the MARKER found, never on the
     true read, so the test measures what the tier would do live.
  3. Asks qwen3:8b twice per window: for a yes/no answer, and for the first
     token's probability of "Yes" (so a cut-off can be chosen after the fact).
  4. RECORDS every verdict, with the window's identity, the marker's score,
     and whether the window really held a read, to data/confirm_verdicts.jsonl.
     That file is the replay log: any future routing rule (keep if p >= x,
     trust the marker above y, ...) can be scored from it without running qwen
     again. A finished window is never asked again, so a stopped run resumes.
  5. Reports, for a range of cut-offs on P(yes): reads kept and lost, false
     windows rejected, show saved per video, and the precision of what remains.

The model is unloaded from the GPU when the run ends, whatever happens.
"""

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch

from score import score_video, smooth
from train import DATA, SLACK, load
from experiments import BASE, canonical, cross_validate, fit_full, predict_full, runs

OLLAMA = "http://localhost:11434"
MODEL = "qwen3:8b"
CONTEXT = 8      # caption lines of run-up and run-out around the flagged region
MAX_LINES = 70   # about three minutes of captions; more than that and the read is a needle
VERDICTS = DATA / "confirm_verdicts.jsonl"

ASK = """You are given numbered caption lines from part of a video.

Is there a PROMOTIONAL READ in these lines? That means either:
  - a sponsor read: an advertisement the host reads out for a company that paid them, or
  - self-promotion: the host pitching their own app, course, merch, membership, Patreon or event.
Either one interrupts the video's actual subject, and the subject resumes afterwards.

NOT a promotional read: a review, unboxing or demonstration of a product that IS the video's
subject; a plain "like and subscribe"; mentioning a previous video.

Answer ONLY with JSON: {"promo": true} or {"promo": false}"""

SCHEMA = {"type": "object", "properties": {"promo": {"type": "boolean"}}, "required": ["promo"]}


def ask_bool(prompt: str, timeout: float = 120.0) -> bool | None:
    body = json.dumps({
        "model": MODEL, "stream": False, "think": False, "format": SCHEMA, "keep_alive": "300s",
        "options": {"num_ctx": 4096, "temperature": 0},
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return bool(json.loads(json.load(r)["message"]["content"])["promo"])
    except (urllib.error.URLError, KeyError, ValueError, TimeoutError):
        return None


def ask_p_yes(prompt: str, timeout: float = 120.0) -> float | None:
    """P(Yes) against P(No) on the first generated token."""
    body = json.dumps({
        "model": MODEL, "prompt": prompt + "\n\nAnswer with one word, Yes or No.", "stream": False, "think": False,
        "logprobs": True, "top_logprobs": 12, "keep_alive": "300s",
        "options": {"temperature": 0, "num_predict": 1, "num_ctx": 4096},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            top = json.load(r)["logprobs"][0]["top_logprobs"]
    except (urllib.error.URLError, KeyError, IndexError, ValueError, TimeoutError):
        return None
    yes = sum(math.exp(e["logprob"]) for e in top if e["token"].strip().lower().startswith("yes"))
    no = sum(math.exp(e["logprob"]) for e in top if e["token"].strip().lower().startswith("no"))
    return yes / (yes + no) if yes + no > 0 else None


def unload() -> None:
    """Never leave a model sitting in VRAM."""
    body = json.dumps({"model": MODEL, "keep_alive": 0}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(f"{OLLAMA}/api/generate", data=body,
                                                      headers={"Content-Type": "application/json"}), timeout=20).read()
    except Exception:
        pass


def checker_threshold(oof: np.ndarray, rows, gate: float, smooth_w: int) -> float:
    best = None
    for th in np.round(np.arange(0.50, 1.00, 0.005), 3):
        found = total = 0
        for vid in np.unique(rows.video):
            r = np.flatnonzero(rows.video == vid)
            f, t, _, _ = score_video(rows.y[r], smooth((oof[r] >= th).astype(np.int8), smooth_w), SLACK)
            found += f
            total += t
        if found / max(total, 1) >= gate:
            best = float(th)
    return best


def windows(oof: np.ndarray, rows, threshold: float, captions_dir: Path, smooth_w: int) -> list[dict]:
    """One record per flagged region, with the caption text the checker will see."""
    out = []
    for vid in np.unique(rows.video):
        r = np.flatnonzero(rows.video == vid)
        caps = json.loads((captions_dir / f"{vid}.json").read_text(encoding="utf-8"))
        lines = caps["lines"]
        flags = smooth((oof[r] >= threshold).astype(np.int8), smooth_w)
        labels = rows.y[r]
        for lo, hi in runs(flags):
            wlo, whi = max(0, lo - CONTEXT), min(len(lines), hi + CONTEXT)
            if whi - wlo > MAX_LINES:   # keep the window centred on the region
                mid = (lo + hi) // 2
                wlo, whi = max(0, mid - MAX_LINES // 2), min(len(lines), mid + MAX_LINES // 2)
            true_inside = bool(labels[max(0, lo - SLACK):hi + SLACK].any())
            out.append({
                "id": f"{vid}:{lo}-{hi}", "video": vid, "channel": str(rows.channel[r][0]),
                "lo": int(lo), "hi": int(hi), "start": float(rows.start_seconds[r][lo]),
                "seconds": float(rows.start_seconds[r][min(hi, len(r) - 1)] - rows.start_seconds[r][lo]),
                "marker_peak": float(oof[r][lo:hi].max()),
                "is_read": true_inside,
                "text": "\n".join(f"{i - wlo:3d}: {lines[i]['text']}" for i in range(wlo, whi)),
            })
    return out


def report(verdicts: list[dict], n_videos: int, reads_total: int) -> None:
    reads = {v["id"] for v in verdicts if v["is_read"]}
    print(f"\n{len(verdicts)} flagged windows on {n_videos} videos: {len(reads)} hold a read, "
          f"{len(verdicts) - len(reads)} are false alarms; {reads_total} reads in total")
    print("  rule              windows kept  reads kept  false kept  show saved/video  precision")
    rules = [("marker alone", lambda v: True), ("qwen says yes", lambda v: v["said"] is True)]
    rules += [(f"P(yes) >= {c:.2f}", (lambda c: lambda v: v["p_yes"] is not None and v["p_yes"] >= c)(c))
              for c in (0.3, 0.5, 0.7, 0.9)]
    for name, keep in rules:
        kept = [v for v in verdicts if keep(v)]
        kept_reads = {v["id"] for v in kept if v["is_read"]}
        false_kept = sum(1 for v in kept if not v["is_read"])
        saved = sum(v["seconds"] for v in verdicts if not v["is_read"] and not keep(v)) / max(n_videos, 1)
        print(f"  {name:<17} {len(kept):12d}  {len(kept_reads):3d} / {len(reads):<3d}  {false_kept:10d}  "
              f"{saved:14.0f} s  {len(kept_reads) / max(len(kept), 1):8.1%}")


def trained_scores(cfg: dict, train_path: Path, rows) -> np.ndarray:
    """Scores from one marker trained on EVERY row of train_path: how the holdout is graded."""
    return predict_full(fit_full(cfg, load(train_path)), rows.X)


def main() -> int:
    global VERDICTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=DATA / "features.npz")
    ap.add_argument("--captions", type=Path, default=DATA / "captions")
    ap.add_argument("--train-on", type=Path,
                    help="score --features with a marker trained on every row of this file instead of "
                         "out-of-fold (use for the holdout: --features features_tail.npz --train-on features.npz)")
    ap.add_argument("--verdicts", type=Path, default=VERDICTS)
    ap.add_argument("--config", type=Path, help="a zoo candidate's config.json (default: the baseline marker)")
    ap.add_argument("--reuse", type=Path, nargs="*", default=[],
                    help="other verdict files: a window with the same id there has the same text, so its answer is reused")
    ap.add_argument("--no-logprob", action="store_true", help="ask only for the yes/no (half the GPU time)")
    ap.add_argument("--gate", type=float, default=0.90)
    ap.add_argument("--threshold", type=float, help="skip the rule and use this checker threshold")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new windows (0 = all)")
    ap.add_argument("--report-only", action="store_true", help="replay the recorded verdicts, ask nothing")
    args = ap.parse_args()
    VERDICTS = args.verdicts

    torch.set_num_threads(4)
    cfg = canonical(json.loads(args.config.read_text(encoding="utf-8"))) if args.config else dict(BASE)
    rows = load(args.features)
    if args.train_on:
        if args.threshold is None:
            raise SystemExit("--train-on needs --threshold: a holdout threshold must be chosen on cross-validation")
        oof = trained_scores(cfg, args.train_on, rows)
    else:
        oof = cross_validate(cfg, rows, seed=0)
    threshold = args.threshold or checker_threshold(oof, rows, args.gate, cfg["smooth"])
    # Everything replay.py needs to rebuild these exact regions without training anything.
    np.save(VERDICTS.with_suffix(".scores.npy"), oof)
    VERDICTS.with_suffix(".manifest.json").write_text(json.dumps({
        "config": cfg, "threshold": threshold, "features": str(args.features), "captions": str(args.captions),
        "train_on": str(args.train_on) if args.train_on else None, "scores": str(VERDICTS.with_suffix(".scores.npy")),
    }, indent=1), encoding="utf-8")
    wins = windows(oof, rows, threshold, args.captions, cfg["smooth"])
    print(f"checker threshold {threshold}: {len(wins)} flagged windows on {rows.videos} videos "
          f"({len(wins) / rows.videos:.1f} per video), {sum(w['is_read'] for w in wins)} of them real")

    done = {}
    if VERDICTS.exists():
        for line in VERDICTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["id"]] = rec
    reused = 0
    for path in args.reuse:   # same id = same lines = same prompt, so the same answer
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec["id"] not in done and rec.get("said") is not None:
                    done[rec["id"]] = {"id": rec["id"], "said": rec["said"], "p_yes": rec.get("p_yes"),
                                       "model": rec.get("model", MODEL), "reused_from": path.name}
                    reused += 1
    if reused:
        print(f"{reused} answers available from --reuse files")
    todo = [w for w in wins if w["id"] not in done]
    if args.report_only:
        todo = []
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} verdicts recorded, {len(todo)} to ask {MODEL}", flush=True)

    t0 = time.time()
    try:
        with VERDICTS.open("a", encoding="utf-8") as f:
            for n, w in enumerate(todo, 1):
                prompt = f"{ASK}\n\n{w['text']}"
                t1 = time.time()
                said = ask_bool(prompt)
                p_yes = None if args.no_logprob else ask_p_yes(prompt)
                rec = {k: v for k, v in w.items() if k != "text"}
                rec.update({"said": said, "p_yes": p_yes, "model": MODEL, "seconds_taken": round(time.time() - t1, 1)})
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[rec["id"]] = rec
                if n % 10 == 0 or n == len(todo):
                    rate = (time.time() - t0) / n
                    print(f"  [{n}/{len(todo)}] {rate:.1f} s per window, ~{rate * (len(todo) - n) / 60:.0f} min left",
                          flush=True)
    finally:
        unload()

    # The window's own fields (is_read, seconds, marker_peak) come from THIS run; only the answer is cached.
    verdicts = [dict({k: v for k, v in w.items() if k != "text"}, **{k: done[w["id"]][k] for k in ("said", "p_yes")})
                for w in wins if w["id"] in done]
    report(verdicts, rows.videos, rows.reads)
    if len(verdicts) < len(wins):
        print(f"\n({len(wins) - len(verdicts)} windows not yet asked; run again to finish)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
