"""Choose the cascade's detection threshold honestly, on the pooled data. (2026-09-22)

Holdout 4 showed something the shipped thresholds cannot use. With Claude placing the edges:

    B = 10's threshold: 71.2% of ad time for 4.6 s of show lost
    B =  5's threshold: 41.8% of ad time for 2.3 s

Both sit INSIDE a 5-second budget, and the looser one takes 29 more points of ad time. The reason is
that Claude narrows a region to the read, so a threshold that costs 7.9 s with our own edge heads
costs 4.6 s with Claude's. The cascade can therefore afford to look much harder for reads.

Acting on that straight from the holdout would be choosing a system by its holdout numbers, which is
the one thing this project does not do. So the threshold is chosen here, the same way every other
threshold was: on the pooled 205 videos, by the written rule (most ad time subject to the show
budget, with at most 2% of videos losing more than 60 s), using the out-of-fold stack scores.

Claude is asked once per region, and regions repeat across thresholds, so answers are cached by
region and each distinct one is paid for once.

    python marker/cascade_pooled.py --model haiku --workers 4
"""

import argparse
import ctypes
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from build_production import pooled
from cascade import ASK
from claude_label import ask_claude
from detector_bakeoff import graded, line, pick
from edge_heads import AFTER, BEFORE, INSIDE, TAIL, place
from qwen_edges import edge_window
from replay import regions_by_video

from train import DATA

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(8)

CACHE = DATA / "cascade_pooled_edges.json"


def texts_for(rows) -> list[str]:
    text = {}
    for name in ("examples.jsonl", "examples_tail.jsonl", "examples_holdout2.jsonl"):
        with open(DATA / name, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    r = json.loads(ln)
                    text[(r["videoID"], r["i"])] = r["text"]
    out, seen = [], {}
    for v in rows.video:
        seen[str(v)] = seen.get(str(v), -1) + 1
        out.append(text[(str(v), seen[str(v)])])
    return out


def place_with(found: dict, rows, answers: dict, fallback: dict) -> dict:
    """Claude's edges where it answered and the answer is near the region; the heads' otherwise."""
    out = {}
    for vid, spans in found.items():
        n = int((rows.video == vid).sum())
        heads = list(fallback.get(vid, []))
        for k, (lo, hi) in enumerate(spans):
            lo, hi = int(lo), int(hi)
            back = heads[k] if k < len(heads) else (lo, hi)
            a = answers.get(f"{vid}:{lo}-{hi}")
            if not a or a.get("start_line") is None or a.get("end_line") is None:
                out.setdefault(vid, []).append(back)
                continue
            start = min(max(int(a["wlo"]) + int(a["start_line"]), 0), n - 1)
            end = min(max(int(a["wlo"]) + int(a["end_line"]), start + 1), n)
            if lo - BEFORE <= start <= lo + INSIDE and hi - TAIL <= end <= hi + AFTER:
                out.setdefault(vid, []).append((start, end))
            else:
                out.setdefault(vid, []).append(back)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shares", type=float, nargs="*", default=None,
                    help="shares of lines to flag, i.e. how hard to look. Default: the standard sweep plus "
                         "looser settings, because Claude's narrower edges can afford them.")
    ap.add_argument("--dry", action="store_true", help="count the regions the sweep needs and stop")
    args = ap.parse_args()

    rows, is_start, is_resume = pooled()
    ft = dict(np.load(DATA / "finetune_streams.npz"))
    scores = ft["stack_bge_seed0"]
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    fs = np.load(DATA / "finetune_oof_edge_start_seed0.npy")
    fe = np.load(DATA / "finetune_oof_edge_resume_seed0.npy")
    ps, pe = (p_start + fs) / 2, (p_end + fe) / 2
    texts = texts_for(rows)
    lines = [{"text": t, "start": float(s)} for t, s in zip(texts, rows.start_seconds)]
    starts = {str(v): int(np.flatnonzero(rows.video == v)[0]) for v in np.unique(rows.video)}

    # A coarse ladder, not stack_check.SHARES' 90 points. Every threshold costs Claude calls, and the
    # full sweep needs 21,344 distinct regions; the question here is only how much harder the cascade
    # can afford to look, which seven rungs answer.
    shares = args.shares if args.shares else [0.005, 0.008, 0.012, 0.02, 0.03, 0.05, 0.08]
    ths = sorted({float(np.quantile(scores, 1 - s)) for s in shares})
    found_by_th = {th: regions_by_video(scores, rows, th, 1) for th in ths}
    jobs = {}
    for th, found in found_by_th.items():
        for vid, spans in found.items():
            for lo, hi in spans:
                jobs[f"{vid}:{int(lo)}-{int(hi)}"] = (vid, int(lo), int(hi))
    answers = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [k for k in jobs if k not in answers]
    print(f"pooled: {rows.videos} videos, {rows.reads} reads; {len(ths)} thresholds, "
          f"{len(jobs)} distinct regions, {len(todo)} still to ask claude-{args.model}", flush=True)
    if args.dry:
        return 0

    def one(key):
        vid, lo, hi = jobs[key]
        base = starts[vid]
        n = int((rows.video == vid).sum())
        vlines = lines[base:base + n]
        wlo, _, text = edge_window(vlines, lo, hi)
        a = ask_claude(ASK + "\n\n" + text, model=args.model)
        if not isinstance(a, dict) or "_error" in a:
            return key, {"wlo": int(wlo), "start_line": None, "end_line": None, "failed": True}
        return key, {"wlo": int(wlo), "start_line": a.get("start_line"), "end_line": a.get("end_line")}

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for n, (key, val) in enumerate(pool.map(one, todo), 1):
                answers[key] = val
                if n % 25 == 0:
                    CACHE.write_text(json.dumps(answers, indent=1), encoding="utf-8")
                    print(f"  {n}/{len(todo)}", flush=True)
        CACHE.write_text(json.dumps(answers, indent=1), encoding="utf-8")
    failed = sum(1 for a in answers.values() if a.get("failed"))
    print(f"  {len(answers)} answers, {failed} failed\n", flush=True)

    heads_cands, cascade_cands = [], []
    for th, found in found_by_th.items():
        by_heads = place(found, rows, ps, pe)
        heads_cands.append((th, by_heads))
        cascade_cands.append((th, place_with(found, rows, answers, by_heads)))
    g_heads, g_casc = graded(heads_cands, rows), graded(cascade_cands, rows)
    for b in (5, 10):
        print(line(f"pooled CV B={b:>2} our averaged heads", pick(g_heads, b)))
        print(line(f"pooled CV B={b:>2} claude edges (cascade)", pick(g_casc, b)))
    chosen = {b: pick(g_casc, b) for b in (5, 10)}
    print("\nthresholds the rule chooses FOR THE CASCADE, on pooled CV:")
    for b in (5, 10):
        if chosen[b]:
            print(f"  B={b:>2}: {chosen[b][0]:.4f}")
    out = DATA / "cascade_thresholds.json"
    out.write_text(json.dumps({str(b): (chosen[b][0] if chosen[b] else None) for b in (5, 10)}, indent=1),
                   encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
