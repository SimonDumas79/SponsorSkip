"""Two fixes for the local model's edge placement, both from Simon's questions. (2026-09-22)

The first measurement of qwen3:8b placing edges (cascade.py --edges --local) put it at 62.0% of ad
time against Claude's 71.2%. That comparison was NOT clean, and Simon caught it: the two models got
different prompts. Claude was asked with cascade.ASK, which spells out that the start is where the
host stops talking about the subject and that the turn into the ad ("but first", "this video is
sponsored by") belongs inside. qwen was asked with qwen_edges.ASK, which just says "the FIRST line of
the promotional read". So part of the gap may be the instruction, not the model.

  --same-prompt   qwen, asked exactly what Claude was asked. Separates model from prompt.

The second is Simon's walk idea, put where it might actually fit. It was rejected for our own edge
heads because the heads already read both sides of a candidate edge at once, so a sequential walk
only threw information away. A small language model has the opposite problem: asking it for a LINE
NUMBER makes it count lines, which is exactly what small models are bad at, and qwen's behaviour fits
that -- it accepted 70 of 93 offered edges (more than Claude's 59) and placed them short, covering
only 18% of reads fully against Claude's 46%. A walk asks no arithmetic at all. It shows a small
group of lines and asks one yes/no question: is this still the advertisement? Then it steps outward.

  --walk          from inside the region, step out in groups of GROUP lines until PATIENCE groups in
                  a row come back "no". No indices, no counting, one binary judgement at a time.

What the record says qwen is good at, for context: as a DETECTOR it finds 22.7% of ad time no other
detector finds, and alone takes 70.4% of ad time at 50 s of show lost per video. It is a wide net
with poor precision -- good at finding, bad at bounding. Edges are the opposite job, which is why
this is worth checking rather than assuming.

    python marker/local_edges.py --same-prompt
    python marker/local_edges.py --walk --group 3 --patience 2
"""

import argparse
import ctypes
import json
import sys

import numpy as np
import torch

from candidate import POTION, report
from cascade import ASK as CLAUDE_ASK, parts
from edge_heads import AFTER, BEFORE, INSIDE, TAIL, place
from qwen_edges import edge_window
from replay import regions_by_video
from train import DATA

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(8)

WALK_ASK = """These numbered caption lines come from the middle of a YouTube video.

An advertisement that the host reads for a sponsor is running nearby. Look ONLY at the lines marked >>> and answer one question about them: are those lines still part of that advertisement, or are they the video's actual subject (the show)?

The advertisement includes the host's turn into it ("but first", "this video is sponsored by") and everything until the show resumes.

Answer ONLY with JSON: {"promo": true} if the marked lines are still the advertisement, {"promo": false} if they are the show."""

MAX_WALK = 30   # lines: a read does not extend further than this past where the region already reaches


def window(lines: list[dict], lo: int, hi: int, a: int, b: int) -> str:
    """The neighbourhood, with lines [a, b) marked as the ones being asked about."""
    wlo, whi = max(0, min(a, lo) - 12), min(len(lines), max(b, hi) + 12)
    out = []
    for i in range(wlo, whi):
        mark = ">>>" if a <= i < b else "   "
        out.append(f"{mark} {' '.join(lines[i]['text'].split())}")
    return "\n".join(out)


def walk_region(lines: list[dict], lo: int, hi: int, group: int, patience: int, ask) -> tuple[int, int]:
    """Step outward from the region in groups of `group` lines while the model still says 'advert'."""
    a, misses = lo, 0
    while a > 0 and lo - a < MAX_WALK:
        nxt = max(0, a - group)
        if ask(window(lines, lo, hi, nxt, a)):
            a, misses = nxt, 0
        else:
            misses += 1
            if misses >= patience:
                break
            a = nxt
    if misses:
        a = min(lo, a + misses * group)   # give back the groups the walk ate on its way to stopping
    b, misses = hi, 0
    while b < len(lines) and b - hi < MAX_WALK:
        nxt = min(len(lines), b + group)
        if ask(window(lines, lo, hi, b, nxt)):
            b, misses = nxt, 0
        else:
            misses += 1
            if misses >= patience:
                break
            b = nxt
    if misses:
        b = max(hi, b - misses * group)
    return max(0, a), min(len(lines), max(b, a + 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="holdout4")
    ap.add_argument("--budget", type=int, choices=[5, 10], default=10)
    ap.add_argument("--same-prompt", action="store_true", help="ask qwen exactly what Claude was asked")
    ap.add_argument("--walk", action="store_true", help="walk outward with yes/no questions, no line numbers")
    ap.add_argument("--group", type=int, default=3)
    ap.add_argument("--patience", type=int, default=2)
    args = ap.parse_args()
    if not (args.same_prompt or args.walk):
        raise SystemExit("choose --same-prompt or --walk")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c, T, pot, texts, bge, fs, fe = parts(args.set, dev)
    ctx, ps, pe = c.score(T, pot, bge)
    ps, pe = (ps + fs) / 2, (pe + fe) / 2
    th = c.thresholds_v2[args.budget]
    found = regions_by_video(ctx, T, th, 1)
    by_heads = place(found, T, ps, pe)
    starts = {str(v): int(np.flatnonzero(T.video == v)[0]) for v in np.unique(T.video)}
    all_lines = [{"text": t, "start": float(s)} for t, s in zip(texts, T.start_seconds)]

    from confirm_check import ask_bool, unload
    from qwen_edges import ask_edges
    tag = "walk" if args.walk else "same prompt"
    cache = DATA / f"local_edges_{args.set}_{'walk' if args.walk else 'sameprompt'}_B{args.budget}.json"
    saved = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    n_regions = sum(len(s) for s in found.values())
    print(f"{args.set}: {T.videos} videos, {T.reads} reads; {n_regions} regions at B={args.budget}. "
          f"qwen3:8b, {tag}.", flush=True)

    out, calls, done = {}, 0, 0
    try:
        for vid, spans in found.items():
            base = starts[vid]
            n = int((T.video == vid).sum())
            lines = all_lines[base:base + n]
            heads = list(by_heads.get(vid, []))
            for k, (lo, hi) in enumerate(spans):
                lo, hi = int(lo), int(hi)
                back = heads[k] if k < len(heads) else (lo, hi)
                key = f"{vid}:{lo}-{hi}"
                if key in saved:
                    out.setdefault(vid, []).append(tuple(saved[key]))
                    done += 1
                    continue
                if args.walk:
                    def ask(text: str) -> bool:
                        nonlocal calls
                        calls += 1
                        return bool(ask_bool(f"{WALK_ASK}\n\n{text}"))
                    span = walk_region(lines, lo, hi, args.group, args.patience, ask)
                else:
                    wlo, _, text = edge_window(lines, lo, hi)
                    calls += 1
                    a = ask_edges(f"{CLAUDE_ASK}\n\n{text}")
                    span = back
                    if isinstance(a, dict) and a.get("start_line") is not None and a.get("end_line") is not None:
                        s0 = min(max(int(wlo) + int(a["start_line"]), 0), n - 1)
                        e0 = min(max(int(wlo) + int(a["end_line"]), s0 + 1), n)
                        if lo - BEFORE <= s0 <= lo + INSIDE and hi - TAIL <= e0 <= hi + AFTER:
                            span = (s0, e0)
                saved[key] = list(span)
                out.setdefault(vid, []).append(tuple(span))
                done += 1
                if done % 10 == 0:
                    cache.write_text(json.dumps(saved, indent=1), encoding="utf-8")
                    print(f"  {done}/{n_regions} regions, {calls} model calls", flush=True)
    finally:
        cache.write_text(json.dumps(saved, indent=1), encoding="utf-8")
        unload()   # never leave qwen in VRAM

    print(f"  {done} regions, {calls} model calls\n", flush=True)
    report(f"B={args.budget:>2} our averaged heads", by_heads, T)
    report(f"B={args.budget:>2} qwen, {tag}", out, T)
    report(f"B={args.budget:>2} regions unplaced (raw)", found, T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
