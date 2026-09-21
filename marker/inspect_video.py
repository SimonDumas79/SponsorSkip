"""Print what the free tier would skip in one CV video at a threshold, region by region, with the caption text.

For reading the outliers by eye: is a "false" region real show, or a promotion SponsorBlock never marked?

    python marker/inspect_video.py bqtppv75MJg --threshold 0.9677
"""

import argparse
import json

import numpy as np

from edge_heads import place
from experiments import runs
from replay import regions_by_video
from train import DATA, LAST_LINE_SECONDS, load


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--threshold", type=float, default=0.9781)
    ap.add_argument("--chars", type=int, default=260)
    args = ap.parse_args()
    rows = load(DATA / "features.npz")
    level2 = np.load(DATA / "context_stack_oof.npy")
    p_start, p_end = np.load(DATA / "edge_heads_oof.npy")
    text = {}
    with (DATA / "examples.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if args.video in line:
                rec = json.loads(line)
                if rec["videoID"] == args.video:
                    text[rec["i"]] = rec["text"]
    r = np.flatnonzero(rows.video == args.video)
    if not len(r):
        raise SystemExit(f"{args.video} is not in features.npz")
    s = rows.start_seconds[r]
    on_screen = np.diff(np.append(s, s[-1] + LAST_LINE_SECONDS))
    y = rows.y[r]
    print(f"{args.video}: {len(r)} lines, {on_screen.sum() / 60:.1f} min, {on_screen[y == 1].sum():.0f} s labelled ad")
    for a, b in runs(y):
        print(f"  LABELLED READ {s[a]:7.0f}-{s[b - 1] + on_screen[b - 1]:7.0f} s ({on_screen[a:b].sum():.0f} s): "
              f"{' '.join(text.get(i, '') for i in range(a, min(b, a + 6)))[:args.chars]}")
    kept = place(regions_by_video(level2, rows, args.threshold, 1), rows, p_start, p_end).get(args.video, [])
    print(f"\nfree tier at {args.threshold}: {len(kept)} regions")
    for lo, hi in kept:
        lost = on_screen[lo:hi][y[lo:hi] == 0].sum()
        print(f"  {s[lo]:7.0f}-{s[hi - 1] + on_screen[hi - 1]:7.0f} s  {on_screen[lo:hi].sum():4.0f} s, "
              f"{lost:4.0f} s unlabelled, peak ctx {level2[r][lo:hi].max():.3f}")
        print(f"      {' '.join(text.get(i, '') for i in range(lo, hi))[:args.chars]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
