"""Would more of the same data help? Train the free tier on a fraction of the channels and watch the curve.

The pooled set (205 videos, 280 reads) is subsampled BY CHANNEL to 25, 50, 75 and
100%, three draws each. For every subset the whole free tier is rebuilt out of
fold (marker, context model, edge heads, all with channel folds) and graded by
build_production.py's rule. If the curve is still rising at 100%, more crawl data
of the same kind is worth its wait; if it is flat, it is not, and the data worth
having is a different kind (hard negatives, labels owing nothing to SponsorBlock).

Cross-validation only; no holdout is touched. About 15 minutes on the CPU.

    python marker/learning_curve.py
"""

import numpy as np
import torch

from build_production import OVER_CAP_SHARE, over_cap, pooled
from context_stack import context_features, out_of_fold
from edge_heads import edge_oof, place, soft
from experiments import BASE, cross_validate
from replay import grade_regions, regions_by_video
from short_reads import CTX_THRESHOLDS, low_priority
from train import Rows

FRACTIONS = (0.25, 0.5, 0.75, 1.0)
DRAWS = (0, 1, 2)


def subset(rows: Rows, is_start, is_resume, fraction: float, seed: int):
    channels = np.array(sorted(set(rows.channel)))
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(channels, size=max(5, int(round(len(channels) * fraction))), replace=False).tolist())
    m = np.isin(rows.channel, list(keep))
    return rows.subset(m), is_start[m], is_resume[m]


def free_tier_grade(rows: Rows, is_start, is_resume, budget: float) -> dict | None:
    level1 = cross_validate(BASE, rows, seed=0)
    F = context_features(rows, level1)
    level2 = out_of_fold(rows, F, hidden=32)
    p_start = edge_oof(rows, F, soft(is_start, rows.video))
    p_end = edge_oof(rows, F, soft(is_resume, rows.video))
    best = None
    for th in CTX_THRESHOLDS:
        kept = place(regions_by_video(level2, rows, th, 1), rows, p_start, p_end)
        g = grade_regions(kept, rows)
        if g["show"] <= budget and over_cap(kept, rows) <= OVER_CAP_SHARE and (
                best is None or g["coverage"] > best["coverage"]):
            best = g
    return best


def main() -> int:
    low_priority()
    rows, is_start, is_resume = pooled()
    print(f"pooled: {rows.videos} videos, {len(set(rows.channel))} channels, {rows.reads} reads; "
          f"rule: most ad time under B s lost per video, at most {OVER_CAP_SHARE:.0%} of videos over 60 s")
    print(f"{'fraction':>8} {'channels':>8} {'reads':>6}   B=5: ad time / show     B=10: ad time / show")
    for fraction in FRACTIONS:
        results = []
        for seed in DRAWS if fraction < 1.0 else (0,):
            sub, st, rs = subset(rows, is_start, is_resume, fraction, seed)
            g5, g10 = free_tier_grade(sub, st, rs, 5), free_tier_grade(sub, st, rs, 10)
            results.append((len(set(sub.channel)), sub.reads, g5, g10))
        fmt = lambda g: f"{g['coverage']:5.1%} / {g['show']:4.1f} s" if g else "   none under budget"
        for ch, reads, g5, g10 in results:
            print(f"{fraction:>8.0%} {ch:>8} {reads:>6}   {fmt(g5):<22}  {fmt(g10)}", flush=True)
        cov10 = [r[3]["coverage"] for r in results if r[3]]
        if len(cov10) > 1:
            print(f"{'':>8} mean at B=10: {np.mean(cov10):.1%} (spread {np.min(cov10):.1%}-{np.max(cov10):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
