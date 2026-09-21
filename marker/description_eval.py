"""Does the description signal help the free tier? The same pipeline with and without the two columns, on CV.

Three systems, each rebuilt out of fold with the marker's channel folds and chosen by
system_eval.py's rule (most ad time under B s of show lost per video, no video over 60 s):
  A  the marker on the 395 original columns            (= the shipped free tier)
  B  the marker on 397 columns, description included
  C  as B, and the context model ALSO sees the two description columns directly
Reported with the short-read buckets, because that is where the signal should land.

    python marker/description_eval.py             # features.npz (CV set)
    python marker/description_eval.py --pooled    # the pooled 205-video set, build_production's rule
"""

import argparse

import numpy as np

from context_stack import context_features, out_of_fold
from edge_heads import edge_oof, place, soft
from experiments import BASE, cross_validate
from replay import HEADER, grade_regions, regions_by_video
from short_reads import CTX_THRESHOLDS, bucket_line, choose, coverage_by_bucket, low_priority, read_table
from train import DATA, Rows, load


def system(rows: Rows, is_start, is_resume, columns: int, ctx_sees_desc: bool):
    sub = Rows(rows.X[:, :columns], rows.y, rows.video, rows.channel, rows.start_seconds, rows.split,
               rows.feature_names[:columns])
    level1 = cross_validate({**BASE, "use_description": columns > 395}, sub, seed=0)
    F = context_features(sub, level1)
    if ctx_sees_desc:
        F = np.column_stack([F, rows.X[:, 395:397]])
    level2 = out_of_fold(sub, F, hidden=32)
    p_start = edge_oof(sub, F, soft(is_start, rows.video))
    p_end = edge_oof(sub, F, soft(is_resume, rows.video))
    return level1, level2, p_start, p_end


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pooled", action="store_true")
    args = ap.parse_args()
    low_priority()

    over_cap = None
    if args.pooled:
        from build_production import over_cap, pooled
        rows, is_start, is_resume = pooled()
        class D(dict):
            def __getitem__(self, k):
                return np.array([""] * len(rows)) if k == "category" else super().__getitem__(k)
        d = D()
    else:
        rows, d = load(DATA / "features.npz"), np.load(DATA / "features.npz")
        is_start, is_resume = d["is_start"], d["is_resume"]
    if rows.X.shape[1] < 397:
        raise SystemExit("no description columns: run fetch_descriptions.py then add_descriptions.py")
    reads = read_table(rows, d)
    hit = rows.X[:, 395]
    print(f"{rows.videos} videos, {len(reads)} reads; description hit lines: {int(hit.sum())}, "
          f"{int((hit * rows.y).sum())} inside a read ({(hit * rows.y).sum() / max(hit.sum(), 1):.0%}); "
          f"reads with a hit line: {sum(1 for rd in reads if hit[rd['rows'][rd['lo']:rd['hi']]].any())}")
    print(HEADER)
    for name, columns, ctx in (("A 395 columns (shipped)", 395, False), ("B +description in marker", 397, False),
                               ("C +description in marker AND context", 397, True)):
        print(f"  building {name}...", flush=True)
        level1, level2, p_start, p_end = system(rows, is_start, is_resume, columns, ctx)
        g1 = grade_regions(regions_by_video(level1, rows, 0.9692, 1), rows)
        print(f"      marker alone at 0.9692: {g1['coverage']:.1%} ad time / {g1['show']:.1f} s; "
              f"{bucket_line(coverage_by_bucket(regions_by_video(level1, rows, 0.9692, 1), rows, reads))}")
        for budget in (5, 10):
            cands = [(f"th {t}", place(regions_by_video(level2, rows, t, 1), rows, p_start, p_end))
                     for t in CTX_THRESHOLDS]
            choose(name, cands, rows, reads, budget, over_cap=over_cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
