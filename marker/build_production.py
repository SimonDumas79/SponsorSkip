"""Train the shipped free tier on EVERY labelled video, now that both holdouts have been graded.

A holdout is only a test until it has been looked at. Both have been, once each
(see system_eval.py), so the most useful thing they can do now is teach: this
script pools training, holdout 1 and holdout 2 (three channel-disjoint sets),
re-runs the same channel-grouped cross-validation over all of it, chooses the
context model's threshold, and saves the result as free_tier_pooled.pt.

The rule differs from system_eval.py's in one place, decided on 2026-09-21 after
seeing the pooled curve: "no video over 60 s" is a MAXIMUM over videos, so it
tightens as videos are added, and on 205 videos it forced a threshold that skips
less ad time than the marker alone. Here the cap is a share: at most 2% of videos
may lose more than 60 s of real show.

free_tier.pt (what predict.py and the server load by default) stays the graded
system, the one with holdout numbers on record. The pooled bundle is the
candidate to test on the next fresh videos; copy it over free_tier.pt only after
it passes. The numbers printed here are cross-validated estimates, not a grade.

    python marker/build_production.py
"""

import argparse

import numpy as np
import torch

from context_stack import context_features, fit_stage2, out_of_fold
from edge_heads import edge_oof, place, soft
from experiments import BASE, cross_validate, fit_full
from features import MODEL_NAME
from predict import BUNDLE, stage_state
from replay import HEADER, grade_regions, regions_by_video, row
from train import DATA, FEATURES, Rows, load

SETS = ["features.npz", "features_tail.npz", "features_holdout2.npz"]
POOLED = BUNDLE.parent / "free_tier_pooled.pt"
WORST_CAP = 60.0
OVER_CAP_SHARE = 0.02   # at most this share of videos may lose more than WORST_CAP seconds


def pooled() -> tuple[Rows, np.ndarray, np.ndarray]:
    parts = [load(DATA / f) for f in SETS]
    raw = [np.load(DATA / f) for f in SETS]
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            shared = set(parts[i].channel) & set(parts[j].channel)
            if shared:
                raise SystemExit(f"{SETS[i]} and {SETS[j]} share channels: {sorted(shared)[:5]}")
    cat = lambda name: np.concatenate([getattr(p, name) for p in parts])
    rows = Rows(cat("X"), cat("y"), cat("video"), cat("channel"), cat("start_seconds"), cat("split"),
                parts[0].feature_names)
    return rows, np.concatenate([d["is_start"] for d in raw]), np.concatenate([d["is_resume"] for d in raw])


def over_cap(kept: dict, rows) -> float:
    """The share of videos that lose more than WORST_CAP seconds of real show."""
    from train import LAST_LINE_SECONDS
    over = 0
    for vid in np.unique(rows.video):
        r = np.flatnonzero(rows.video == vid)
        flags = np.zeros(len(r), dtype=np.int8)
        for a, b in kept.get(str(vid), []):
            flags[a:b] = 1
        s = rows.start_seconds[r]
        on_screen = np.diff(np.append(s, s[-1] + LAST_LINE_SECONDS))
        over += on_screen[(flags == 1) & (rows.y[r] == 0)].sum() > WORST_CAP
    return over / rows.videos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    torch.set_num_threads(4)

    rows, is_start, is_resume = pooled()
    print(f"pooled: {rows.videos} videos, {len(set(rows.channel))} channels, {rows.reads} reads")
    cache = DATA / "pooled_oof.npy"   # the four out-of-fold score sets; deterministic, so reusable
    if cache.exists() and np.load(cache).shape[1] == len(rows):
        level1, level2, p_start, p_end = np.load(cache)
        F = context_features(rows, level1)
    else:
        level1 = cross_validate(BASE, rows, seed=0)
        F = context_features(rows, level1)
        level2 = out_of_fold(rows, F, hidden=32)
        p_start = edge_oof(rows, F, soft(is_start, rows.video))
        p_end = edge_oof(rows, F, soft(is_resume, rows.video))
        np.save(cache, np.stack([level1, level2, p_start, p_end]))

    print(f"\ncross-validated on all {rows.videos} videos (estimates, not a holdout grade):")
    print(HEADER)
    chosen = {}
    for budget in (5, 10):
        best_alone = best = None
        for th in np.round(1 / (1 + np.exp(-np.arange(0, 9, 0.05))), 4):
            g = grade_regions(regions_by_video(level1, rows, th, 1), rows)
            if g["show"] <= budget and over_cap(regions_by_video(level1, rows, th, 1), rows) <= OVER_CAP_SHARE and (
                    best_alone is None or g["coverage"] > best_alone[1]["coverage"]):
                best_alone = (float(th), g)
        for th in np.round(1 / (1 + np.exp(-np.arange(2.0, 6.0, 0.1))), 4):
            kept = place(regions_by_video(level2, rows, th, 1), rows, p_start, p_end)
            g = grade_regions(kept, rows)
            if g["show"] <= budget and over_cap(kept, rows) <= OVER_CAP_SHARE and (
                    best is None or g["coverage"] > best[1]["coverage"]):
                best = (float(th), g)
        if best_alone:
            print(row(f"B={budget}: marker alone th {best_alone[0]}", best_alone[1]))
        if best:
            chosen[budget] = best[0]
            print(row(f"B={budget}: free tier th {best[0]}", best[1]))
    if not chosen:
        raise SystemExit("no threshold met the rule; nothing exported")
    threshold = chosen.get(10, chosen.get(5))

    marker = fit_full(BASE, rows, seed=0)
    stages = {"context": fit_stage2(F, rows.y.astype(np.float32), hidden=32),
              "start": fit_stage2(F, soft(is_start, rows.video), hidden=32),
              "end": fit_stage2(F, soft(is_resume, rows.video), hidden=32)}
    torch.save({
        "marker": [{"columns": torch.from_numpy(p["columns"]), "mean": torch.from_numpy(p["mean"]),
                    "std": torch.from_numpy(p["std"]), "state": p["model"].state_dict()} for p in marker],
        "stages": {name: stage_state(s) for name, s in stages.items()},
        "context_inputs": F.shape[1], "threshold": threshold, "features": FEATURES, "encoder": MODEL_NAME,
        "trained_on": f"{' + '.join(SETS)}: {rows.videos} videos, {rows.reads} reads",
    }, POOLED)
    print(f"\nsaved {POOLED} (threshold {threshold}); free_tier.pt stays the graded system")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
