"""Is the stacked context model's gain real? Seeds, a read-by-read test, and which inputs carry it. (2026-09-21)

detector_bakeoff.py found ONE context model reading every detector's scores (marker, meaning-only,
structure-only, potion, sequence, plus the cue and description columns) skipping 49.7% of ad time
at 6.3 s of lost show per video, against 41.7% for the shipped free tier on the same threshold grid.
Every lever before it landed inside ~2 points, so this checks it three ways before anyone believes it:

  seeds      the context model retrained from three random starts (same folds), shipped and stacked
  paired     read by read at B = 10: how many reads does the stack skip more of, how many less (sign test)
  ablation   which inputs carry the gain, looking for the smallest stack that keeps it

A finer threshold grid than the bake-off's (the grid alone moved the shipped tier 41.7 -> 43.4%).
Cross-validated on the pooled 205 videos only.

    python marker/stack_check.py          # needs data/bakeoff_streams.npz from detector_bakeoff.py
"""

import ctypes
import math
import sys

import numpy as np
import torch

from build_production import pooled
from context_stack import context_features, fit_stage2
from detector_bakeoff import CACHE, graded, line, pick
from edge_heads import place
from replay import per_read, regions_by_video
from train import DATA

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(4)

SHARES = np.geomspace(0.004, 0.25, 90)
SEEDS = (0, 1, 2)


def oof_seeded(rows, F: np.ndarray, seed: int, folds: int = 5) -> np.ndarray:
    """context_stack.out_of_fold with the model's random start as a parameter; the folds stay seed 0."""
    channels = np.array(sorted(set(rows.channel)))
    np.random.default_rng(0).shuffle(channels)
    p = np.zeros(len(rows), dtype=np.float32)
    for fold in np.array_split(channels, folds):
        test = np.isin(rows.channel, fold)
        p[test] = fit_stage2(F[~test], rows.y[~test].astype(np.float32), hidden=32, seed=seed)(F[test])
    return p


def sweep_fine(scores, rows, p_start, p_end) -> list:
    return [(float(th), place(regions_by_video(scores, rows, float(th), 1), rows, p_start, p_end))
            for th in np.unique(np.quantile(scores, 1 - SHARES))]


def sign_test(better: int, worse: int) -> float:
    """Two-sided binomial test of better vs worse, ties dropped."""
    n, k = better + worse, max(better, worse)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> int:
    rows, _, _ = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    cache = dict(np.load(CACHE))
    streams = {"marker": level1, "meaning": cache["meaning"], "structure": cache["structure"],
               "potion": cache["potion"], "sequence": np.load(DATA / "sequence_oof_h32_r7.npy")}
    names = [str(n) for n in rows.feature_names]
    extra = rows.X[:, [i for i, n in enumerate(names) if n.startswith("cue_") or n.startswith("desc_")]]
    ctx = {n: context_features(rows, s) for n, s in streams.items()}
    print(f"pooled: {rows.videos} videos, {rows.reads} reads\n")

    def inputs(parts: list[str], raw: bool) -> np.ndarray:
        return np.column_stack([ctx[p] for p in parts] + ([extra] if raw else []))

    everything = ["marker", "meaning", "structure", "potion", "sequence"]
    results = {}

    print("SEEDS (same folds, three random starts of the context model), shipped heads, pooled rule:")
    for label, F in (("shipped: marker only", inputs(["marker"], False)),
                     ("stack: every detector + cues + description", inputs(everything, True))):
        for seed in SEEDS:
            key = f"seed{seed}_{label[:5]}"
            p = level2 if (seed == 0 and label.startswith("shipped")) else (
                cache["ctx_stack"] if seed == 0 else oof_seeded(rows, F, seed))
            g = graded(sweep_fine(p, rows, p_start, p_end), rows)
            results[key] = g
            for budget in (5, 10):
                print(line(f"B={budget:>2} {label[:34]} seed {seed}", pick(g, budget)))

    print("\nREAD BY READ at B = 10, stack vs shipped (seed 0): share of each read's time skipped")
    b_ship, b_stack = pick(results["seed0_shipp"], 10), pick(results["seed0_stack"], 10)
    a = np.array([s for *_, s in per_read(b_ship[1], rows)])
    b = np.array([s for *_, s in per_read(b_stack[1], rows)])
    better, worse = int(np.sum(b > a + 0.05)), int(np.sum(b < a - 0.05))
    print(f"  the stack skips more of {better} reads, less of {worse}, about the same on {len(a) - better - worse}; "
          f"sign test p = {sign_test(better, worse):.4f}")
    print(f"  reads the stack touches that the shipped tier misses entirely: {int(np.sum((b > 0) & (a == 0)))}; "
          f"the reverse: {int(np.sum((a > 0) & (b == 0)))}")

    print("\nABLATION (seed 0): which inputs carry it")
    variants = [
        ("marker + cue/description columns", ["marker"], True),
        ("marker + structure", ["marker", "structure"], False),
        ("marker + potion", ["marker", "potion"], False),
        ("marker + sequence", ["marker", "sequence"], False),
        ("marker + meaning + structure (one encoder)", ["marker", "meaning", "structure"], False),
        ("every detector, no raw columns", everything, False),
        ("every detector except potion, + raw columns", ["marker", "meaning", "structure", "sequence"], True),
        ("every detector except sequence, + raw columns", ["marker", "meaning", "structure", "potion"], True),
    ]
    for label, parts, raw in variants:
        print(f"  training: {label}...", flush=True)
        g = graded(sweep_fine(oof_seeded(rows, inputs(parts, raw), 0), rows, p_start, p_end), rows)
        for budget in (5, 10):
            print(line(f"B={budget:>2} {label[:44]}", pick(g, budget)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
