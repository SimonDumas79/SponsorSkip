"""Hard-negative mining: weight the show lines the best system wrongly flags, and retrain. (2026-09-22)

All three method rooms ranked this first among builds: the false alarms that pin the threshold are
product talk, so make the model pay more for them. The best system (the stack context model over
five detectors + fine-tuned MiniLM) is run out of fold at a LOOSE threshold (flagging LOOSE_SHARE of
lines) so there are enough mistakes to learn from; every show line inside a region that overlaps no
read gets weight W in the context model's loss (the rest weight 1), and the context model is
retrained out of fold with the same folds. W is chosen by the pooled rule like any threshold.

Caveat, on the record: the regions come from out-of-fold scores, and a training row's out-of-fold
score was made by a model that saw the current test fold, so the weights carry a trace of it.
Nothing about the test fold's own rows is used.

The rooms' gate: report fixed and newly broken false alarms separately (the (b, c) pair), and treat
under ~13 of 18 fixed as not established.

    python marker/hardneg.py
"""

import ctypes
import sys

import numpy as np
import torch

from build_production import pooled
from context_stack import context_features
from detector_bakeoff import CACHE, graded, line, pick
from replay import regions_by_video
from short_reads import fit_weighted
from stack_check import sweep_fine
from train import DATA

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(8)
LOOSE_SHARE = 0.10
WEIGHTS = [3.0, 10.0]


def false_regions(kept: dict, rows, idx) -> set:
    return {(v, lo, hi) for v, spans in kept.items() for lo, hi in spans if not rows.y[idx[v]][lo:hi].any()}


def main() -> int:
    rows, _, _ = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    bake = dict(np.load(CACHE))
    ft = dict(np.load(DATA / "finetune_streams.npz"))
    idx = {str(v): np.flatnonzero(rows.video == v) for v in np.unique(rows.video)}
    base = ft["stack_ft0"]
    th = float(np.quantile(base, 1 - LOOSE_SHARE))
    loose = regions_by_video(base, rows, th, 1)
    hard = np.zeros(len(rows), dtype=bool)
    for v, lo, hi in false_regions(loose, rows, idx):
        hard[idx[v][lo:hi]] = True
    print(f"pooled: {rows.videos} videos; at the loose threshold ({LOOSE_SHARE:.0%} of lines) "
          f"{len(false_regions(loose, rows, idx))} false regions, {int(hard.sum()):,} hard-negative lines\n")

    names = [str(n) for n in rows.feature_names]
    extra = rows.X[:, [i for i, n in enumerate(names) if n.startswith("cue_") or n.startswith("desc_")]]
    parts = [level1, bake["meaning"], bake["structure"], bake["potion"], np.load(DATA / "sequence_oof_h32_r7.npy"),
             np.load(DATA / "finetune_oof_seed0.npy")]
    F = np.column_stack([context_features(rows, p) for p in parts] + [extra])
    channels = np.array(sorted(set(rows.channel)))
    np.random.default_rng(0).shuffle(channels)

    g_base = graded(sweep_fine(base, rows, p_start, p_end), rows)
    results = {"none (reference)": (base, g_base)}
    for w in WEIGHTS:
        weight = np.where(hard, w, 1.0).astype(np.float32)
        p = np.zeros(len(rows), dtype=np.float32)
        for fold in np.array_split(channels, 5):
            test = np.isin(rows.channel, fold)
            p[test] = fit_weighted(F[~test], rows.y[~test].astype(np.float32), weight[~test])(F[test])
        results[f"weight {w:g}"] = (p, graded(sweep_fine(p, rows, p_start, p_end), rows))
        print(f"  trained with weight {w:g}", flush=True)

    ref10 = pick(g_base, 10)
    ref_false = false_regions(ref10[1], rows, idx)
    for name, (_, g) in results.items():
        for b in (5, 10):
            print(line(f"B={b:>2} hard negatives: {name}", pick(g, b)))
        b10 = pick(g, 10)
        now = false_regions(b10[1], rows, idx)
        fixed = {r for r in ref_false if not any(v == r[0] and lo < r[2] and r[1] < hi for v, lo, hi in now)}
        broken = {r for r in now if not any(v == r[0] and lo < r[2] and r[1] < hi for v, lo, hi in ref_false)}
        print(f"        B=10 false regions: {len(now)} (reference {len(ref_false)}); fixed {len(fixed)}, newly broken {len(broken)}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
