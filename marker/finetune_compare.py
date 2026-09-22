"""Compare every fine-tune run (the recipe of record's seeds and each tagged variant) the same way. (2026-09-22)

For each data/finetune_oof_*seed*.npy: line-level average precision and ROC AUC; the free-tier shape
(a context model on its scores + the shipped edge heads, pooled rule, B = 5 and 10); and the stack
(the five bake-off detectors + this run, one context model). Context models use seed 0 and the same
channel folds. Results are cached in data/finetune_streams.npz under the run's name.

    python marker/finetune_compare.py
"""

import ctypes
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from build_production import pooled
from context_stack import context_features
from detector_bakeoff import CACHE, graded, pick
from stack_check import oof_seeded, sweep_fine
from train import DATA

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(8)
FT_CACHE = DATA / "finetune_streams.npz"


def cell(b) -> str:
    return "   --   " if b is None else f"{b[2]['coverage']:5.1%} {b[2]['show']:3.1f}s"


def main() -> int:
    rows, _, _ = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    bake = dict(np.load(CACHE))
    cache = dict(np.load(FT_CACHE)) if FT_CACHE.exists() else {}
    runs = sorted(p.stem for p in DATA.glob("finetune_oof_*seed*.npy") if "partial" not in p.name and "edge_" not in p.name)
    names = [str(n) for n in rows.feature_names]
    extra = rows.X[:, [i for i, n in enumerate(names) if n.startswith("cue_") or n.startswith("desc_")]]
    parts = [level1, bake["meaning"], bake["structure"], bake["potion"], np.load(DATA / "sequence_oof_h32_r7.npy")]
    base_F = [context_features(rows, p) for p in parts]
    y = rows.y

    print(f"pooled: {rows.videos} videos, {rows.reads} reads. Cells: ad time skipped, show lost per video.\n")
    print(f"  {'run':<34}{'AP':>7}{'AUC':>7}   {'free tier B5':>14} {'B10':>13}   {'stack+run B5':>14} {'B10':>13}")
    g = graded(sweep_fine(level2, rows, p_start, p_end), rows)
    gs = graded(sweep_fine(bake["ctx_stack"], rows, p_start, p_end), rows)
    print(f"  {'frozen marker (shipped)':<34}{average_precision_score(y, level1):7.3f}{roc_auc_score(y, level1):7.3f}"
          f"   {cell(pick(g, 5)):>14} {cell(pick(g, 10)):>13}   {cell(pick(gs, 5)):>14} {cell(pick(gs, 10)):>13}  (stack = 5 detectors)",
          flush=True)
    for stem in runs:
        p = np.load(DATA / f"{stem}.npy")
        name = stem.replace("finetune_oof_", "")
        # the recipe-of-record seeds were cached by finetune_eval.py under ctx_ft<seed> / stack_ft<seed>
        legacy = name.startswith("seed") and name[4:].isdigit()
        k_ctx, k_st = (f"ctx_ft{name[4:]}", f"stack_ft{name[4:]}") if legacy else (f"ctx_{name}", f"stack_{name}")
        if k_ctx not in cache:
            cache[k_ctx] = oof_seeded(rows, context_features(rows, p), 0)
            np.savez(FT_CACHE, **cache)
        if k_st not in cache:
            cache[k_st] = oof_seeded(rows, np.column_stack(base_F + [context_features(rows, p), extra]), 0)
            np.savez(FT_CACHE, **cache)
        g = graded(sweep_fine(cache[k_ctx], rows, p_start, p_end), rows)
        gs = graded(sweep_fine(cache[k_st], rows, p_start, p_end), rows)
        print(f"  {name:<34}{average_precision_score(y, p):7.3f}{roc_auc_score(y, p):7.3f}"
              f"   {cell(pick(g, 5)):>14} {cell(pick(g, 10)):>13}   {cell(pick(gs, 5)):>14} {cell(pick(gs, 10)):>13}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
