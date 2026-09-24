"""The enlarged stack-training pool (2026-09-24): honest out-of-fold streams over 494 videos.

Every stage after a detector must be OUT-OF-FOLD with channel-grouped folds over the pool it trains
on. build_production.pooled() already generalises to any list of feature files (`sets`); this script
uses ENLARGED_SETS (the graded 205 plus features_holdout3/channels/holdout4/holdout5.npz -- 494
videos, 326 channels, all already graded or used as a tripwire in PREREGISTRATION.md, so none of them
is a live holdout any more) and recomputes, over that pool:

  pooled_oof_enlarged.npy       level1 (linear marker), level2 (its context model), p_start, p_end
                                (mirrors build_production.py's pooled_oof.npy cache, generalised)
  bakeoff_streams_enlarged.npz  meaning / structure / potion / cues, via detector_bakeoff.level_one
                                (needs sequence_oof_h32_r7_enlarged.npy already built by
                                `sequence_model.py --extra-sets ... --variants 32,7` first)

Every one of these reuses the SAME code path as the 205-video versions (experiments.cross_validate,
context_stack.out_of_fold, edge_heads.edge_oof, detector_bakeoff.level_one), just given the enlarged
rows, so the fold logic is identical -- only the pool is bigger. Nothing here touches a live holdout;
holdout 6 (not yet built) is the only fresh test left.

    python marker/enlarge_pool.py            # builds pooled_oof_enlarged.npy and bakeoff_streams_enlarged.npz
"""

import ctypes
import sys

import numpy as np
import torch

from build_production import ENLARGED_SETS, pooled
from context_stack import context_features, out_of_fold
from edge_heads import edge_oof, soft
from experiments import BASE, cross_validate
from train import DATA

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(6)

POOLED_OOF = DATA / "pooled_oof_enlarged.npy"
BAKEOFF_CACHE = DATA / "bakeoff_streams_enlarged.npz"


def enlarged_rows():
    return pooled(ENLARGED_SETS)


def build_pooled_oof(rows, is_start, is_resume) -> tuple:
    if POOLED_OOF.exists() and np.load(POOLED_OOF).shape[1] == len(rows):
        level1, level2, p_start, p_end = np.load(POOLED_OOF)
        return level1, level2, p_start, p_end
    print(f"enlarged pool: {rows.videos} videos, {len(set(rows.channel))} channels, {rows.reads} reads", flush=True)
    print("  level 1 (linear marker), out of fold...", flush=True)
    level1 = cross_validate(BASE, rows, seed=0)
    F = context_features(rows, level1)
    print("  level 2 (context model), out of fold...", flush=True)
    level2 = out_of_fold(rows, F, hidden=32)
    print("  edge heads (start/resume), out of fold...", flush=True)
    p_start = edge_oof(rows, F, soft(is_start, rows.video))
    p_end = edge_oof(rows, F, soft(is_resume, rows.video))
    np.save(POOLED_OOF, np.stack([level1, level2, p_start, p_end]))
    print(f"  -> {POOLED_OOF}", flush=True)
    return level1, level2, p_start, p_end


def main() -> int:
    rows, is_start, is_resume = enlarged_rows()
    build_pooled_oof(rows, is_start, is_resume)

    from detector_bakeoff import POTION_ENLARGED, level_one
    if not (DATA / "sequence_oof_h32_r7_enlarged.npy").exists():
        print("sequence_oof_h32_r7_enlarged.npy not built yet -- run:\n"
              "  python marker/sequence_model.py --extra-sets features_holdout3.npz features_channels.npz "
              "features_holdout4.npz features_holdout5.npz --variants 32,7\nfirst; skipping the bake-off streams.")
        return 0
    print("\nbake-off streams (meaning, structure, potion, cues, sequence) over the enlarged pool...", flush=True)
    streams, level2 = level_one(rows, cache=BAKEOFF_CACHE, potion_files=POTION_ENLARGED, pooled_oof=POOLED_OOF,
                                sequence_variants=("h32_r7",), sequence_suffix="_enlarged")
    print(f"  -> {BAKEOFF_CACHE}, streams: {list(streams)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
