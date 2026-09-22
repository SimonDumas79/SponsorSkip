"""With fine-tuned BGE in the stack, which of the other detectors still earn their place? (2026-09-22)

Shipping cost matters: every level-1 detector is another model to run per video. The stack of five
bake-off detectors + fine-tuned BGE-small gives 57.0-58.7% (B = 10). Smaller stacks, same context
model recipe (seed 0, same folds), same edge heads, pooled rule, for two BGE seeds:
  BGE alone (its own context model), marker + BGE, marker + meaning + structure + BGE (one encoder
  family plus BGE: no potion, no conv model), and all six.

    python marker/bge_ablation.py
"""

import ctypes
import sys

import numpy as np
import torch

from build_production import pooled
from context_stack import context_features
from detector_bakeoff import CACHE, graded, line, pick
from stack_check import oof_seeded, sweep_fine
from train import DATA

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(8)


def main() -> int:
    rows, _, _ = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    bake = dict(np.load(CACHE))
    ft = dict(np.load(DATA / "finetune_streams.npz"))
    names = [str(n) for n in rows.feature_names]
    extra = rows.X[:, [i for i, n in enumerate(names) if n.startswith("cue_") or n.startswith("desc_")]]
    base = {"marker": level1, "meaning": bake["meaning"], "structure": bake["structure"], "potion": bake["potion"],
            "sequence": np.load(DATA / "sequence_oof_h32_r7.npy")}
    ctx = {k: context_features(rows, v) for k, v in base.items()}
    for s in (0, 1):
        bge = np.load(DATA / f"finetune_oof_bge_seed{s}.npy")
        cb = context_features(rows, bge)
        variants = {
            "BGE alone": ft[f"ctx_bge_seed{s}"],
            "marker + BGE": oof_seeded(rows, np.column_stack([ctx["marker"], cb, extra]), 0),
            "marker + meaning + structure + BGE": oof_seeded(rows, np.column_stack(
                [ctx["marker"], ctx["meaning"], ctx["structure"], cb, extra]), 0),
            "all six": ft[f"stack_bge_seed{s}"],
        }
        print(f"BGE seed {s}:")
        for name, sc in variants.items():
            g = graded(sweep_fine(sc, rows, p_start, p_end), rows)
            for b in (5, 10):
                print(line(f"  B={b:>2} {name}", pick(g, b)), flush=True)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
