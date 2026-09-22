"""qwen3:8b as a detector (qwen_sweep.py's recorded answers), alone and in the stack. (2026-09-22)

qwen was asked about every window of every pooled video (45 lines, 40 apart) and gave yes/no plus the
read's first and last line. Those line ranges become a line score (1 inside a range qwen gave, else 0).
qwen never saw SponsorBlock labels, so all 205 pooled videos are fair: graded alone as it answers, and
as one more detector in the stack + BGE (context model out of fold with the usual channel folds,
averaged edge heads), three BGE seeds.

    python marker/qwen_eval.py
"""

import ctypes
import sys

import numpy as np
import torch

from build_production import over_cap, pooled
from context_stack import context_features
from detector_bakeoff import CACHE, graded, line, pick
from experiments import runs
from overlap import qwen_flags
from replay import grade_regions
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
    fs, fe = np.load(DATA / "finetune_oof_edge_start_seed0.npy"), np.load(DATA / "finetune_oof_edge_resume_seed0.npy")
    idx = {str(v): np.flatnonzero(rows.video == v) for v in np.unique(rows.video)}
    q, complete = qwen_flags(idx, len(rows))
    print(f"qwen answered every window of {len(complete)} of {rows.videos} pooled videos\n")
    kept = {v: runs(q[r].astype(np.int8)) for v, r in idx.items() if q[r].any()}
    g = grade_regions(kept, rows)
    print(f"  qwen alone, as it answers: ad time {g['coverage']:.1%}  show lost {g['show']:.1f} s/video  "
          f"over 60 s {over_cap(kept, rows):.1%}\n")
    names = [str(n) for n in rows.feature_names]
    extra = rows.X[:, [i for i, n in enumerate(names) if n.startswith("cue_") or n.startswith("desc_")]]
    heads = (p_start + fs) / 2, (p_end + fe) / 2
    qs = q.astype(np.float32) * 0.98 + 0.01
    for seed in (0, 1, 2):
        bge = np.load(DATA / f"finetune_oof_bge_seed{seed}.npy")
        F = [context_features(rows, x) for x in (level1, bake["meaning"], bake["structure"], bake["potion"],
                                                  np.load(DATA / "sequence_oof_h32_r7.npy"), bge)]
        for name, parts in (("stack + BGE", F), ("stack + BGE + qwen", F + [context_features(rows, qs)])):
            sc = oof_seeded(rows, np.column_stack(parts + [extra]), 0)
            gg = graded(sweep_fine(sc, rows, *heads), rows)
            for b in (5, 10):
                print(line(f"BGE s{seed} B={b:>2} {name}", pick(gg, b)), flush=True)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
