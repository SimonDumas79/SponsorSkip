"""Pre-registration 5's challenger: candidate v3 with dropout 0.5 in its stacking step. (2026-09-26)

On the enlarged 434-video pool, dropout 0.5 won at B = 5 on every seed (+1.9) and tied at B = 10
(README "Enlarging the stack's training pool"). Simon, 2026-09-26: grade it once on holdout 6 as a
challenger to v3. Everything else stays v3's: the same 160 pooled videos (labels after the community
model's training), the same seven detector streams, the same edge heads. Simon chose no retrain on the
434, so only the dropout changes.

This fixes the challenger's thresholds on channel-grouped CV over those 160 videos, the way candidate.py
fixes v3's (seed-0 streams, folds seed 0), and writes them to data/dropout_thresholds.json BEFORE holdout
6 is built. It also prints v3 vs the challenger on the same CV for seeds 0-2, for reference only.

    python marker/dropout_challenger.py
"""

import ctypes
import datetime
import json
import sys

import numpy as np
import torch

from build_production import pooled
from context_stack import context_features
from detector_bakeoff import CACHE, graded, line, pick
from sbml_eval import load_preds
from stack_check import oof_seeded, sweep_fine
from stack_sbml import sbml_stream
from train import DATA, Rows

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(4)   # papercut 2026-09-24: 8 threads oversubscribe the 8 physical cores
DROPOUT = 0.5


def main() -> int:
    rows, _, _ = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    bake = dict(np.load(CACHE))
    fs, fe = np.load(DATA / "finetune_oof_edge_start_seed0.npy"), np.load(DATA / "finetune_oof_edge_resume_seed0.npy")
    d = json.load(open(DATA / "sb_dates.json"))
    cut = datetime.datetime(2022, 4, 1).timestamp() * 1000
    unseen = np.array([not (d.get(str(v), {}).get("first") and d[str(v)]["first"] < cut) for v in rows.video])
    S = Rows(rows.X[unseen], rows.y[unseen], rows.video[unseen], rows.channel[unseen], rows.start_seconds[unseen],
             rows.split[unseen], rows.feature_names)
    names = [str(n) for n in rows.feature_names]
    extra = rows.X[:, [i for i, n in enumerate(names) if n.startswith("cue_") or n.startswith("desc_")]][unseen]
    sb = context_features(S, sbml_stream(rows, load_preds())[unseen])
    heads = ((p_start + fs) / 2)[unseen], ((p_end + fe) / 2)[unseen]
    print(f"{S.videos} pooled videos with labels after 2022-04, {S.reads} reads\n")
    fixed = None
    for seed in (0, 1, 2):
        bge = np.load(DATA / f"finetune_oof_bge_seed{seed}.npy")
        streams = [level1, bake["meaning"], bake["structure"], bake["potion"], np.load(DATA / "sequence_oof_h32_r7.npy"), bge]
        F = np.column_stack([context_features(S, x[unseen]) for x in streams] + [sb, extra])
        v3 = np.load(DATA / f"stack_sbml_oof_s{seed}.npy")[1]
        challenger = oof_seeded(S, F, 0, dropout=DROPOUT)
        for name, sc in (("v3 (dropout 0.2)", v3), (f"challenger (dropout {DROPOUT})", challenger)):
            g = graded(sweep_fine(sc, S, *heads), S)
            for b in (5, 10):
                print(line(f"BGE s{seed} B={b:>2} {name}", pick(g, b)), flush=True)
            if seed == 0 and name.startswith("challenger"):
                fixed = {f"B{b}": pick(g, b)[0] for b in (5, 10)}
    out = {"dropout": DROPOUT, "thresholds": fixed, "fixed": datetime.date.today().isoformat(),
           "note": "pre-registration 5; chosen on CV over the 160 pooled videos, before holdout 6 exists"}
    (DATA / "dropout_thresholds.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nchallenger thresholds fixed: B=5 {fixed['B5']:.5f}, B=10 {fixed['B10']:.5f} -> data/dropout_thresholds.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
