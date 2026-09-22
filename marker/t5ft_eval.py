"""Did fine-tuning the community T5 on our labels help? Alone and in the stack. (2026-09-22)

Reads data/t5ft_predictions.jsonl (t5_finetune.py: out of fold on the 160 pooled videos whose labels
postdate the community model's training) and data/sbml_predictions.jsonl (the original community
model), and grades both on those 160 videos: as they answer (classifier cut 0.5), and each as the
seventh detector of the stack (context model trained and graded on the 160 with channel-grouped CV,
averaged edge heads), three BGE seeds.

    python marker/t5ft_eval.py
"""

import ctypes
import datetime
import json
import sys

import numpy as np
import torch

from build_production import over_cap, pooled
from context_stack import context_features
from detector_bakeoff import CACHE, graded, line, pick
from experiments import runs
from replay import grade_regions
from stack_check import oof_seeded, sweep_fine
from stack_sbml import sbml_stream
from train import DATA, Rows

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(8)


def load(path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for rec in map(json.loads, f):
            out.setdefault(rec["video"], [])
            if rec.get("start") is not None:
                out[rec["video"]].append(rec)
    return out


def main() -> int:
    rows, _, _ = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    bake = dict(np.load(CACHE))
    fs, fe = np.load(DATA / "finetune_oof_edge_start_seed0.npy"), np.load(DATA / "finetune_oof_edge_resume_seed0.npy")
    d = json.load(open(DATA / "sb_dates.json"))
    cutoff = datetime.datetime(2022, 4, 1).timestamp() * 1000
    unseen = np.array([not (d.get(str(v), {}).get("first") and d[str(v)]["first"] < cutoff) for v in rows.video])
    S = Rows(rows.X[unseen], rows.y[unseen], rows.video[unseen], rows.channel[unseen], rows.start_seconds[unseen],
             rows.split[unseen], rows.feature_names)
    streams = {"original community model": sbml_stream(rows, load(DATA / "sbml_predictions.jsonl")),
               "fine-tuned on our labels": sbml_stream(rows, load(DATA / "t5ft_predictions.jsonl"))}
    print(f"{S.videos} unseen pooled videos, {S.reads} reads\n")
    for name, st in streams.items():
        kept = {}
        for v in np.unique(S.video):
            r = np.flatnonzero(S.video == v)
            spans = runs((st[unseen][r] >= 0.5).astype(np.int8))
            if spans:
                kept[str(v)] = spans
        g = grade_regions(kept, S)
        print(f"  alone, {name:<28} ad time {g['coverage']:6.1%}  show lost {g['show']:5.1f} s/video  over 60 s {over_cap(kept, S):5.1%}")
    print()
    names = [str(n) for n in rows.feature_names]
    extra = rows.X[:, [i for i, n in enumerate(names) if n.startswith("cue_") or n.startswith("desc_")]][unseen]
    heads = ((p_start + fs) / 2)[unseen], ((p_end + fe) / 2)[unseen]
    for seed in (0, 1, 2):
        bge = np.load(DATA / f"finetune_oof_bge_seed{seed}.npy")
        base = [level1, bake["meaning"], bake["structure"], bake["potion"], np.load(DATA / "sequence_oof_h32_r7.npy"), bge]
        F = [context_features(S, x[unseen]) for x in base]
        for name, st in streams.items():
            sc = oof_seeded(S, np.column_stack(F + [context_features(S, st[unseen]), extra]), 0)
            g = graded(sweep_fine(sc, S, *heads), S)
            for b in (5, 10):
                print(line(f"BGE s{seed} B={b:>2} stack + {name}", pick(g, b)), flush=True)
        both = oof_seeded(S, np.column_stack(F + [context_features(S, st[unseen]) for st in streams.values()] + [extra]), 0)
        g = graded(sweep_fine(both, S, *heads), S)
        for b in (5, 10):
            print(line(f"BGE s{seed} B={b:>2} stack + both", pick(g, b)), flush=True)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
