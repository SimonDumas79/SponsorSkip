"""The community SponsorBlock model as a seventh detector in our stack, without its leakage. (2026-09-22)

On the pooled videos whose labels appeared after its training (early 2022), the community model skips
73.4% of ad time but loses 17.0 s of show per video (6.9% of videos over 60 s); our stack + BGE with
averaged heads skips 61.5% at 9.5 s. It reads ~500-token chunks and extracts the sponsor text, a very
different view from ours, so it may find reads we miss.

Its finds become a line score (the classifier's probability for sponsor/self-promo on lines whose start
falls inside a find, else 0). The stack's context model is trained and graded ONLY on the 160 pooled
videos whose SponsorBlock labels postdate its training (sb_dates.py), with channel-grouped 5-fold CV
inside them, so it cannot have learned our answers. Compared against the same stack without it,
retrained on the same 160 videos. Edge heads: the averaged MLP + fine-tuned heads (candidate v2).

    python marker/stack_sbml.py
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
from sbml_eval import READS, load_preds
from stack_check import oof_seeded, sweep_fine
from train import DATA, Rows

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(8)


def sbml_stream(rows, preds) -> np.ndarray:
    out = np.zeros(len(rows), dtype=np.float32)
    for v in np.unique(rows.video):
        r = np.flatnonzero(rows.video == v)
        s = rows.start_seconds[r]
        for p in preds.get(str(v), []):
            probs = p["probs"]
            q = max(probs.get(c, 0.0) for c in READS)
            m = (s >= p["start"]) & (s < p["end"])
            out[r[m]] = np.maximum(out[r[m]], q)
    return out


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
    sb = sbml_stream(rows, load_preds())
    heads = ((p_start + fs) / 2)[unseen], ((p_end + fe) / 2)[unseen]
    print(f"{S.videos} pooled videos with labels after 2022-04, {S.reads} reads\n")
    for seed in (0, 1, 2):
        bge = np.load(DATA / f"finetune_oof_bge_seed{seed}.npy")
        streams = [level1, bake["meaning"], bake["structure"], bake["potion"], np.load(DATA / "sequence_oof_h32_r7.npy"), bge]
        F = [context_features(S, x[unseen]) for x in streams]
        without = oof_seeded(S, np.column_stack(F + [extra]), 0)
        with_sb = oof_seeded(S, np.column_stack(F + [context_features(S, sb[unseen]), extra]), 0)
        for name, sc in (("stack + BGE (retrained on the 160)", without), ("stack + BGE + SponsorBlock model", with_sb)):
            g = graded(sweep_fine(sc, S, *heads), S)
            for b in (5, 10):
                print(line(f"BGE s{seed} B={b:>2} {name}", pick(g, b)), flush=True)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
