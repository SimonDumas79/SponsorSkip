"""Targeted fixes for the reads v3 misses (miss_audit.py, 2026-09-23), tested like v3 itself was.

miss_audit found, on v3's out-of-fold scores over the 160 pooled videos: English reads with a cue word
76.5% of ad time skipped, reads in machine-translated captions 52.4% (52 of 160 videos), reads with no cue
word ~37.5% in either language, English self-promo 29.2%. Each variant here adds one signal to v3's
context model and is compared with v3 as built (stack_sbml.py's "with_sb"), same videos, same folds,
seeds 0-2, same threshold rule (B = 5 and 10 s of show lost per video).

    python marker/target_misses.py [--variants translated]
"""

import argparse
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
torch.set_num_threads(8)


def translated_column(video: np.ndarray) -> np.ndarray:
    """1 on every line of a video whose spoken language is not English (its English track is machine-translated)."""
    lang = json.load(open(DATA / "video_language.json", encoding="utf-8"))
    is_tr = {v: float((lang.get(v, {}).get("detected") or "en") != "en") for v in map(str, np.unique(video))}
    return np.array([is_tr[str(v)] for v in video], dtype=np.float32)[:, None]


def island_streams(S, widths=(8, 16, 32), side=12) -> list[np.ndarray]:
    """Per line: how unlike its surroundings the WIDTH lines centred on it are, 1 - max(cos to the SIDE lines
    before, cos to the SIDE lines after), on the frozen MiniLM meaning columns. Simon's island idea (a read is
    unlike both sides, 75% separation in resume_check.py) used as a DETECTOR for missed reads, where
    island_check.py only tried it as a filter on regions already found."""
    E = S.X[:, :384].astype(np.float64)
    out = [np.zeros(len(E), dtype=np.float32) for _ in widths]
    for v in np.unique(S.video):
        r = np.flatnonzero(S.video == v)
        c = np.vstack([np.zeros((1, 384)), np.cumsum(E[r], 0)])
        n = len(r)
        mean = lambda a, b: (c[b] - c[a]) / max(b - a, 1)
        unit = lambda x: x / (np.linalg.norm(x) + 1e-9)
        for k, w in enumerate(widths):
            for i in range(n):
                lo, hi = max(0, i - w // 2), min(n, i + w - w // 2)
                if lo == 0 or hi == n:
                    continue
                mid = unit(mean(lo, hi))
                sim = max(float(mid @ unit(mean(max(0, lo - side), lo))), float(mid @ unit(mean(hi, min(n, hi + side)))))
                out[k][r[i]] = 1.0 - sim
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["translated"])
    a = ap.parse_args()
    rows, _, _ = pooled()
    level1, _, p_start, p_end = np.load(DATA / "pooled_oof.npy")
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
    added = {}
    if "translated" in a.variants:
        added["translated"] = translated_column(S.video)
    if "qwen" in a.variants:
        # qwen3's recorded answers on every window of the pooled videos (qwen_sweep.py, 09-22), never before
        # stacked with the community model: this is level 2 (the local model with v3), priced at zero GPU.
        from overlap import qwen_flags
        idx = {str(v): np.flatnonzero(rows.video == v) for v in np.unique(rows.video)}
        q, _ = qwen_flags(idx, len(rows))
        added["qwen"] = context_features(S, (q.astype(np.float32) * 0.98 + 0.01)[unseen])
    if "island" in a.variants:
        added["island"] = np.column_stack([context_features(S, x) for x in island_streams(S)])
    print(f"{S.videos} videos, {S.reads} reads\n")
    for seed in (0, 1, 2):
        bge = np.load(DATA / f"finetune_oof_bge_seed{seed}.npy")
        streams = [level1, bake["meaning"], bake["structure"], bake["potion"], np.load(DATA / "sequence_oof_h32_r7.npy"), bge]
        F = [context_features(S, x[unseen]) for x in streams] + [context_features(S, sb[unseen])]
        v3 = np.load(DATA / f"stack_sbml_oof_s{seed}.npy")[1]   # v3 as built, not recomputed
        runs = [("v3 as built", v3)]
        for name in a.variants:
            runs.append((f"v3 + {name}", oof_seeded(S, np.column_stack(F + [extra, added[name]]), 0)))
            np.save(DATA / f"target_{name}_oof_s{seed}.npy", runs[-1][1])
        for name, sc in runs:
            g = graded(sweep_fine(sc, S, *heads), S)
            for b in (5, 10):
                print(line(f"s{seed} B={b:>2} {name}", pick(g, b)), flush=True)
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
