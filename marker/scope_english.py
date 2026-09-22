"""English scope, seed-averaged fine-tune, and the 7-detector stack. (2026-09-22)

Three questions from the rooms and today's results, on the pooled out-of-fold scores:
  1. Seed ensembling (a room pre-flight): average the three fine-tuned MiniLM seeds' logits into one
     detector, instead of any single seed.
  2. The single-line fine-tune (window 0) ranks lines far worse alone (AP 0.37) but lifted the stack
     at B = 10 (56.5%): does a stack with BOTH fine-tunes hold up?
  3. Non-English videos (30% of the pooled set, from video_language.json) carry machine-translated
     caption tracks and several false alarms; if they fall back to SponsorBlock, the free tier's scope
     is English. Graded on English videos only, with the context model trained on all videos or on
     English videos only.
Every context model: seed 0, the same channel folds; shipped edge heads; pooled rule.

    python marker/scope_english.py
"""

import ctypes
import json
import sys

import numpy as np
import torch

from build_production import pooled
from context_stack import context_features, logit
from detector_bakeoff import CACHE, graded, line, pick
from stack_check import oof_seeded, sweep_fine
from train import DATA, Rows

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(8)
OUT = DATA / "scope_streams.npz"


def subset(rows, m) -> Rows:
    return Rows(rows.X[m], rows.y[m], rows.video[m], rows.channel[m], rows.start_seconds[m], rows.split[m], rows.feature_names)


def main() -> int:
    rows, _, _ = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    bake = dict(np.load(CACHE))
    ft = dict(np.load(DATA / "finetune_streams.npz"))
    cache = dict(np.load(OUT)) if OUT.exists() else {}
    lang = json.load(open(DATA / "video_language.json", encoding="utf-8"))
    en = np.array([lang.get(str(v), {}).get("lang") == "en" for v in rows.video])
    sub = subset(rows, en)

    seeds = [np.load(DATA / f"finetune_oof_seed{s}.npy") for s in (0, 1, 2)]
    ens = 1 / (1 + np.exp(-np.mean([logit(p) for p in seeds], axis=0)))
    single = np.load(DATA / "finetune_oof_single_seed0.npy")
    names = [str(n) for n in rows.feature_names]
    extra = rows.X[:, [i for i, n in enumerate(names) if n.startswith("cue_") or n.startswith("desc_")]]
    five = [level1, bake["meaning"], bake["structure"], bake["potion"], np.load(DATA / "sequence_oof_h32_r7.npy")]
    F5 = [context_features(rows, p) for p in five]
    F_ens, F_single = context_features(rows, ens), context_features(rows, single)
    inputs = {"stack + fine-tuned ensemble": np.column_stack(F5 + [F_ens, extra]),
              "stack + ensemble + single-line": np.column_stack(F5 + [F_ens, F_single, extra])}

    def get(key, make):
        if key not in cache:
            print(f"  training {key}...", flush=True)
            cache[key] = make()
            np.savez(OUT, **cache)
        return cache[key]

    systems = {"stack + fine-tuned s0 (reference)": (ft["stack_ft0"], None)}
    for name, F in inputs.items():
        all_scores = get(f"all|{name}", lambda F=F: oof_seeded(rows, F, 0))
        en_scores = get(f"en|{name}", lambda F=F: oof_seeded(sub, F[en], 0))
        systems[name] = (all_scores, en_scores)

    print(f"\npooled: {rows.videos} videos, {rows.reads} reads; English: {sub.videos} videos, {sub.reads} reads\n")
    for name, (all_scores, en_scores) in systems.items():
        g_all = graded(sweep_fine(all_scores, rows, p_start, p_end), rows)
        g_en = graded(sweep_fine(all_scores[en], sub, p_start[en], p_end[en]), sub)
        for b in (5, 10):
            print(line(f"B={b:>2} {name[:30]}: all videos", pick(g_all, b)))
            print(line(f"B={b:>2}   graded on English", pick(g_en, b)))
        if en_scores is not None:
            g_en2 = graded(sweep_fine(en_scores, sub, p_start[en], p_end[en]), sub)
            for b in (5, 10):
                print(line(f"B={b:>2}   trained AND graded on English", pick(g_en2, b)))
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
