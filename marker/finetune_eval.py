"""Does fine-tuning MiniLM beat the frozen marker? Same folds, same pipeline, same rule. (2026-09-22)

Reads finetune_minilm.py's out-of-fold scores (every data/finetune_oof_seed*.npy present) and compares
them with the shipped marker's, on the pooled 205 videos:

  line level   how well each ranks ad lines above show lines (average precision, ROC AUC)
  alone        raw scores with their own edges; then a context model on them + the edge heads,
               the free tier's shape, with the pooled rule (most ad time under B s lost per video,
               at most 2% of videos over 60 s)
  in the stack the stacked context model from detector_bakeoff.py with the fine-tuned scores added
  read by read the fine-tuned free tier against the shipped one at B = 10 (sign test)
  false alarms regions that overlap no read, and the show they cost, against the shipped tier

The threshold grid is stack_check.py's (90 points), so the shipped and stacked numbers here match its
seed-0 rows.

    python marker/finetune_eval.py
"""

import ctypes
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from build_production import pooled
from context_stack import context_features
from detector_bakeoff import CACHE, graded, line, pick
from edge_heads import edge_oof, soft
from replay import per_read
from stack_check import oof_seeded, sign_test, sweep_fine
from train import DATA, LAST_LINE_SECONDS

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(4)

FT_CACHE = DATA / "finetune_streams.npz"


def false_alarms(kept: dict, rows, secs: np.ndarray) -> tuple[int, float, set]:
    n, lost, vids = 0, 0.0, set()
    for vid, spans in kept.items():
        r = np.flatnonzero(rows.video == vid)
        for lo, hi in spans:
            if not rows.y[r][lo:hi].any():
                n += 1
                lost += float(secs[r][lo:hi].sum())
                vids.add(vid)
    return n, lost / rows.videos, vids


def main() -> int:
    rows, is_start, is_resume = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    bake = dict(np.load(CACHE))
    cache = dict(np.load(FT_CACHE)) if FT_CACHE.exists() else {}
    seeds = sorted(int(p.stem.split("seed")[1]) for p in DATA.glob("finetune_oof_seed*.npy") if "partial" not in p.name)
    if not seeds:
        raise SystemExit("no data/finetune_oof_seed*.npy yet: run finetune_minilm.py first")
    secs = np.zeros(len(rows))
    for vid in np.unique(rows.video):
        r = np.flatnonzero(rows.video == vid)
        s = rows.start_seconds[r]
        secs[r] = np.diff(np.append(s, s[-1] + LAST_LINE_SECONDS))
    y = rows.y
    print(f"pooled: {rows.videos} videos, {rows.reads} reads; fine-tuned seeds: {seeds}\n")

    print("LINE LEVEL: ranking ad lines above show lines (higher is better)")
    print(f"  {'frozen marker (shipped)':<34} average precision {average_precision_score(y, level1):.3f}   "
          f"ROC AUC {roc_auc_score(y, level1):.3f}")
    ft = {}
    for s in seeds:
        ft[s] = np.load(DATA / f"finetune_oof_seed{s}.npy")
        print(f"  {f'fine-tuned MiniLM, seed {s}':<34} average precision {average_precision_score(y, ft[s]):.3f}   "
              f"ROC AUC {roc_auc_score(y, ft[s]):.3f}")

    print("\nTHROUGH THE PIPELINE, pooled rule:")
    g_ship = graded(sweep_fine(level2, rows, p_start, p_end), rows)
    g_stack = graded(sweep_fine(bake["ctx_stack"], rows, p_start, p_end), rows)
    for budget in (5, 10):
        print(line(f"B={budget:>2} shipped free tier (frozen)", pick(g_ship, budget)))
    for budget in (5, 10):
        print(line(f"B={budget:>2} stacked model (5 detectors)", pick(g_stack, budget)))

    stack_parts = [level1, bake["meaning"], bake["structure"], bake["potion"], np.load(DATA / "sequence_oof_h32_r7.npy")]
    names = [str(n) for n in rows.feature_names]
    extra = rows.X[:, [i for i, n in enumerate(names) if n.startswith("cue_") or n.startswith("desc_")]]
    for s in seeds:
        F = context_features(rows, ft[s])
        for key, make in ((f"ctx_ft{s}", lambda: oof_seeded(rows, F, 0)),
                          (f"start_ft{s}", lambda: edge_oof(rows, F, soft(is_start, rows.video))),
                          (f"end_ft{s}", lambda: edge_oof(rows, F, soft(is_resume, rows.video))),
                          (f"stack_ft{s}", lambda: oof_seeded(rows, np.column_stack(
                              [context_features(rows, p) for p in stack_parts + [ft[s]]] + [extra]), 0))):
            if key not in cache:
                print(f"  training {key}...", flush=True)
                cache[key] = make()
                np.savez(FT_CACHE, **cache)
        g_ctx = graded(sweep_fine(cache[f"ctx_ft{s}"], rows, p_start, p_end), rows)
        g_own = graded(sweep_fine(cache[f"ctx_ft{s}"], rows, cache[f"start_ft{s}"], cache[f"end_ft{s}"]), rows)
        g_st6 = graded(sweep_fine(cache[f"stack_ft{s}"], rows, p_start, p_end), rows)
        for budget in (5, 10):
            print(line(f"B={budget:>2} fine-tuned s{s}: context + shipped heads", pick(g_ctx, budget)))
            print(line(f"B={budget:>2} fine-tuned s{s}: context + its own heads", pick(g_own, budget)))
            print(line(f"B={budget:>2} stack + fine-tuned s{s} (6 detectors)", pick(g_st6, budget)))

        b_ship, b_ft = pick(g_ship, 10), max((pick(g_ctx, 10), pick(g_own, 10)),
                                             key=lambda b: -1 if b is None else b[2]["coverage"])
        if b_ft is not None:
            a = np.array([v for *_, v in per_read(b_ship[1], rows)])
            b = np.array([v for *_, v in per_read(b_ft[1], rows)])
            better, worse = int(np.sum(b > a + 0.05)), int(np.sum(b < a - 0.05))
            print(f"\n  READ BY READ, B=10, fine-tuned s{s} free tier vs shipped: more of {better} reads, less of {worse} "
                  f"(sign test p = {sign_test(better, worse):.4f})")
            for label, bb in (("shipped", b_ship), (f"fine-tuned s{s}", b_ft)):
                n, lost, vids = false_alarms(bb[1], rows, secs)
                print(f"  FALSE ALARMS, {label:<16} {n:3d} regions in {len(vids):3d} videos, {lost:4.1f} s of show per video")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
