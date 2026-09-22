"""Joint start/end placement: choose each region's (start, end) PAIR, not two separate argmaxes. (2026-09-22)

error_budget.py: late starts cost 14% of all ad time; 66 regions have the true start inside the start
head's search window but pick the wrong line, and 31 begin too deep in the read for it to reach.
edge_heads.place picks the start as the argmax of the start model alone, then the end as the argmax
of the end model alone. Here every pair (i, j) in the same windows is scored together:

    S(i, j) = logit p_start(i) + logit p_end(j) + ALPHA * sum over lines k in [i, j) of (logit inside(k) - TAU)

The last term is the region's own interior evidence (the context model's score, the same one that
found the region): a start is pulled earlier while the lines before it still look like the ad, and
held back when they look like show. With ALPHA = 0 it is the two start/end models alone. No weights
are fitted: ALPHA, TAU and the detection threshold are chosen together by the pooled rule on CV.
(The research round's stronger version, a trained pair scorer, comes after this if it pays.)

    python marker/joint_edges.py
"""

import sys

import numpy as np

from build_production import pooled
from context_stack import logit
from detector_bakeoff import CACHE, graded, line, pick
from edge_heads import AFTER, BEFORE, INSIDE, TAIL, merged, place
from replay import regions_by_video
from train import DATA

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
ALPHAS = [0.0, 0.1, 0.25, 0.5, 1.0]
TAUS = [-2.0, -1.0, 0.0, 1.0, 2.0]
SHARES = np.geomspace(0.004, 0.25, 36)


def place_joint(kept: dict, rows, p_start, p_end, inside, alpha: float, tau: float, idx: dict) -> dict:
    zs_all, ze_all, zi_all = logit(p_start), logit(p_end), logit(inside) - tau
    out = {}
    for vid, spans in kept.items():
        r = idx[vid]
        zs, ze, zi = zs_all[r], ze_all[r], zi_all[r]
        csum = np.concatenate([[0.0], np.cumsum(zi)])
        placed = []
        for lo, hi in spans:
            starts = np.arange(max(0, lo - BEFORE), min(len(r), lo + INSIDE))
            ends = np.arange(max(1, hi - TAIL), min(len(r), hi + AFTER + 1))
            if not len(starts) or not len(ends):
                placed.append((lo, hi))
                continue
            S = zs[starts][:, None] + ze[np.minimum(ends, len(r) - 1)][None, :] \
                + alpha * (csum[ends][None, :] - csum[starts][:, None])
            S[ends[None, :] <= starts[:, None]] = -np.inf
            a, b = np.unravel_index(np.argmax(S), S.shape)
            placed.append((int(starts[a]), int(ends[b])) if np.isfinite(S[a, b]) else (lo, hi))
        out[vid] = merged(placed)
    return out


def main() -> int:
    rows, _, _ = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    bake = dict(np.load(CACHE))
    ft_path = DATA / "finetune_streams.npz"
    ft = dict(np.load(ft_path)) if ft_path.exists() else {}
    idx = {str(v): np.flatnonzero(rows.video == v) for v in np.unique(rows.video)}
    systems = {"shipped free tier": level2, "stack (5 detectors)": bake["ctx_stack"]}
    if "stack_ft0" in ft:
        systems["stack + fine-tuned seed 0"] = ft["stack_ft0"]
    print(f"pooled: {rows.videos} videos, {rows.reads} reads\n")
    for name, scores in systems.items():
        ths = np.unique(np.quantile(scores, 1 - SHARES))
        regions = {float(t): regions_by_video(scores, rows, float(t), 1) for t in ths}
        base = graded([(t, place(k, rows, p_start, p_end)) for t, k in regions.items()], rows)
        results = {}
        for alpha in ALPHAS:
            for tau in (TAUS if alpha > 0 else [0.0]):
                results[(alpha, tau)] = graded(
                    [(t, place_joint(k, rows, p_start, p_end, scores, alpha, tau, idx)) for t, k in regions.items()], rows)
        for budget in (5, 10):
            print(line(f"B={budget:>2} {name}: separate start/end (shipped)", pick(base, budget)))
            best_key, best = None, None
            for key, g in results.items():
                b = pick(g, budget)
                if b and (best is None or b[2]["coverage"] > best[2]["coverage"]):
                    best_key, best = key, b
            print(line(f"B={budget:>2}   joint pair, ALPHA {best_key[0]} TAU {best_key[1]}" if best_key else
                       f"B={budget:>2}   joint pair", best))
            b0 = pick(results[(0.0, 0.0)], budget)
            print(line(f"B={budget:>2}   joint pair, ALPHA 0 (start/end models only)", b0))
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
