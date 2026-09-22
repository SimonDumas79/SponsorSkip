"""The island filter: skip a region only if it is unlike BOTH neighbours. Chosen by the rule on CV. (2026-09-22)

resume_check.py found that a region's similarity to its CLOSER side (mean frozen-MiniLM embedding of
the region against the SIDE lines before it and the SIDE lines after it, whichever is more alike)
separates real reads (median 0.65) from free-tier false alarms (0.74) 75% of the time. A read is an
island; product talk resembles the show around it. Simon's correction stands: whether the two sides
match each other carries no signal, because reads often sit between two subjects.

The filter has no fitted weights, only a cutoff: a region is skipped when island <= cutoff. The
cutoff and the detection threshold are chosen together by the pooled rule (most ad time with at most
B s of show lost per video, at most 2% of videos over 60 s), like every other threshold.

Tried on the shipped free tier, the 5-detector stack, and the stack with fine-tuned MiniLM added.

    python marker/island_check.py
"""

import sys

import numpy as np

from build_production import pooled
from detector_bakeoff import CACHE, graded, line, pick
from edge_heads import place
from replay import regions_by_video
from stack_check import SHARES
from train import DATA, LAST_LINE_SECONDS

sys.stdout.reconfigure(encoding="utf-8")
SIDE = 12
CUTOFFS = [None, 0.60, 0.63, 0.66, 0.69, 0.72, 0.75, 0.78, 0.81]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def island(E: np.ndarray, lo: int, hi: int) -> float:
    inside = unit(E[lo:hi].mean(0))
    sides = [E[max(0, lo - SIDE):lo], E[hi:hi + SIDE]]
    sims = [float(inside @ unit(s.mean(0))) for s in sides if len(s)]
    return max(sims) if sims else 0.0


def main() -> int:
    rows, _, _ = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    bake = dict(np.load(CACHE))
    ft_path = DATA / "finetune_streams.npz"
    ft = dict(np.load(ft_path)) if ft_path.exists() else {}
    E = rows.X[:, :384]
    idx = {str(v): np.flatnonzero(rows.video == v) for v in np.unique(rows.video)}
    secs = np.zeros(len(rows))
    for r in idx.values():
        s = rows.start_seconds[r]
        secs[r] = np.diff(np.append(s, s[-1] + LAST_LINE_SECONDS))
    systems = {"shipped free tier": level2, "stack (5 detectors)": bake["ctx_stack"]}
    for s in (0, 1):
        if f"stack_ft{s}" in ft:
            systems[f"stack + fine-tuned seed {s}"] = ft[f"stack_ft{s}"]

    print(f"pooled: {rows.videos} videos, {rows.reads} reads; island = region vs its closer side, {SIDE} lines\n")
    for name, scores in systems.items():
        cands = {c: [] for c in CUTOFFS}
        for th in np.unique(np.quantile(scores, 1 - SHARES)):
            kept = place(regions_by_video(scores, rows, float(th), 1), rows, p_start, p_end)
            isl = {v: [island(E[idx[v]], lo, hi) for lo, hi in spans] for v, spans in kept.items()}
            for c in CUTOFFS:
                filt = kept if c is None else {
                    v: [sp for sp, i in zip(spans, isl[v]) if i <= c] for v, spans in kept.items()}
                cands[c].append((float(th), filt))
        graded_by_c = {c: graded(cands[c], rows) for c in CUTOFFS}
        for budget in (5, 10):
            base = pick(graded_by_c[None], budget)
            best_c, best = None, base
            for c in CUTOFFS[1:]:
                b = pick(graded_by_c[c], budget)
                if b and (best is None or b[2]["coverage"] > best[2]["coverage"]):
                    best_c, best = c, b
            print(line(f"B={budget:>2} {name}", base))
            label = f"B={budget:>2}   + island filter (cutoff {best_c})" if best_c else f"B={budget:>2}   + island filter: no cutoff helps"
            print(line(label, best))
            if best_c:
                for tag, b in (("without", base), ("with", best)):
                    false_n = sum(1 for v, spans in b[1].items() for lo, hi in spans if not rows.y[idx[v]][lo:hi].any())
                    print(f"        false-alarm regions {tag} the filter: {false_n}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
