"""Conditional widening: extend a region backwards only while the evidence stays up (hysteresis). (2026-09-22)

The edges room's pick for the costliest edge loss: 31 regions begin more than 20 lines after the
true start (out of the start model's reach; 29 min of ad). Widening every region's search was
measured and rejected, because on false regions the wider reach widens the damage and the rule
pushes the threshold up. This widens CONDITIONALLY: a region must still cross the strict threshold
to exist, but its start then walks backwards (and its end forwards) while the score stays above a
looser LOW threshold, before the edge heads place the final lines. A region with a weak run-up is
not widened at all. No training: the strict threshold and LOW are chosen together by the pooled rule.

    python marker/widen_gate.py
"""

import sys

import numpy as np

from build_production import pooled
from detector_bakeoff import graded, line, pick
from edge_heads import place
from experiments import runs
from stack_check import SHARES
from train import DATA

sys.stdout.reconfigure(encoding="utf-8")
LOW_SHARES = [None, 0.30, 0.20, 0.14, 0.10]   # LOW as the share of lines it would flag on its own


def grown(scores, rows, idx, th: float, low: float | None, back_only: bool) -> dict:
    out = {}
    for v, r in idx.items():
        s = scores[r]
        spans = runs((s >= th).astype(np.int8))
        if low is not None:
            new = []
            for a, b in spans:
                while a > 0 and s[a - 1] >= low:
                    a -= 1
                if not back_only:
                    while b < len(s) and s[b] >= low:
                        b += 1
                new.append((a, b))
            spans = new
        if spans:
            out[v] = spans
    return out


def main() -> int:
    rows, _, _ = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    ft = dict(np.load(DATA / "finetune_streams.npz"))
    idx = {str(v): np.flatnonzero(rows.video == v) for v in np.unique(rows.video)}
    for name, scores in (("shipped free tier", level2), ("stack + fine-tuned s0", ft["stack_ft0"])):
        ths = np.unique(np.quantile(scores, 1 - SHARES))
        for back_only in (True, False):
            best = {}
            for ls in LOW_SHARES:
                low = None if ls is None else float(np.quantile(scores, 1 - ls))
                cands = [(float(t), place(grown(scores, rows, idx, float(t), low, back_only), rows, p_start, p_end))
                         for t in ths if low is None or t > low]
                g = graded(cands, rows)
                for b in (5, 10):
                    pk = pick(g, b)
                    if pk and (b not in best or pk[2]["coverage"] > best[b][1][2]["coverage"]):
                        best[b] = (ls, pk)
                    if ls is None:
                        best.setdefault(("none", b), pk)
            for b in (5, 10):
                print(line(f"B={b:>2} {name}: no widening", best[("none", b)]))
                ls, pk = best[b]
                label = "back only" if back_only else "both ends"
                print(line(f"B={b:>2}   widened {label}, LOW flags {ls if ls is None else f'{ls:.0%}'}", pk))
            print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
