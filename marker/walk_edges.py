"""Simon's edge rule: start inside the read and walk outwards line by line until it is clearly over. (2026-09-22)

The shipped placement asks a start model to score every candidate line in a window and takes the best
one (an argmax). Simon's version is sequential: from a line that is certainly inside the read, step
backwards while each line still looks like the ad, and stop once it clearly is not; then the same
forwards. widen_gate.py tried this on the CONTEXT model's score and the rule always chose not to walk,
but that score is averaged over 31 lines, so it stays high well past the true edge. This walks on the
FINE-TUNED per-line score instead (the model that reads the line plus 8 lines either side, the one
that cut start error from ~4 s to 2.4 s), and stops only after PATIENCE lines in a row below the line
threshold, so a single quiet line inside the ad does not end the walk.

Knobs, chosen together with the detection threshold by the pooled rule: the line threshold (as a share
of lines it would flag on its own) and PATIENCE (1, 2, 3 lines). Compared against the shipped argmax
heads and the averaged heads, three BGE stack seeds.

    python marker/walk_edges.py
"""

import ctypes
import sys

import numpy as np
import torch

from build_production import pooled
from detector_bakeoff import CACHE, graded, line, pick
from edge_ft_eval import agreed_reads, edge_errors
from edge_heads import merged, place
from replay import regions_by_video
from stack_check import SHARES, sweep_fine
from train import DATA

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(8)
LINE_SHARES = [0.06, 0.10, 0.15, 0.22]
PATIENCE = [1, 2, 3]
MAX_WALK = 40   # lines: a read is not longer than this beyond where the region already reaches


def walk(kept: dict, rows, idx: dict, ft: np.ndarray, cut: float, patience: int) -> dict:
    out = {}
    for v, spans in kept.items():
        r = idx[v]
        f = ft[r]
        placed = []
        for lo, hi in spans:
            a, miss = lo, 0
            while a > 0 and lo - a < MAX_WALK:
                if f[a - 1] >= cut:
                    a, miss = a - 1, 0
                else:
                    miss += 1
                    if miss >= patience:
                        break
                    a -= 1
            a += miss if miss < patience else 0   # give back the quiet lines the walk ate
            b, miss = hi, 0
            while b < len(r) and b - hi < MAX_WALK:
                if f[b] >= cut:
                    b, miss = b + 1, 0
                else:
                    miss += 1
                    if miss >= patience:
                        break
                    b += 1
            b -= miss if miss < patience else 0
            placed.append((max(0, a), min(len(r), max(b, a + 1))))
        out[v] = merged(placed)
    return out


def main() -> int:
    rows, _, _ = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    stacks = dict(np.load(DATA / "finetune_streams.npz"))
    fs, fe = np.load(DATA / "finetune_oof_edge_start_seed0.npy"), np.load(DATA / "finetune_oof_edge_resume_seed0.npy")
    ft = np.load(DATA / "finetune_oof_bge_seed0.npy")   # the sharp per-line score the walk follows
    idx = {str(v): np.flatnonzero(rows.video == v) for v in np.unique(rows.video)}
    agreed = agreed_reads(rows, idx)
    cuts = [(s, float(np.quantile(ft, 1 - s))) for s in LINE_SHARES]
    print(f"pooled: {rows.videos} videos, {rows.reads} reads; the walk follows the fine-tuned line score\n")
    for s in (0, 1, 2):
        sc = stacks[f"stack_bge_seed{s}"]
        ths = np.unique(np.quantile(sc, 1 - SHARES))
        regions = {float(t): regions_by_video(sc, rows, float(t), 1) for t in ths}
        systems = {"argmax heads (shipped)": graded([(t, place(k, rows, p_start, p_end)) for t, k in regions.items()], rows),
                   "averaged heads (v2)": graded([(t, place(k, rows, (p_start + fs) / 2, (p_end + fe) / 2))
                                                  for t, k in regions.items()], rows)}
        best = None
        for share, cut in cuts:
            for pat in PATIENCE:
                g = graded([(f"t {t:.4f} share {share} patience {pat}", walk(k, rows, idx, ft, cut, pat))
                            for t, k in regions.items()], rows)
                for b in (5, 10):
                    pk = pick(g, b)
                    if pk and (best is None or b not in best or pk[2]["coverage"] > best[b][2]["coverage"]):
                        best = best or {}
                        best[b] = pk
        for name, g in systems.items():
            for b in (5, 10):
                print(line(f"bge s{s} B={b:>2} {name}", pick(g, b)))
        for b in (5, 10):
            print(line(f"bge s{s} B={b:>2} walk outwards (Simon's rule)", best[b]))
            print(f"        {edge_errors(best[b][1], rows, idx, agreed)}  [{best[b][0]}]", flush=True)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
