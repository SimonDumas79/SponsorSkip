"""Train the start/end models on cleaner edges: drop or replace the ones Claude disputes. (2026-09-22)

Simon's point, measured by verify_edges.py: Claude and SponsorBlock place a read's edges on the same
line most of the time (median gap 0 s), but only 53% of reads agree within 3 s at BOTH edges and 10%
differ by more than 10 s at each. Reading the disagreements, neither side is always right (Claude
keeps lead-ins like "a quick word from..." that SponsorBlock drops, and drops story-style lead-ins
that SponsorBlock keeps), so disputed edges are noise to the start/end models either way.

Four ways to build the start/resume training targets, everything else fixed (the shipped context
model finds the regions; the edge models are the same MLPs on the same window features):
  as shipped      SponsorBlock's edges
  masked          disputed edges (Claude placed it more than TOL s away) carry no weight in training
  claude-if-off   Claude's edge where the two disagree, SponsorBlock's elsewhere
  claude          Claude's edge wherever Claude placed one
Graded two ways: the pooled rule on the standard labels, and the start/end error of the placed
region on reads where both sources AGREE (the only edges we trust as truth).

    python marker/edge_clean.py
"""

import ctypes
import json
import sys

import numpy as np
import torch

from build_production import pooled
from context_stack import context_features
from detector_bakeoff import graded, line, pick
from edge_heads import place, soft
from experiments import runs
from short_reads import fit_weighted
from stack_check import sweep_fine
from train import DATA

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(8)
TOL = 3.0     # seconds: closer than this, the two sources agree
MASK = 3      # lines either side of a disputed edge that carry no weight


def oof_weighted(rows, F, target, weight) -> np.ndarray:
    channels = np.array(sorted(set(rows.channel)))
    np.random.default_rng(0).shuffle(channels)
    p = np.zeros(len(rows), dtype=np.float32)
    for fold in np.array_split(channels, 5):
        test = np.isin(rows.channel, fold)
        p[test] = fit_weighted(F[~test], target[~test], weight[~test])(F[test])
    return p


def main() -> int:
    rows, is_start, is_resume = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    ft = dict(np.load(DATA / "finetune_streams.npz"))
    F = context_features(rows, level1)
    videos = [str(v) for v in np.unique(rows.video)]
    idx = {v: np.flatnonzero(rows.video == v) for v in videos}
    ver = {(r["video"], r["read"]): r for r in map(json.loads, open(DATA / "edge_verify.jsonl", encoding="utf-8"))}

    targets = {k: (np.zeros(len(rows), np.float32), np.zeros(len(rows), np.float32)) for k in ("shipped", "claude-if-off", "claude")}
    weight = np.ones(len(rows), np.float32)
    agreed = []   # (video, row-local start, row-local end) of reads where both sources agree at both edges
    disputed = 0
    for v in videos:
        r = idx[v]
        for n, (a, b) in enumerate(runs(rows.y[r])):
            rec = ver.get((v, n))
            cs = ce = None
            ok_s = ok_e = True
            if rec and rec["claude_start"] is not None:
                cs, ce = rec["claude_start"], rec["claude_end"]
                ok_s, ok_e = abs(rec["d_start_s"]) <= TOL, abs(rec["d_end_s"]) <= TOL
                if ok_s and ok_e:
                    agreed.append((v, a, b))
            for key, (ts, te) in targets.items():
                s_line = cs if (key == "claude" and cs is not None) or (key == "claude-if-off" and cs is not None and not ok_s) else a
                e_line = ce if (key == "claude" and ce is not None) or (key == "claude-if-off" and ce is not None and not ok_e) else b
                ts[r[min(s_line, len(r) - 1)]] = 1
                if e_line < len(r):
                    te[r[e_line]] = 1
            for ok, sb, cl in ((ok_s, a, cs), (ok_e, b, ce)):
                if not ok and cl is not None:
                    disputed += 1
                    for line_i in (sb, cl):
                        weight[r[max(0, line_i - MASK):min(len(r), line_i + MASK + 1)]] = 0.0
    print(f"pooled: {rows.videos} videos, {rows.reads} reads; {len(agreed)} reads agree at both edges, "
          f"{disputed} disputed edges masked in the 'masked' variant\n")

    variants = {"as shipped": (p_start, p_end)}
    ones = np.ones(len(rows), np.float32)
    ss, se = soft(targets["shipped"][0], rows.video), soft(targets["shipped"][1], rows.video)
    variants["masked"] = (oof_weighted(rows, F, ss, weight), oof_weighted(rows, F, se, weight))
    for key in ("claude-if-off", "claude"):
        ts, te = targets[key]
        variants[key] = (oof_weighted(rows, F, soft(ts, rows.video), ones), oof_weighted(rows, F, soft(te, rows.video), ones))
    variants["as shipped (retrained, same trainer)"] = (oof_weighted(rows, F, ss, ones), oof_weighted(rows, F, se, ones))

    for system, scores in (("shipped free tier", level2), ("stack + fine-tuned s0", ft["stack_ft0"])):
        print(f"{system}:")
        for name, (ps, pe) in variants.items():
            g = graded(sweep_fine(scores, rows, ps, pe), rows)
            b10 = pick(g, 10)
            # edge error on agreed reads, at the B=10 choice
            errs_s, errs_e = [], []
            for v, a, b in agreed:
                r = idx[v]
                s = rows.start_seconds[r]
                spans = [sp for sp in b10[1].get(v, []) if sp[0] < b and a < sp[1]]
                if spans:
                    sp = min(spans, key=lambda x: abs(x[0] - a))
                    errs_s.append(s[min(sp[0], len(s) - 1)] - s[a])
                    errs_e.append(s[min(sp[1], len(s) - 1)] - s[min(b, len(s) - 1)])
            es, ee = np.abs(errs_s), np.abs(errs_e)
            print(line(f"  B= 5 {name}", pick(g, 5)))
            print(line(f"  B=10 {name}", b10))
            print(f"        on {len(es)} agreed reads it touches: start error median {np.median(es):.1f} s "
                  f"({np.mean(es <= 2):.0%} within 2 s), end error median {np.median(ee):.1f} s ({np.mean(ee <= 2):.0%} within 2 s)",
                  flush=True)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
