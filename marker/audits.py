"""The rooms' zero-cost audits, on scores already on disk. (2026-09-22)

  bootstrap   channel-clustered bootstrap of each candidate's ad-time gain over the shipped tier at its
              own rule-chosen setting (resample the 203 channels with replacement, 2000 draws): the room
              wanted the gain's interval with channels, not reads, as the unit
  stability   do the same reads win across the three fine-tune seeds, or do they reshuffle?
  misses      per-read complete-miss / partial / full rates, with Wilson 95% intervals
  starts      on reads where Claude and SponsorBlock agree, does the placed start's error track a caption
              gap (a long pause right before the read) or a [Music] line? (the data room's A4)

    python marker/audits.py
"""

import json
import math
import sys

import numpy as np

from build_production import pooled
from detector_bakeoff import graded, pick
from experiments import runs
from replay import per_read
from stack_check import sweep_fine
from train import DATA, LAST_LINE_SECONDS

sys.stdout.reconfigure(encoding="utf-8")


def wilson(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{p:.0%} [{c - h:.0%}, {c + h:.0%}]"


def per_video(kept: dict, rows, idx) -> dict:
    """Per video: (ad seconds, ad seconds skipped)."""
    out = {}
    for v, r in idx.items():
        s = rows.start_seconds[r]
        sec = np.diff(np.append(s, s[-1] + LAST_LINE_SECONDS))
        f = np.zeros(len(r), bool)
        for lo, hi in kept.get(v, []):
            f[lo:hi] = True
        y = rows.y[r] == 1
        out[v] = (float(sec[y].sum()), float(sec[y & f].sum()))
    return out


def main() -> int:
    rows, _, _ = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    ft = dict(np.load(DATA / "finetune_streams.npz"))
    idx = {str(v): np.flatnonzero(rows.video == v) for v in np.unique(rows.video)}
    chan = {v: rows.channel[r[0]] for v, r in idx.items()}
    channels = sorted(set(chan.values()))
    by_chan = {c: [v for v in idx if chan[v] == c] for c in channels}
    print(f"pooled: {rows.videos} videos, {len(channels)} channels "
          f"({sum(len(v) > 1 for v in by_chan.values())} with more than one video)\n")

    systems = {"shipped": level2}
    for key, name in (("stack_ft0", "stack + ft s0"), ("stack_ft1", "stack + ft s1"), ("stack_ft2", "stack + ft s2"),
                      ("stack_bge_seed0", "stack + bge s0")):
        if key in ft:
            systems[name] = ft[key]
    picked = {name: {b: pick(graded(sweep_fine(s, rows, p_start, p_end), rows), b) for b in (5, 10)}
              for name, s in systems.items()}

    print("BOOTSTRAP over channels (2000 draws): gain in ad time over the shipped tier, each at its own rule choice")
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(channels), size=(2000, len(channels)))
    for b in (5, 10):
        base = per_video(picked["shipped"][b][1], rows, idx)
        for name in systems:
            if name == "shipped":
                continue
            cand = per_video(picked[name][b][1], rows, idx)
            ad = np.array([sum(base[v][0] for v in by_chan[c]) for c in channels])
            got_b = np.array([sum(base[v][1] for v in by_chan[c]) for c in channels])
            got_c = np.array([sum(cand[v][1] for v in by_chan[c]) for c in channels])
            gains = (got_c[draws].sum(1) - got_b[draws].sum(1)) / ad[draws].sum(1)
            print(f"  B={b:>2} {name:<16} gain {np.mean(gains):+6.1%}  95% interval [{np.percentile(gains, 2.5):+.1%}, "
                  f"{np.percentile(gains, 97.5):+.1%}]  share of draws > 0: {np.mean(gains > 0):.0%}")

    print("\nSTABILITY across fine-tune seeds (B = 10): reads each stack skips more of than the shipped tier (by 5+ points)")
    base_pr = np.array([s for *_, s in per_read(picked["shipped"][10][1], rows)])
    wins = {}
    for name in ("stack + ft s0", "stack + ft s1", "stack + ft s2"):
        if name in picked:
            pr = np.array([s for *_, s in per_read(picked[name][10][1], rows)])
            wins[name] = set(np.flatnonzero(pr > base_pr + 0.05))
    if len(wins) >= 2:
        sets = list(wins.values())
        common = set.intersection(*sets)
        union = set.union(*sets)
        print(f"  wins per seed: {[len(s) for s in sets]}; in every seed: {len(common)}; in any seed: {len(union)}")

    print("\nMISSES per read (B = 10), Wilson 95% intervals:")
    for name in systems:
        pr = np.array([s for *_, s in per_read(picked[name][10][1], rows)])
        n = len(pr)
        print(f"  {name:<16} complete miss {wilson(int((pr == 0).sum()), n)}   partial {wilson(int(((pr > 0) & (pr < 0.95)).sum()), n)}"
              f"   full {wilson(int((pr >= 0.95).sum()), n)}")

    print("\nSTART ERROR vs caption features, on reads where Claude and SponsorBlock agree (shipped tier, B = 10):")
    ver = {(r["video"], r["read"]): r for r in map(json.loads, open(DATA / "edge_verify.jsonl", encoding="utf-8"))}
    errs, gaps, music = [], [], []
    kept = picked["shipped"][10][1]
    for v, r in idx.items():
        s = rows.start_seconds[r]
        for n_read, (a, bb) in enumerate(runs(rows.y[r])):
            rec = ver.get((v, n_read))
            if not rec or rec["claude_start"] is None or abs(rec["d_start_s"]) > 3 or abs(rec["d_end_s"]) > 3:
                continue
            spans = [sp for sp in kept.get(v, []) if sp[0] < bb and a < sp[1]]
            if not spans or a == 0:
                continue
            sp = min(spans, key=lambda x: abs(x[0] - a))
            errs.append(abs(s[min(sp[0], len(s) - 1)] - s[a]))
            gaps.append(float(s[a] - s[a - 1]))
    errs, gaps = np.array(errs), np.array(gaps)
    if len(errs) > 5:
        rho = np.corrcoef(np.argsort(np.argsort(errs)), np.argsort(np.argsort(gaps)))[0, 1]
        print(f"  {len(errs)} reads: Spearman correlation of start error with the caption gap before the read: {rho:+.2f}")
        big = gaps > np.percentile(gaps, 75)
        print(f"  median start error when the gap is in the top quarter: {np.median(errs[big]):.1f} s; otherwise {np.median(errs[~big]):.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
