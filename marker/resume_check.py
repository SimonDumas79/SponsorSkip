"""Does the show come back after a read? And is a read unlike BOTH sides? Measured before building anything. (2026-09-22)

Two candidate features for telling real reads from product-talk false alarms, from the research
round and Simon's correction:
  resume   the stretch BEFORE a region and the stretch AFTER it are about the same thing
           (the show resumes). Simon: a read often sits between two subjects, so this can fail.
  island   the region is unlike BOTH neighbours, whether or not they match each other.

Each is a cosine similarity between mean MiniLM line embeddings (the frozen marker's meaning
columns), over SIDE lines either side of the region. Compared on: every real read (label spans),
and the free tier's false-alarm regions at a loose threshold (so there are enough of them).

    python marker/resume_check.py
"""

import sys

import numpy as np

from build_production import pooled
from edge_heads import place
from experiments import runs
from replay import regions_by_video
from train import DATA

sys.stdout.reconfigure(encoding="utf-8")
SIDE = 12
LOOSE = 0.95


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def facts(E, lo, hi):
    n = len(E)
    if lo < 3 or hi > n - 3:
        return None
    before, after, inside = E[max(0, lo - SIDE):lo].mean(0), E[hi:min(n, hi + SIDE)].mean(0), E[lo:hi].mean(0)
    b, a, i = unit(before), unit(after), unit(inside)
    return {"resume": float(b @ a), "to_before": float(i @ b), "to_after": float(i @ a),
            "island": float(max(i @ b, i @ a))}


def main() -> int:
    rows, _, _ = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    E = rows.X[:, :384]
    kept = place(regions_by_video(level2, rows, LOOSE, 1), rows, p_start, p_end)
    real, false = [], []
    for vid in np.unique(rows.video):
        r = np.flatnonzero(rows.video == vid)
        y = rows.y[r]
        for a, b in runs(y):
            f = facts(E[r], a, b)
            if f:
                real.append(f)
        for lo, hi in kept.get(str(vid), []):
            if not y[lo:hi].any():
                f = facts(E[r], lo, hi)
                if f:
                    false.append(f)
    print(f"{len(real)} real reads, {len(false)} false-alarm regions (free tier at context threshold {LOOSE})\n")
    print(f"{'feature':<44}{'real reads':>22}{'false alarms':>22}")
    for key, name in (("resume", "before vs after (does the show resume?)"),
                      ("to_before", "region vs before"), ("to_after", "region vs after"),
                      ("island", "region vs its CLOSER side (island test)")):
        rv, fv = np.array([f[key] for f in real]), np.array([f[key] for f in false])
        print(f"  {name:<42}  median {np.median(rv):.2f} [{np.percentile(rv, 25):.2f}-{np.percentile(rv, 75):.2f}]"
              f"  median {np.median(fv):.2f} [{np.percentile(fv, 25):.2f}-{np.percentile(fv, 75):.2f}]")
    rv = np.array([f["resume"] for f in real])
    print(f"\n  real reads whose before and after are LESS alike than the median false alarm's: "
          f"{np.mean(rv < np.median([f['resume'] for f in false])):.0%}")
    for key in ("resume", "island"):
        rv, fv = np.array([f[key] for f in real]), np.array([f[key] for f in false])
        # how well one number separates the two groups: the chance a random real read scores lower than a random false alarm
        auc = np.mean(rv[:, None] < fv[None, :]) + 0.5 * np.mean(rv[:, None] == fv[None, :])
        print(f"  separation by '{key}': a random real read scores LOWER than a random false alarm {auc:.0%} of the time "
              f"(50% = no signal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
