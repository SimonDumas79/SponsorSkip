"""Does the seam signal actually know where a sponsor read BEGINS?

The setup mimics what the default tier has to do live. The marker flags a
region, but only roughly -- per-line confidence gives fuzzy boundaries. So:
take each real read start, pretend the marker located it to within a window of
+/- W caption lines, and ask each method to pick the exact line.

  seam        the sharpest change of subject in the window (MiniLM, free)
  centre      guess the middle of the window -- the score to beat, since it is
              what you get from knowing nothing beyond the marker's region
  cues        the first line in the window whose context matches a cue pattern

Errors are reported in caption lines and in seconds, because seconds is what a
viewer actually experiences: land 3 s early and you clip the host's last word,
land 10 s late and the ad plays.

Run: python marker/edge_check.py [--window 10]
"""

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def depth_scores(seam: np.ndarray) -> np.ndarray:
    """TextTiling's depth score, on a dissimilarity curve.

    Hearst's 1997 insight was never "take the sharpest dip". It was to measure
    how far a point stands ABOVE the valleys either side of it. A stretch of
    noisy, low-content captions can produce a high raw seam while being flat in
    context; a real change of subject stands proud of its neighbours even when
    its absolute value is modest. Depth is what tells those apart.
    """
    n = len(seam)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        left = i
        while left > 0 and seam[left - 1] <= seam[left]:
            left -= 1
        right = i
        while right < n - 1 and seam[right + 1] <= seam[right]:
            right += 1
        out[i] = (seam[i] - seam[left]) + (seam[i] - seam[right])
    return out


def summarise(name: str, errors_lines: list[int], errors_secs: list[float], n: int) -> None:
    if not errors_secs:
        print(f"  {name:8s}  no placements")
        return
    secs = np.abs(np.array(errors_secs))
    lines = np.abs(np.array(errors_lines))
    within = lambda s: int((secs <= s).sum())
    print(
        f"  {name:8s}  median {np.median(secs):5.1f} s   worst {secs.max():5.1f} s   "
        f"within 2 s {100 * within(2) / len(secs):4.0f}%   within 5 s {100 * within(5) / len(secs):4.0f}%   median {np.median(lines):.0f} lines"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=HERE / "data" / "features.npz")
    ap.add_argument("--window", type=int, default=10, help="how fuzzy the marker's region is, in caption lines")
    args = ap.parse_args()

    data = np.load(args.features, allow_pickle=False)
    names = list(data["feature_names"])
    seam_col = names.index("seam_here")
    cue_cols = [i for i, n in enumerate(names) if n.startswith("cue_")]

    X, video, line = data["X"], data["video"], data["line"]
    is_start, secs = data["is_start"], data["start_seconds"]

    results: dict[str, tuple[list[int], list[float]]] = {k: ([], []) for k in ("seam", "depth", "centre", "cues")}
    starts = 0

    for vid in np.unique(video):
        rows = np.flatnonzero(video == vid)
        order = rows[np.argsort(line[rows])]
        seam = X[order, seam_col]
        depth = depth_scores(seam)
        cues = X[order][:, cue_cols].sum(axis=1)
        t = secs[order]

        for i in np.flatnonzero(is_start[order]):
            starts += 1
            # The marker does NOT hand over a window neatly centred on the start --
            # it hands over a fuzzy region the start sits somewhere inside. Sweep
            # the offset so every method is asked to FIND the line, not sit on it.
            span = 2 * args.window + 1
            for offset in range(2, span - 2, 3):
                lo, hi = i - offset, i - offset + span
                if lo < 0 or hi > len(order):
                    continue

                picks = {
                    "seam": lo + int(np.argmax(seam[lo:hi])),
                    "depth": lo + int(np.argmax(depth[lo:hi])),
                    "centre": (lo + hi) // 2,
                }
                hit = np.flatnonzero(cues[lo:hi] > 0)
                picks["cues"] = lo + int(hit[0]) if len(hit) else None

                for name, pick in picks.items():
                    if pick is None:
                        continue
                    results[name][0].append(int(pick - i))
                    results[name][1].append(float(t[pick] - t[i]))

    trials = len(results["centre"][1])
    print(f"{starts} sponsor-read starts, {trials} trials (the start placed at varying offsets in the window)")
    print(f"window of {2 * args.window + 1} caption lines; 'centre' is what knowing nothing gets you")
    print()
    for name in ("centre", "cues", "seam", "depth"):
        summarise(name, *results[name], starts)

    seam_secs = np.abs(np.array(results["seam"][1]))
    centre_secs = np.abs(np.array(results["centre"][1]))
    print()
    if len(seam_secs) and np.median(seam_secs) < np.median(centre_secs):
        print(f"The seam beats guessing by {np.median(centre_secs) - np.median(seam_secs):.1f} s at the median.")
    else:
        print("The seam does NOT beat guessing -- on its own it is not an edge placer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
