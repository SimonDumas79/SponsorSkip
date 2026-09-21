"""Score routing policies for free, by replaying recorded verdicts (the Dream-RSI idea).

Dream-RSI (arXiv 2609.14858) keeps the expensive generator fixed and improves the
POLICY that decides what to do with its outputs, scoring each candidate policy
offline against a recorded history instead of paying for fresh runs. Its limit
is the one that matters here: replay is only honest for policies that SELECT
over events that were actually recorded. A policy that would ask qwen about a
window nobody asked it about cannot be scored this way.

Here the fixed generators are the marker (which flags regions) and qwen3 (which
said yes or no about each flagged region, recorded by confirm_check.py in
data/confirm_verdicts.jsonl). A policy decides which flagged regions get
skipped, from what was recorded: the marker's peak score, qwen's answer, qwen's
P(yes), the region's length. Every policy is graded exactly the way the search
grades a threshold: reads caught (region recall), seconds of real show skipped
per video, the worst single video, and windows per video.

    python marker/replay.py                   # the standard policy table
    python marker/replay.py --search          # search policy space over the log

No model runs here: the marker's out-of-fold scores are recomputed in seconds on
the CPU to recover which caption lines each region covers.
"""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

from score import score_video, smooth
from train import DATA, LAST_LINE_SECONDS, SLACK, load
from experiments import BASE, cross_validate, runs

VERDICTS = DATA / "confirm_verdicts.jsonl"


def regions_by_video(oof: np.ndarray, rows, threshold: float, smooth_w: int) -> dict[str, list[tuple[int, int]]]:
    out = {}
    for vid in np.unique(rows.video):
        r = np.flatnonzero(rows.video == vid)
        out[str(vid)] = runs(smooth((oof[r] >= threshold).astype(np.int8), smooth_w))
    return out


def grow(flags: np.ndarray, scores: np.ndarray, lo: int, hi: int, low: float) -> np.ndarray:
    """Extend each skipped run inside [lo, hi) outwards while the marker's score stays >= low."""
    out = flags.copy()
    for a, b in runs(flags[lo:hi]):
        a, b = a + lo, b + lo
        while a > lo and scores[a - 1] >= low:
            a -= 1
        while b < hi and scores[b] >= low:
            b += 1
        out[a:b] = 1
    return out


def grade_regions(kept: dict[str, list[tuple[int, int]]], rows, oof: np.ndarray | None = None,
                  core: float = 0.0, bridge: int = 1, low: float | None = None) -> dict:
    """Region recall, ad coverage and cost when these regions (line ranges per video) are skipped.

    With core > 0 only the confirmed region's CORE is skipped: its lines scoring >= core, with
    gaps of up to `bridge` lines closed. The loose threshold finds the read and qwen confirms it;
    the tight one decides how much of it to skip, so a region's long false tails stay playing.
    With `low`, each core then grows outwards while the marker's score stays >= low (hysteresis),
    to win back the ad seconds a tight core leaves playing.

    Region recall alone would flatter a tight core: one skipped line anywhere in a read counts it
    as found. So `coverage` is reported beside it: the share of all ad seconds actually skipped.
    """
    found = total = windows = 0
    lost, ad_total, ad_skipped = [], 0.0, 0.0
    for vid in np.unique(rows.video):
        r = np.flatnonzero(rows.video == vid)
        labels = rows.y[r]
        flags = np.zeros(len(r), dtype=np.int8)
        # Bridged over the whole video, then cut to each region: np.convolve(mode="same") returns
        # max(len, window) values, so bridging a region shorter than the window would misalign.
        core_flags = smooth((oof[r] >= core).astype(np.int8), bridge) if core > 0 else None
        for lo, hi in kept.get(str(vid), []):
            if core > 0:
                inner = np.zeros(len(r), dtype=np.int8)
                inner[lo:hi] = core_flags[lo:hi]
                if not inner[lo:hi].any():   # confirmed but nothing crosses the core: keep its peak line
                    inner[lo + int(np.argmax(oof[r][lo:hi]))] = 1
                if low is not None:
                    inner = grow(inner, oof[r], lo, hi, low)
                flags = np.maximum(flags, inner)
            else:
                flags[lo:hi] = 1
        f, t, w, _ = score_video(labels, flags, SLACK)
        found += f
        total += t
        windows += w
        s = rows.start_seconds[r]
        on_screen = np.diff(np.append(s, s[-1] + LAST_LINE_SECONDS))
        lost.append(float(on_screen[(flags == 1) & (labels == 0)].sum()))
        ad_total += float(on_screen[labels == 1].sum())
        ad_skipped += float(on_screen[(flags == 1) & (labels == 1)].sum())
    lost = np.array(lost)
    return {"found": found, "total": total, "recall": found / max(total, 1), "windows": windows / len(lost),
            "show": float(lost.mean()), "worst": float(lost.max()), "over15": int((lost > 15).sum()),
            "coverage": ad_skipped / max(ad_total, 1e-9)}


def per_read(kept: dict[str, list[tuple[int, int]]], rows) -> list[tuple[str, int, float, float]]:
    """(video, read number, read length in seconds, share of the read skipped) for every real read."""
    out = []
    for vid in np.unique(rows.video):
        r = np.flatnonzero(rows.video == vid)
        labels = rows.y[r]
        flags = np.zeros(len(r), dtype=np.int8)
        for lo, hi in kept.get(str(vid), []):
            flags[lo:hi] = 1
        s = rows.start_seconds[r]
        on_screen = np.diff(np.append(s, s[-1] + LAST_LINE_SECONDS))
        for k, (a, b) in enumerate(runs(labels)):
            length = float(on_screen[a:b].sum())
            out.append((str(vid), k, length, float(on_screen[a:b][flags[a:b] == 1].sum()) / max(length, 1e-9)))
    return out


def load_verdicts(path: Path = VERDICTS) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def apply(policy, verdicts: dict[str, dict]) -> dict[str, list[tuple[int, int]]]:
    kept: dict[str, list[tuple[int, int]]] = {}
    for v in verdicts.values():
        if policy(v):
            kept.setdefault(v["video"], []).append((v["lo"], v["hi"]))
    return kept


def row(name: str, g: dict) -> str:
    return (f"  {name:<44} {g['found']:3d}/{g['total']:<3d} {g['recall']:6.1%}  {g['coverage']:6.1%}  "
            f"{g['show']:6.1f} s  {g['worst']:5.0f} s  {g['over15']:3d}  {g['windows']:5.1f}")


HEADER = ("  policy                                       reads  recall   ad cov   show/vid  worst  >15s  win/vid\n"
          "  (ad cov = share of all ad seconds actually skipped; show/vid = seconds of real show skipped per video;\n"
          "   worst = the most show any one video loses; >15s = videos losing more than 15 s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", action="store_true", help="search a grid of policies over the recorded log")
    ap.add_argument("--verdicts", type=Path, default=VERDICTS, help="a confirm_check.py verdict log")
    ap.add_argument("--policy", help="grade one policy, 'a,b,c,w': keep if (qwen yes and peak >= a) or peak >= b; "
                                     "skip the core >= c with gaps <= w bridged (c = 0: skip the whole region)")
    ap.add_argument("--alone", type=float, nargs="*", default=[],
                    help="also grade the marker ALONE at these thresholds, on the same rows, for comparison")
    args = ap.parse_args()

    torch.set_num_threads(4)
    verdicts = load_verdicts(args.verdicts)
    manifest_path = args.verdicts.with_suffix(".manifest.json")
    if manifest_path.exists():   # confirm_check.py recorded exactly which scores and threshold it used
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = load(Path(manifest["features"]))
        oof = np.load(manifest["scores"])
        th, smooth_w = manifest["threshold"], manifest["config"]["smooth"]
    else:   # the first run predates the manifest: the baseline marker, out-of-fold, at 0.84
        rows = load(DATA / "features.npz")
        oof = cross_validate(BASE, rows, seed=0)
        th, smooth_w = 0.84, 3
    regions = regions_by_video(oof, rows, th, smooth_w)
    recorded = {f"{vid}:{lo}-{hi}" for vid, rs in regions.items() for lo, hi in rs}
    missing = recorded - set(verdicts)
    print(f"{len(verdicts)} recorded verdicts; {len(recorded)} regions at threshold {th}; "
          f"{len(missing)} regions with no verdict (must be 0 for an honest replay)")

    print("\n" + HEADER)
    policies = [
        ("marker alone (skip every flagged region)", lambda v: True),
        ("qwen says yes", lambda v: v["said"] is True),
        ("qwen yes, or marker peak >= 0.99", lambda v: v["said"] is True or v["marker_peak"] >= 0.99),
        ("qwen yes, or marker peak >= 0.995", lambda v: v["said"] is True or v["marker_peak"] >= 0.995),
        ("qwen yes and marker peak >= 0.9", lambda v: v["said"] is True and v["marker_peak"] >= 0.9),
        ("qwen yes and marker peak >= 0.95", lambda v: v["said"] is True and v["marker_peak"] >= 0.95),
        ("P(yes) >= 0.1", lambda v: (v["p_yes"] or 0) >= 0.1),
        ("P(yes) >= 0.3", lambda v: (v["p_yes"] or 0) >= 0.3),
        ("qwen yes or P(yes) >= 0.3", lambda v: v["said"] is True or (v["p_yes"] or 0) >= 0.3),
    ]
    for name, policy in policies:
        print(row(name, grade_regions(apply(policy, verdicts), rows)))

    # What qwen got wrong, for reading by eye.
    lost_reads = [v for v in verdicts.values() if v["is_read"] and v["said"] is not True]
    kept_false = [v for v in verdicts.values() if not v["is_read"] and v["said"] is True]
    print(f"\nqwen rejected {len(lost_reads)} windows holding a read; accepted {len(kept_false)} false windows")
    (DATA / "replay_errors.json").write_text(json.dumps({"rejected_reads": lost_reads, "accepted_false": kept_false},
                                                        indent=1), encoding="utf-8")

    for alone in args.alone:
        flags_regions = regions_by_video(oof, rows, alone, smooth_w)
        print(row(f"marker alone at {alone}", grade_regions(flags_regions, rows)))
    if args.policy:
        a, b, c, w = (float(x) for x in args.policy.split(","))
        policy = lambda v: (v["said"] is True and v["marker_peak"] >= a) or v["marker_peak"] >= b
        print(row(f"POLICY a={a} b={b} c={c} w={int(w)}", grade_regions(apply(policy, verdicts), rows, oof, c, int(w))))

    print("\nconfirm the loose region, skip only its core (lines >= core, gaps <= bridge closed):")
    for core, bridge in [(0.9, 3), (0.95, 3), (0.95, 5), (0.97, 5), (0.98, 5), (0.99, 5)]:
        print(row(f"qwen yes; core {core}, bridge {bridge}",
                  grade_regions(apply(lambda v: v["said"] is True, verdicts), rows, oof, core, bridge)))

    if args.search:
        print("\npolicy search: keep if (qwen yes and peak >= a) or peak >= b; skip the core >= c, bridge w")
        results = []
        for a, b, c, w in product([0.84, 0.9, 0.95], [0.995, 0.998, 1.01], [0.0, 0.9, 0.95, 0.97, 0.98, 0.99],
                                  [1, 3, 5, 9]):
            if c == 0.0 and w != 1:
                continue
            policy = (lambda a, b: lambda v: (v["said"] is True and v["marker_peak"] >= a)
                      or v["marker_peak"] >= b)(a, b)
            g = grade_regions(apply(policy, verdicts), rows, oof, c, w)
            results.append(((a, b, c, w), g))
        for budget in (5, 10, 15, 20):
            ok = [(p, g) for p, g in results if g["show"] <= budget and g["worst"] <= 60]
            if ok:
                p, g = max(ok, key=lambda pg: (pg[1]["recall"], -pg[1]["show"]))
                print(row(f"best <= {budget:2d} s, worst <= 60: a={p[0]} b={p[1]} c={p[2]} w={p[3]}", g))
            else:
                print(f"  no policy stays under {budget} s with no video over 60 s")
        (DATA / "replay_search.json").write_text(json.dumps([{"policy": p, **g} for p, g in results], indent=1),
                                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
