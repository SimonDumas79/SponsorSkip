"""Step 7: grade a marker, on the only measure that matters.

What is being measured
----------------------
NOT "what fraction of caption lines did it get right". A model that answers
"no" to every line gets about 95% of lines right and is worthless, because only
one line in twenty is inside a sponsor read.

What counts is REGION RECALL: of all the real sponsor reads, how many did the
marker flag ANYWHERE inside? That is the question the pipeline actually asks of
it, because one flag anywhere inside a read is enough -- Claude is then handed
the window around it and finds the exact edges itself.

Precision is reported beside it, but it is a cost, not a gate: every false
region costs one cheap Claude window (about 7 s), while every missed region
costs a whole sponsor read played at full volume.

Two things can be graded
------------------------
  --baseline          the six cue patterns from features.py, nothing learned.
                      This is the number a trained model has to beat.
  --scores FILE.npy   one probability per row of the validation split, in the
                      same order features.npz stores them. Have train.py write
                      that file and this script will grade it. Saving a plain
                      array keeps your training script free to be built any way
                      you like -- nothing here needs to know what is inside it.

Recall is also reported per channel, because the average hides the thing we
care about: a marker that scores 95% by being perfect on the big channels and
blind on the small ones is exactly the failure the channel split exists to
expose.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def regions(labels: np.ndarray) -> list[tuple[int, int]]:
    """The runs of consecutive 1s -- i.e. where the real sponsor reads are."""
    out, start = [], None
    for i, v in enumerate(labels):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(labels)))
    return out


def score_video(labels: np.ndarray, flagged: np.ndarray, slack: int) -> tuple[int, int, int, int]:
    """(reads found, reads total, regions flagged, regions flagged that were real)."""
    true_regions = regions(labels)
    found = 0
    for lo, hi in true_regions:
        window = flagged[max(0, lo - slack) : hi + slack]
        found += int(window.any())

    flagged_regions = regions(flagged.astype(np.int8))
    right = 0
    for lo, hi in flagged_regions:
        window = labels[max(0, lo - slack) : hi + slack]
        right += int(window.any())
    return found, len(true_regions), len(flagged_regions), right


def smooth(flags: np.ndarray, window: int) -> np.ndarray:
    """Bridge one-line gaps: a read is a run, not a scatter of single lines."""
    if window <= 1:
        return flags
    padded = np.convolve(flags.astype(np.float32), np.ones(window), mode="same")
    return (padded > 0).astype(np.int8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=HERE / "data" / "features.npz")
    ap.add_argument("--baseline", action="store_true", help="grade the cue patterns instead of a model")
    ap.add_argument("--scores", type=Path, help="a .npy of one probability per validation row")
    ap.add_argument("--threshold", type=float, default=0.5, help="flag a line when its probability is above this")
    ap.add_argument("--slack", type=int, default=3, help="caption lines of slack when matching a flag to a read")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    data = np.load(args.features, allow_pickle=False)
    names = list(data["feature_names"])
    mask = data["split"] == args.split
    videos, labels, channels = data["video"][mask], data["y"][mask], data["channel"][mask]

    if args.baseline:
        cue_columns = [i for i, n in enumerate(names) if n.startswith("cue_")]
        flags = (data["X"][mask][:, cue_columns].sum(axis=1) > 0).astype(np.int8)
        what = "cue patterns (no model)"
    elif args.scores:
        probs = np.load(args.scores)
        if len(probs) != mask.sum():
            raise SystemExit(f"{args.scores} has {len(probs)} scores but the {args.split} split has {mask.sum()} rows")
        flags = (probs >= args.threshold).astype(np.int8)
        what = f"{args.scores.name} at threshold {args.threshold}"
    else:
        raise SystemExit("choose --baseline or --scores FILE.npy")

    by_channel: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    totals = [0, 0, 0, 0]
    for vid in np.unique(videos):
        rows = videos == vid
        channel = str(channels[rows][0])
        for i, v in enumerate(score_video(labels[rows], smooth(flags[rows], 3), args.slack)):
            totals[i] += v
            by_channel[channel][i] += v

    found, total, flagged_n, right = totals
    recall = 100 * found / max(total, 1)
    precision = 100 * right / max(flagged_n, 1)

    print(f"grading: {what}")
    print(f"split '{args.split}': {len(np.unique(videos))} videos, {total} real sponsor reads")
    print()
    print(f"  REGION RECALL   {recall:5.1f}%   ({found} of {total} reads flagged)      <- the gate is 90%")
    print(f"  precision       {precision:5.1f}%   ({right} of {flagged_n} flagged regions were real)")
    print(f"  cost            {flagged_n / max(len(np.unique(videos)), 1):.1f} Claude windows per video")
    print()
    print("A read is 'flagged' when any line inside it, give or take "
          f"{args.slack} caption lines, scored above the threshold.")
    # The average hides the failure that matters, so show the channels it fails on.
    missed = sorted(
        ((c, f, t) for c, (f, t, _, _) in by_channel.items() if f < t),
        key=lambda r: (r[1] / max(r[2], 1), -r[2]),
    )
    print("")
    if missed:
        print(f"channels with a missed read ({len(missed)} of {len(by_channel)}):")
        for channel, f, t in missed[:12]:
            print(f"  {f}/{t}  {channel}")
    else:
        print(f"no missed reads on any of the {len(by_channel)} channels in this split.")

    if recall < 90:
        print(f"\nBelow the 90% gate: {total - found} reads would have played in full.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
