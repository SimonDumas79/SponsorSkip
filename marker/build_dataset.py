"""Step 4: turn captions + SponsorBlock timings into labelled examples.

The idea in one paragraph
-------------------------
A transcript is a list of caption lines, each with a start time. SponsorBlock
tells us "the sponsor read runs from 1254.8 s to 1322.4 s". So every caption
line whose time falls inside that range is an example of sponsor-read text
(label 1), and every other line is an example of normal show text (label 0).
Nobody labels anything by hand; the arithmetic does it.

Each example carries CONTEXT as well as its own line, because a single caption
line is only a few words ("and that's why we") and means very little alone. The
context is the handful of lines before and after it.

Two extra flags per example, which matter later:
  is_start    this is the FIRST line of a sponsor read. Start accuracy is the
              metric the whole project is judged on, so the starts are marked.
  is_resume   this is the first line where the show comes back.

The split
---------
Videos are split BY CHANNEL, never at random. If Dwarkesh episodes appeared in
both training and validation, the model could learn "this host's ad voice" and
score wonderfully while being useless on a channel it has never heard. Splitting
by channel forces the validation number to mean "works on a NEW channel".

Output: data/examples.jsonl  (one JSON object per caption line)
        data/split.json      (which channel went to train / val)
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent

CONTEXT_BEFORE = 4  # caption lines of run-up given to the model
CONTEXT_AFTER = 4


def line_span(lines: list[dict], i: int, fallback: float = 3.0) -> tuple[float, float]:
    """When a caption line is on screen: from its start to the next line's start."""
    start = lines[i]["start"]
    end = lines[i + 1]["start"] if i + 1 < len(lines) else start + fallback
    return start, max(end, start + 0.1)


def label_video(caps: dict, segments: list[list[float]]) -> list[dict]:
    """One example per caption line, labelled from the SponsorBlock segments."""
    lines = caps["lines"]
    rows = []
    for i, line in enumerate(lines):
        start, end = line_span(lines, i)
        middle = (start + end) / 2
        inside = any(s <= middle <= e for s, e in segments)
        rows.append(
            {
                "videoID": caps["videoID"],
                "channel_id": caps["channel_id"] or caps["channel"] or "unknown",
                "i": i,
                "start": round(start, 2),
                "text": line["text"],
                "context": " ".join(
                    l["text"] for l in lines[max(0, i - CONTEXT_BEFORE) : i + CONTEXT_AFTER + 1]
                ),
                "label": int(inside),
                "is_start": 0,
                "is_resume": 0,
            }
        )

    # Mark the edges: the first sponsor line of each run, and the line after it ends.
    for i, row in enumerate(rows):
        if row["label"] and (i == 0 or not rows[i - 1]["label"]):
            row["is_start"] = 1
        if not row["label"] and i > 0 and rows[i - 1]["label"]:
            row["is_resume"] = 1
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captions", type=Path, default=HERE / "data" / "captions")
    ap.add_argument("--candidates", type=Path, default=HERE / "data" / "candidates.json")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "examples.jsonl")
    ap.add_argument("--val-fraction", type=float, default=0.25, help="share of CHANNELS held out for validation")
    ap.add_argument(
        "--all-to",
        default=None,
        help="put every row in this one split instead of dividing into train and val. Used for the "
             "tail holdout, which is a separate set of videos scored on its own and never trained on.",
    )
    ap.add_argument("--seed", type=int, default=20260920)
    args = ap.parse_args()

    segments_by_video = {v["videoID"]: v["segments"] for v in json.loads(args.candidates.read_text(encoding="utf-8"))}

    rows_by_channel: dict[str, list[dict]] = defaultdict(list)
    videos = skipped = 0
    for path in sorted(args.captions.glob("*.json")):
        caps = json.loads(path.read_text(encoding="utf-8"))
        segments = segments_by_video.get(caps["videoID"])
        if not segments:
            skipped += 1
            continue
        rows = label_video(caps, segments)
        # A video where the labels found nothing is a labelling failure (caption
        # times not matching the segment times), not a video without a sponsor.
        if not any(r["label"] for r in rows):
            skipped += 1
            continue
        rows_by_channel[rows[0]["channel_id"]].extend(rows)
        videos += 1

    channels = sorted(rows_by_channel)
    if args.all_to:
        val_channels, train_channels = set(), set(channels)
    else:
        random.Random(args.seed).shuffle(channels)
        cut = max(1, int(len(channels) * args.val_fraction))
        val_channels, train_channels = set(channels[:cut]), set(channels[cut:])

    with args.out.open("w", encoding="utf-8") as f:
        for channel, rows in rows_by_channel.items():
            split = args.all_to or ("val" if channel in val_channels else "train")
            for row in rows:
                row["split"] = split
                f.write(json.dumps(row) + "\n")

    total = sum(len(r) for r in rows_by_channel.values())
    positives = sum(1 for rs in rows_by_channel.values() for r in rs if r["label"])
    starts = sum(1 for rs in rows_by_channel.values() for r in rs if r["is_start"])
    print(f"{videos} videos across {len(channels)} channels ({skipped} skipped)")
    print(f"{total:,} caption lines, {positives:,} inside a sponsor read ({100 * positives / max(total, 1):.1f}%), {starts} read starts")
    if args.all_to:
        print(f"every row marked split '{args.all_to}' -- {len(channels)} channels, scored on its own")
    else:
        print(f"train: {len(train_channels)} channels   val: {len(val_channels)} channels (never seen in training)")
    print(f"-> {args.out}")

    (args.out.parent / "split.json").write_text(
        json.dumps({"train": sorted(train_channels), "val": sorted(val_channels)}, indent=1), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
