"""Does the free tier survive a different caption rhythm? The first question for Twitch and TikTok.

YouTube's auto-captions arrive as short lines, about 2.4 s each, and every window in
features.py is counted in LINES (4 either side for meaning, 5 for the seam, 10
back for the sharpest seam). Twitch VODs have no captions at all, so they would
be transcribed locally (Whisper), and Whisper writes longer segments, about 6-8 s.
If the windows are really tuned to 2.4 s lines, the same model should degrade on
longer ones, and the fix is to count the windows in seconds.

This takes the holdout-2 videos, merges their caption lines into segments of at
least SECONDS each (the text joined, the first line's start kept), labels them
from the same SponsorBlock segments, rebuilds the 395 numbers with the same
functions, runs the graded free tier unchanged, and grades it in seconds of ad
time and show, so the rhythms are directly comparable.

    python marker/cadence_check.py
"""

import argparse
import json

import numpy as np
import torch

from build_dataset import label_video
from features import embed_lines, video_features
from predict import BUNDLE, FreeTier
from replay import HEADER, grade_regions, row
from train import DATA, Rows


def merged(caps: dict, seconds: float) -> dict:
    """The same captions with consecutive lines joined until each segment spans at least `seconds`."""
    if seconds <= 0:
        return caps
    out, cur = [], None
    for line in caps["lines"]:
        if cur is None:
            cur = {"start": line["start"], "text": line["text"]}
        elif line["start"] - cur["start"] < seconds:
            cur["text"] += " " + line["text"]
        else:
            out.append(cur)
            cur = {"start": line["start"], "text": line["text"]}
    if cur:
        out.append(cur)
    return dict(caps, lines=out)


def spans_for(vid: str, candidates: dict, selfpromo: dict) -> list[tuple[float, float, str]]:
    spans = [(s, e, "sponsor") for s, e in candidates.get(vid, [])]
    spans += [(p["start"], p["end"], "selfpromo") for p in selfpromo.get(vid, [])
              if p["votes"] >= 0 and 5 <= p["end"] - p["start"] <= 300]   # the rule build_dataset.py uses
    return spans


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="examples_holdout2.jsonl", help="whose videos to re-rhythm")
    args = ap.parse_args()
    torch.set_num_threads(4)

    videos = sorted({json.loads(l)["videoID"] for l in open(DATA / args.set, encoding="utf-8")})
    candidates = {v["videoID"]: v["segments"] for v in json.loads((DATA / "candidates.json").read_text(encoding="utf-8"))}
    selfpromo = json.loads((DATA / "selfpromo.json").read_text(encoding="utf-8"))
    tier = FreeTier(BUNDLE)
    print(f"{len(videos)} videos from {args.set}; the graded free tier, unchanged")
    print(HEADER)
    for seconds in (0, 4, 6, 8):
        parts, kept, lengths = [], {}, []
        for vid in videos:
            caps = merged(json.loads((DATA / "captions" / f"{vid}.json").read_text(encoding="utf-8")), seconds)
            lines = label_video(caps, spans_for(vid, candidates, selfpromo))
            vectors = embed_lines([r["text"] for r in lines], tier.encoder, 128)
            X = video_features(vectors, lines, caps.get("duration") or 0.0).astype(np.float32)
            n = len(lines)
            one = Rows(X, np.array([r["label"] for r in lines], dtype=np.int8), np.array([vid] * n),
                       np.array([lines[0]["channel_id"]] * n), np.array([r["start"] for r in lines], dtype=np.float32),
                       np.array(["cadence"] * n))
            kept[vid] = tier.regions(one)
            parts.append(one)
            starts = one.start_seconds
            lengths.extend(np.diff(starts).tolist())
        rows = Rows(*(np.concatenate([getattr(p, f) for p in parts]) for f in
                      ("X", "y", "video", "channel", "start_seconds", "split")))
        label = "original captions" if seconds == 0 else f"lines merged to >= {seconds} s"
        print(row(f"{label} (median line {np.median(lengths):.1f} s)", grade_regions(kept, rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
