"""Step 2: decide which of the sampled videos are worth spending a caption fetch on.

Every filter here is a judgement about LABEL QUALITY, because a sponsor segment
we believe is wrong teaches the model something wrong. The funnel is printed so
the cost of each judgement is visible instead of buried in a list comprehension.

The filters, and why:
  not downvoted   SponsorBlock hides a segment once it is voted below zero, so
                  "votes >= 0" is exactly the set SponsorBlock still serves to
                  users. Demanding votes >= 1 would keep only 15% of them: most
                  segments simply never get voted on either way.
  3-120 minutes   Shorter than 3 min is rarely a real sponsor read; longer than
                  2 h is usually a stream, where captions are enormous and the
                  ad conventions are different.
  10-180 s reads  The median read is 49 s. A "segment" of 90 minutes is someone
                  marking a whole video, not an ad.
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent


def keep(v: dict) -> str | None:
    """Return the reason to DROP this video, or None to keep it."""
    if not v["duration"]:
        return "duration unknown"
    if not (180 <= v["duration"] <= 7200):
        return "duration outside 3-120 min"
    if any(vote <= -1 for vote in v["votes"]):
        return "has a downvoted segment"
    if not all(10 <= end - start <= 180 for start, end in v["segments"]):
        return "a segment is not 10-180 s"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=HERE / "data" / "raw_segments.json")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "candidates.json")
    ap.add_argument(
        "--take",
        type=int,
        default=0,
        help="keep at most this many (0 = all). Take a surplus: captions fail on a good share of videos.",
    )
    ap.add_argument(
        "--tail",
        action="store_true",
        help="take the LEAST-reviewed videos instead of the best-evidenced ones. This is the holdout: "
             "segments nobody has voted on come from the thin end of SponsorBlock's attention, which is "
             "where stock sponsor phrasing is least likely and where the extension has to earn its keep.",
    )
    ap.add_argument("--exclude", type=Path, help="a candidates file whose videos must not be picked again")
    args = ap.parse_args()

    videos = json.loads(args.raw.read_text(encoding="utf-8"))
    kept, dropped = [], {}
    for v in videos:
        reason = keep(v)
        if reason:
            dropped[reason] = dropped.get(reason, 0) + 1
        else:
            kept.append(v)

    print(f"{len(videos)} sampled videos")
    for reason, n in sorted(dropped.items(), key=lambda kv: -kv[1]):
        print(f"  -{n:5d}  {reason}")
    print(f"  ={len(kept):5d}  kept, {sum(len(v['segments']) for v in kept)} sponsor segments")

    # Prefer the best-evidenced videos when taking a subset: a locked (VIP-approved)
    # or upvoted segment is a stronger label than an unvoted one.
    if args.exclude:
        already = {v["videoID"] for v in json.loads(args.exclude.read_text(encoding="utf-8"))}
        before = len(kept)
        kept = [v for v in kept if v["videoID"] not in already]
        print(f"  -{before - len(kept):5d}  already in {args.exclude.name}")

    if args.take:
        kept.sort(key=lambda v: (sum(v["locked"]), sum(v["votes"])), reverse=not args.tail)
        kept = kept[: args.take]
        print(f"  ={len(kept):5d}  taken ({'least-reviewed first' if args.tail else 'best-evidenced first'})")

    args.out.write_text(json.dumps(kept, indent=1), encoding="utf-8")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
