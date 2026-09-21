"""Step 1 of the dataset: find videos that have human-verified sponsor segments.

Where the labels come from
--------------------------
SponsorBlock users mark sponsor reads by hand and vote on each other's marks.
That is our ground truth: exact start and end seconds, already agreed by people.
We never hand-label anything.

Why the API and not the database dump
-------------------------------------
SponsorBlock turned off the CSV dump downloads for bandwidth (2026-09-20: "CSV
downloads have been disabled, please use sb-mirror"), and sb-mirror needs rsync,
which this PC does not have. The hash-prefix endpoint gives the same segments a
few dozen videos per call. It is also the privacy-preserving call: we send four
hex characters of the SHA-256 of a video id, never a video id, so SponsorBlock
cannot tell which video we are asking about. The extension already talks to this
same endpoint.

Output: data/raw_segments.json, a list of {videoID, duration, segments:[[s,e]]}.
Nothing is filtered here on purpose -- the filtering is the next step, so the
funnel from "what SponsorBlock has" to "what we can train on" stays visible.
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://sponsor.ajay.app/api/skipSegments"
HERE = Path(__file__).parent
UA = "SponsorSkip-research/0.1 (personal, non-commercial)"


def fetch_prefix(prefix: str, timeout: float = 30.0):
    """All sponsor segments for every video whose id-hash starts with `prefix`."""
    query = urllib.parse.urlencode({"categories": json.dumps(["sponsor"])})
    req = urllib.request.Request(f"{API}/{prefix}?{query}", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:  # no videos under this prefix
            return []
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefixes", type=int, default=40, help="how many 4-hex prefixes to sweep")
    ap.add_argument("--seed", type=int, default=20260920, help="fixed so the same sample can be rebuilt")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between calls; be a polite guest")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "raw_segments.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # 4 hex characters = 1 of 65,536 buckets, so a random draw is a random
    # sample of SponsorBlock, not a sample of what is popular this week.
    prefixes = [f"{rng.randrange(16**4):04x}" for _ in range(args.prefixes)]

    videos, seen = [], set()
    for i, prefix in enumerate(prefixes, 1):
        try:
            batch = fetch_prefix(prefix)
        except Exception as e:  # one bad prefix must not lose the whole sweep
            print(f"  {prefix}: failed ({e})", file=sys.stderr)
            continue
        new = 0
        for v in batch:
            if v["videoID"] in seen:
                continue
            seen.add(v["videoID"])
            segments = [list(s["segment"]) for s in v["segments"]]
            votes = [s.get("votes", 0) for s in v["segments"]]
            locked = [bool(s.get("locked")) for s in v["segments"]]
            duration = max((s.get("videoDuration") or 0) for s in v["segments"])
            videos.append(
                {"videoID": v["videoID"], "duration": duration, "segments": segments, "votes": votes, "locked": locked}
            )
            new += 1
        print(f"[{i}/{len(prefixes)}] {prefix}: {len(batch)} videos, {new} new  (running total {len(videos)})")
        time.sleep(args.pause)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(videos, indent=1), encoding="utf-8")
    total_segments = sum(len(v["segments"]) for v in videos)
    print(f"\n{len(videos)} videos, {total_segments} sponsor segments -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
