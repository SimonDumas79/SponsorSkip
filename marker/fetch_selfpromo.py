"""Fetch SponsorBlock's SELF-PROMO segments for the videos the dataset already has.

Why this exists
---------------
sample_segments.py asked SponsorBlock for the "sponsor" category only, but the
extension skips "sponsor" AND "selfpromo" by default (the host's own app, course,
merch, Patreon). Measured on 2026-09-21: the marker's worst "false alarm" was a
93-minute video where it flagged 215 s of the host's own app launch, which the
labels called normal show. The marker was right and the labels were wrong.

So this script asks SponsorBlock, video by video, for both categories, and
writes the self-promo segments to data/selfpromo.json. build_dataset.py merges
them into the labels; nothing else in the pipeline changes.

It uses the same privacy-preserving hash-prefix call as the extension: four hex
characters of the SHA-256 of the video id, never the id itself.

Output: data/selfpromo.json  {videoID: [{"start", "end", "votes", "locked"}, ...]}
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://sponsor.ajay.app/api/skipSegments"
HERE = Path(__file__).parent
UA = "SponsorSkip-research/0.1 (personal, non-commercial)"


def fetch_bucket(prefix: str, timeout: float = 30.0) -> list[dict]:
    query = urllib.parse.urlencode({"categories": json.dumps(["sponsor", "selfpromo"])})
    req = urllib.request.Request(f"{API}/{prefix}?{query}", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise


def video_ids(paths: list[Path]) -> list[str]:
    ids = []
    for p in paths:
        if p.is_file():
            ids += [v["videoID"] for v in json.loads(p.read_text(encoding="utf-8"))]
        elif p.is_dir():
            ids += [c.stem for c in p.glob("*.json")]
    return sorted(set(ids))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, nargs="+",
                    default=[HERE / "data" / "candidates.json", HERE / "data" / "candidates_tail.json"],
                    help="candidate files or caption folders naming the videos to ask about")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "selfpromo.json")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between calls; be a polite guest")
    args = ap.parse_args()

    ids = video_ids(args.videos)
    done = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    todo = [v for v in ids if v not in done]
    print(f"{len(ids)} videos, {len(done)} already fetched, {len(todo)} to ask about", flush=True)

    buckets: dict[str, list[dict]] = {}
    for n, vid in enumerate(todo, 1):
        prefix = hashlib.sha256(vid.encode()).hexdigest()[:4]
        if prefix not in buckets:
            try:
                buckets[prefix] = fetch_bucket(prefix)
            except Exception as e:   # one bad call must not lose the run
                print(f"  {vid}: failed ({e})", file=sys.stderr, flush=True)
                continue
            time.sleep(args.pause)
        match = next((v for v in buckets[prefix] if v["videoID"] == vid), None)
        done[vid] = [
            {"start": s["segment"][0], "end": s["segment"][1], "votes": s.get("votes", 0), "locked": bool(s.get("locked"))}
            for s in (match["segments"] if match else [])
            if s.get("category") == "selfpromo"
        ]
        if n % 25 == 0 or n == len(todo):
            args.out.write_text(json.dumps(done, indent=1), encoding="utf-8")
            print(f"  [{n}/{len(todo)}] {sum(1 for v in done.values() if v)} videos with self-promo so far", flush=True)

    args.out.write_text(json.dumps(done, indent=1), encoding="utf-8")
    with_promo = {k: v for k, v in done.items() if v}
    print(f"\n{len(with_promo)} of {len(done)} videos have self-promo segments "
          f"({sum(len(v) for v in with_promo.values())} segments) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
