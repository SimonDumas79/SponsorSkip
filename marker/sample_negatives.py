"""Find SPONSOR-FREE videos, so false alarms can be measured the way real use sees them. (2026-09-22)

Every video in the training and test sets has at least one read (they were sampled from SponsorBlock
sponsor segments), so the free tier has never been graded on a video it should leave alone, which is
most videos. The research round found the way in: the same privacy-preserving hash-prefix call, asked
for EVERY category, returns videos that the community marked (intro, outro, filler...) but never
marked a sponsor or self-promotion in. A video with two or more such marks in two or more categories,
or a locked or upvoted one, has probably been watched through by someone who would have marked a read:
that is tier B, the one trusted for a headline number. Absence of a mark is still not proof, so a
sample gets checked by Claude before any number from this set is trusted.

Output: data/candidates_negatives.json, in the candidates shape with no segments, so
fetch_captions.py can crawl it:  [{videoID, duration, segments: [], tier, categories}]

    python marker/sample_negatives.py --prefixes 12
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

HERE = Path(__file__).parent
API = "https://sponsor.ajay.app/api/skipSegments"
UA = "SponsorSkip-research/0.1 (personal, non-commercial)"
CATEGORIES = ["sponsor", "selfpromo", "interaction", "intro", "outro", "preview", "hook", "filler",
              "music_offtopic", "exclusive_access", "poi_highlight", "chapter"]
READS = {"sponsor", "selfpromo", "exclusive_access"}


def fetch_prefix(prefix: str) -> list:
    query = urllib.parse.urlencode({"categories": json.dumps(CATEGORIES),
                                    "actionTypes": json.dumps(["skip", "mute", "full", "poi", "chapter"])})
    req = urllib.request.Request(f"{API}/{prefix}?{query}", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefixes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--pause", type=float, default=1.0)
    ap.add_argument("--min-minutes", type=float, default=4.0)
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=HERE / "data" / "candidates_negatives.json")
    args = ap.parse_args()

    known = set()
    for f in ("candidates.json", "candidates_tail.json", "candidates_channels.json"):
        p = HERE / "data" / f
        if p.exists():
            known |= {c["videoID"] for c in json.loads(p.read_text(encoding="utf-8"))}
    rng = random.Random(args.seed)
    prefixes = [f"{rng.randrange(16 ** 4):04x}" for _ in range(args.prefixes)]
    out, seen = [], set()
    counts = {"videos": 0, "with_reads": 0, "no_reads": 0, "tier_a": 0, "tier_b": 0}
    for i, prefix in enumerate(prefixes, 1):
        try:
            batch = fetch_prefix(prefix)
        except Exception as e:
            print(f"  {prefix}: failed ({e})", file=sys.stderr)
            continue
        for v in batch:
            vid = v["videoID"]
            if vid in seen or vid in known:
                continue
            seen.add(vid)
            counts["videos"] += 1
            segs = v["segments"]
            cats = {s["category"] for s in segs}
            if cats & READS:
                counts["with_reads"] += 1
                continue
            counts["no_reads"] += 1
            duration = max((s.get("videoDuration") or 0) for s in segs)
            if not args.min_minutes * 60 <= duration <= args.max_minutes * 60:
                continue
            trusted = (len(segs) >= 2 and len(cats) >= 2) or any(s.get("locked") or s.get("votes", 0) > 0 for s in segs)
            tier = "B" if trusted else "A"
            counts[f"tier_{tier.lower()}"] += 1
            out.append({"videoID": vid, "duration": duration, "segments": [], "votes": [], "locked": [],
                        "tier": tier, "categories": sorted(cats)})
        print(f"[{i}/{len(prefixes)}] {prefix}: {len(batch)} videos; running {counts}", flush=True)
        time.sleep(args.pause)
    rng.shuffle(out)
    out.sort(key=lambda c: c["tier"] != "B")   # tier B first, so a crawl with a limit takes the trusted ones
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n{len(out)} sponsor-free candidates ({counts['tier_b']} tier B, {counts['tier_a']} tier A) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
