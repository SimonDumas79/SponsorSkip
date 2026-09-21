"""Candidates from named channels: their recent videos that SponsorBlock has sponsor or self-promo marks for.

Simon's pick, 2026-09-21: Sabine Hossenfelder (short science videos with a sponsor
read AND her own courses), Wes Roth, How Money Works. Targeted real data, English,
with short reads, instead of more of the random SponsorBlock tail.

Two cheap steps before any caption is fetched:
  1. yt-dlp lists the channel's videos (one flat-playlist request per channel, no
     captions, no video pages);
  2. SponsorBlock's hash-prefix endpoint gives each video's sponsor and self-promo
     segments (four hex characters of the id's hash, never the id itself).
Videos with at least one segment are kept, most recent first, up to --per-channel.

Output: data/candidates_channels.json (the candidates.json shape, sponsor segments only,
so build_dataset.py can read it) and the self-promo segments merged into data/selfpromo.json.
Then: python marker/fetch_captions.py --candidates marker/data/candidates_channels.json
                                     --out marker/data/captions_channels --pause 60

    python marker/channel_videos.py @SabineHossenfelder @WesRoth @HowMoneyWorks --per-channel 60
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
API = "https://sponsor.ajay.app/api/skipSegments"
UA = "SponsorSkip-research/0.1 (personal, non-commercial)"
FLAGS = subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def list_videos(handle: str, limit: int) -> list[dict]:
    """The channel's most recent videos: id, duration, title. One request, no captions."""
    p = subprocess.run([sys.executable, "-m", "yt_dlp", "--flat-playlist", "--no-warnings", "--quiet",
                        "--playlist-end", str(limit), "--print", "%(id)s\t%(duration)s\t%(title)s",
                        f"https://www.youtube.com/{handle}/videos"],
                       capture_output=True, text=True, encoding="utf-8", timeout=180, creationflags=FLAGS)
    if p.returncode != 0:
        raise SystemExit(f"{handle}: yt-dlp failed: {(p.stderr or '').strip()[-200:]}")
    out = []
    for line in p.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0]:
            dur = float(parts[1]) if parts[1].replace(".", "").isdigit() else 0.0
            out.append({"videoID": parts[0], "duration": dur, "title": parts[2] if len(parts) > 2 else ""})
    return out


_prefix_cache: dict[str, list] = {}


def segments_for(video_id: str, pause: float) -> list[dict]:
    prefix = hashlib.sha256(video_id.encode()).hexdigest()[:4]
    if prefix not in _prefix_cache:
        query = urllib.parse.urlencode({"categories": json.dumps(["sponsor", "selfpromo"])})
        req = urllib.request.Request(f"{API}/{prefix}?{query}", headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                _prefix_cache[prefix] = json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            _prefix_cache[prefix] = []
        time.sleep(pause)
    match = next((v for v in _prefix_cache[prefix] if v["videoID"] == video_id), None)
    return match["segments"] if match else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("handles", nargs="+", help="@SabineHossenfelder, @WesRoth, ...")
    ap.add_argument("--per-channel", type=int, default=60, help="videos with segments to keep per channel")
    ap.add_argument("--scan", type=int, default=200, help="most recent videos to look at per channel")
    ap.add_argument("--pause", type=float, default=0.5, help="seconds between SponsorBlock calls")
    ap.add_argument("--out", type=Path, default=DATA / "candidates_channels.json")
    ap.add_argument("--selfpromo", type=Path, default=DATA / "selfpromo.json")
    args = ap.parse_args()

    candidates = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else []
    known = {c["videoID"] for c in candidates}
    selfpromo = json.loads(args.selfpromo.read_text(encoding="utf-8")) if args.selfpromo.exists() else {}
    for handle in args.handles:
        videos = list_videos(handle, args.scan)
        kept = 0
        for v in videos:
            if kept >= args.per_channel:
                break
            if v["videoID"] in known:
                kept += 1
                continue
            segs = segments_for(v["videoID"], args.pause)
            sponsor = [s for s in segs if s.get("category") == "sponsor" and s.get("votes", 0) >= 0]
            promo = [s for s in segs if s.get("category") == "selfpromo" and s.get("votes", 0) >= 0]
            if not sponsor and not promo:
                continue
            duration = v["duration"] or max((s.get("videoDuration") or 0) for s in segs)
            candidates.append({"videoID": v["videoID"], "duration": duration, "channel": handle, "title": v["title"],
                               "segments": [list(s["segment"]) for s in sponsor],
                               "votes": [s.get("votes", 0) for s in sponsor],
                               "locked": [bool(s.get("locked")) for s in sponsor]})
            if promo:
                selfpromo[v["videoID"]] = [{"start": s["segment"][0], "end": s["segment"][1],
                                            "votes": s.get("votes", 0), "locked": bool(s.get("locked"))} for s in promo]
            known.add(v["videoID"])
            kept += 1
        with_sponsor = sum(1 for c in candidates if c.get("channel") == handle and c["segments"])
        with_promo = sum(1 for c in candidates if c.get("channel") == handle and c["videoID"] in selfpromo)
        print(f"{handle}: {len(videos)} recent videos scanned, {kept} kept "
              f"({with_sponsor} with a sponsor read, {with_promo} with self-promo)", flush=True)
        args.out.write_text(json.dumps(candidates, ensure_ascii=False, indent=1), encoding="utf-8")
        args.selfpromo.write_text(json.dumps(selfpromo, indent=1), encoding="utf-8")
    print(f"\n{len(candidates)} candidates -> {args.out}; self-promo merged into {args.selfpromo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
