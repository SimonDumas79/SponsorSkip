"""When did each video's SponsorBlock segments appear? Prices the community model's leakage. (2026-09-22)

The community model (xenova/sponsorblock-ml) was trained on SponsorBlock's database in early 2022.
A video whose sponsor segments were all submitted AFTER that could not have been in its training
labels (its channel may still have been). SponsorBlock's /api/searchSegments returns each segment's
timeSubmitted for one video id. This is a SponsorBlock call, not a YouTube one; one call per video,
a pause between them.

Output: data/sb_dates.json {videoID: {"first": ms, "last": ms, "n": segments}} (sponsor + selfpromo)

    python marker/sb_dates.py
"""

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = DATA / "sb_dates.json"
API = "https://sponsor.ajay.app/api/searchSegments"
UA = "SponsorSkip-research/0.1 (personal, non-commercial)"
SETS = ("features.npz", "features_tail.npz", "features_holdout2.npz", "features_channels.npz", "features_holdout4.npz",
        "features_holdout3.npz", "features_holdout5.npz")


def fetch(video_id: str) -> list:
    q = urllib.parse.urlencode({"videoID": video_id, "categories": json.dumps(["sponsor", "selfpromo"])})
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r).get("segments", [])
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise


def main() -> int:
    got = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    vids = []
    for f in SETS:
        if (DATA / f).exists():
            vids += [str(v) for v in np.unique(np.load(DATA / f)["video"])]
    todo = [v for v in dict.fromkeys(vids) if v not in got]
    print(f"{len(todo)} videos to ask about", flush=True)
    for n, v in enumerate(todo, 1):
        try:
            segs = fetch(v)
        except Exception as e:
            print(f"  {v}: {e}", flush=True)
            time.sleep(5)
            continue
        times = [s.get("timeSubmitted") for s in segs if s.get("timeSubmitted")]
        got[v] = {"first": min(times) if times else None, "last": max(times) if times else None, "n": len(segs)}
        if n % 50 == 0:
            OUT.write_text(json.dumps(got, indent=1), encoding="utf-8")
            print(f"  [{n}/{len(todo)}]", flush=True)
        time.sleep(1.0)
    OUT.write_text(json.dumps(got, indent=1), encoding="utf-8")
    print(f"done: {len(got)} videos", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
