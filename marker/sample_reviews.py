"""Find product-TALK videos (reviews, unboxings, roundups, tutorials): the false-alarm class. (2026-09-22)

The free tier's false alarms concentrate in videos whose subject is a product; the sponsor-free panel
(sample_negatives.py) is mostly ordinary videos, where false alarms are rare. This collects the hard
class directly with yt-dlp searches (one request per query, no captions), keeping 4-40 minute videos
from channels in none of the existing sets. Many reviews also carry a real sponsor read, so these are
NOT assumed clean: claude_label.py labels them before they are used for training or grading.

Output: data/candidates_reviews.json (the candidates shape, no segments)

    python marker/sample_reviews.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
QUERIES = ["phone review", "smartphone review 2025", "watch review", "headphones review", "laptop review",
           "unboxing", "best gadgets under 50", "tech gear roundup", "board game review", "app review",
           "how to use software tutorial", "car review", "camera review", "product comparison vs"]
PER_QUERY = 20


def search(q: str) -> list[dict]:
    p = subprocess.run([sys.executable, "-m", "yt_dlp", "--flat-playlist", "--no-warnings", "--quiet", "--print",
                        "%(id)s\t%(duration)s\t%(channel)s\t%(channel_id)s\t%(title)s", f"ytsearch{PER_QUERY}:{q}"],
                       capture_output=True, text=True, encoding="utf-8", timeout=120,
                       env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
                       creationflags=0x08000000 if sys.platform == "win32" else 0)
    out = []
    for ln in p.stdout.splitlines():
        parts = ln.split("\t")
        if len(parts) == 5:
            vid, dur, ch, cid, title = parts
            out.append({"videoID": vid, "duration": float(dur) if dur not in ("NA", "") else 0.0,
                        "channel": ch, "channel_id": cid, "title": title, "query": q})
    return out


def main() -> int:
    known_channels = set()
    for f in ("features.npz", "features_tail.npz", "features_holdout2.npz", "features_holdout3.npz", "features_channels.npz"):
        if (DATA / f).exists():
            known_channels |= {str(c) for c in np.unique(np.load(DATA / f)["channel"])}
    seen, out = set(), []
    for q in QUERIES:
        got = search(q)
        kept = 0
        for v in got:
            if v["videoID"] in seen or v["channel_id"] in known_channels or not 240 <= v["duration"] <= 2400:
                continue
            seen.add(v["videoID"])
            out.append({**v, "segments": [], "votes": [], "locked": []})
            kept += 1
        print(f"  {q!r}: {len(got)} found, {kept} kept", flush=True)
        time.sleep(3)
    (DATA / "candidates_reviews.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n{len(out)} product-talk candidates -> data/candidates_reviews.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
