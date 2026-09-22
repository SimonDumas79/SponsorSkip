"""Read two free signals from each video's watch page: the "Includes paid promotion" flag and its chapters. (2026-09-22)

From the research round: YouTube shows "Includes paid promotion" when a creator ticks the paid-promotion
box, and the watch page carries it in ytInitialPlayerResponse at
playerOverlays.playerOverlayRenderer.paidContentOverlay.paidContentOverlayRenderer. yt-dlp does not
expose it. The extension's content script can read it on the live page at no cost, so it can serve as a
whole-video hint. Creator chapters (chapterRenderer entries in ytInitialData) sometimes name the read
("Sponsor"), with near-exact edges.

One plain GET of the watch page per video, a pause between them, stopping at the first 429. Resumable.

Output: data/watch_meta.json {videoID: {"paid": bool, "paid_text": str|null,
                                        "chapters": [{"title", "start"}], "ok": bool}}

    python -u marker/fetch_watch_meta.py                         # pooled + channel set
    python -u marker/fetch_watch_meta.py --videos IJx-HT7A2ec    # a test
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = DATA / "watch_meta.json"
SETS = ("features.npz", "features_tail.npz", "features_holdout2.npz", "features_channels.npz", "features_holdout4.npz")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/140.0 Safari/537.36",
           "Accept-Language": "en-US,en;q=0.9", "Cookie": "CONSENT=YES+1"}


def fetch(video_id: str) -> str:
    req = urllib.request.Request(f"https://www.youtube.com/watch?v={video_id}&hl=en", headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def parse(html: str) -> dict:
    ok = "ytInitialPlayerResponse" in html
    m = re.search(r'"paidContentOverlayRenderer":\{"text":\{"simpleText":"([^"]*)"', html)
    paid = "paidContentOverlayRenderer" in html
    chapters, seen = [], set()
    for cm in re.finditer(r'"chapterRenderer":\{"title":\{"simpleText":"((?:[^"\\]|\\.)*)"\},"timeRangeStartMillis":(\d+)', html):
        title, start = json.loads(f'"{cm.group(1)}"'), int(cm.group(2)) / 1000
        if (title, start) not in seen:
            seen.add((title, start))
            chapters.append({"title": title, "start": start})
    return {"paid": paid, "paid_text": m.group(1) if m else None, "chapters": chapters, "ok": ok}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="*", help="only these ids (printed, not saved)")
    ap.add_argument("--pause", type=float, default=4.0)
    args = ap.parse_args()
    if args.videos:
        for v in args.videos:
            meta = parse(fetch(v))
            print(v, json.dumps(meta)[:400])
            time.sleep(args.pause)
        return 0

    got = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    videos = []
    for name in SETS:
        if (DATA / name).exists():
            videos += [str(v) for v in np.unique(np.load(DATA / name)["video"])]
    todo = [v for v in dict.fromkeys(videos) if not got.get(v, {}).get("ok")]
    print(f"{len(dict.fromkeys(videos))} videos, {len(todo)} to fetch", flush=True)
    for n, vid in enumerate(todo, 1):
        try:
            got[vid] = parse(fetch(vid))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"  429 at {vid}: stopping; run again later to resume", flush=True)
                break
            got[vid] = {"paid": None, "paid_text": None, "chapters": [], "ok": False, "error": e.code}
        except Exception as e:   # one bad page must not lose the run
            got[vid] = {"paid": None, "paid_text": None, "chapters": [], "ok": False, "error": str(e)[:80]}
        if n % 25 == 0 or n == len(todo):
            OUT.write_text(json.dumps(got, indent=1), encoding="utf-8")
            ok = [m for m in got.values() if m.get("ok")]
            print(f"  [{n}/{len(todo)}] {len(ok)} pages read; paid flag on {sum(m['paid'] for m in ok)}, "
                  f"chapters on {sum(bool(m['chapters']) for m in ok)}", flush=True)
        time.sleep(args.pause)
    OUT.write_text(json.dumps(got, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
