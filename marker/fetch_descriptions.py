"""Fetch each labelled video's description: the sponsor is nearly always named there.

Metadata only (no captions), one yt-dlp call per video with a pause between
them, BELOW NORMAL priority and no window; stops on the first 429 so it never
competes with a running caption crawl. Resumable: videos already fetched are
skipped, and descriptions_probe.json (the first 25) is folded in.

Output: data/descriptions.json  {videoID: description or null}

    python marker/fetch_descriptions.py                 # every video in every feature set
    python marker/fetch_descriptions.py --pause 10
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = DATA / "descriptions.json"
SETS = ("features.npz", "features_tail.npz", "features_holdout2.npz", "features_holdout3.npz",
        "features_channels.npz")
FLAGS = subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def fetch(video_id: str) -> str | None:
    p = subprocess.run([sys.executable, "-m", "yt_dlp", "--skip-download", "--no-warnings", "--quiet",
                        "--print", "%(description)s", f"https://www.youtube.com/watch?v={video_id}"],
                       capture_output=True, text=True, encoding="utf-8", timeout=90, creationflags=FLAGS,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    if p.returncode != 0:
        err = (p.stderr or "").strip()
        if "429" in err or "Too Many Requests" in err:
            raise RuntimeError("429")
        print(f"  {video_id}: {err.splitlines()[-1][:120] if err else 'yt-dlp failed'}", flush=True)
        return None
    return p.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pause", type=float, default=6.0)
    args = ap.parse_args()

    videos = []
    for name in SETS:
        if (DATA / name).exists():
            videos += [str(v) for v in np.unique(np.load(DATA / name)["video"])]
    got = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    probe = DATA / "descriptions_probe.json"
    if probe.exists():
        for k, v in json.loads(probe.read_text(encoding="utf-8")).items():
            got.setdefault(k, v)
    # a None is a failed fetch, retried: on 2026-09-22 100 of 169 were lost to yt-dlp writing cp1252 into a
    # pipe read as UTF-8, and the decode error happened in subprocess's reader thread, so nothing failed loudly
    todo = [v for v in dict.fromkeys(videos) if got.get(v) is None]
    print(f"{len(dict.fromkeys(videos))} videos, {len(got)} already fetched, {len(todo)} to fetch", flush=True)
    for n, vid in enumerate(todo, 1):
        try:
            got[vid] = fetch(vid)
        except RuntimeError:
            print(f"[{n}/{len(todo)}] {vid}: 429 -- stopping; run again later", flush=True)
            break
        OUT.write_text(json.dumps(got, ensure_ascii=False, indent=0), encoding="utf-8")
        if n % 10 == 0 or n == len(todo):
            print(f"[{n}/{len(todo)}] {sum(1 for v in got.values() if v)} descriptions on disk", flush=True)
        time.sleep(args.pause)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
