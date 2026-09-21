"""Step 3: fetch each candidate video's English captions with yt-dlp.

This is the slow, rude-if-done-wrong part, so it is deliberately gentle:
  * one video at a time, with a pause between them;
  * yt-dlp runs at BELOW NORMAL priority and with no console window, so it
    never fights the PC for attention while you are using it;
  * a 429 ("Too Many Requests") is answered by backing off and trying again,
    and then by giving up on THAT VIDEO and moving on -- measured 2026-09-20,
    a 429 is usually one awkward video rather than a block on us. Five refusals
    in a row IS a block, and stops the run (Kurzgesagt, 2026-09-19);
  * every video already fetched is skipped, so it can be stopped with Ctrl-C
    and restarted without losing anything.

The caption settings match server/youtube.mjs exactly (same SUB_LANGS, same
json3 parsing, same file preference). That matters more than it looks: if the
model trained on prettier text than the extension feeds it live, its accuracy
here would not survive the trip into production.

Output: data/captions/<videoID>.json  {videoID, title, channel, channel_id,
duration, lines: [{start, text}]}.  Segments are NOT joined in here -- captions
are the expensive artifact and stay reusable on their own.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent

# Matches server/youtube.mjs. NOT "en.*", which also pulls YouTube's machine
# translations into English from every other language (en-pl, en-it, ...).
SUB_LANGS = "en,en-orig,en-US,en-GB"

# Windows: keep yt-dlp out of the way and out of sight.
FLAGS = 0
if os.name == "nt":
    FLAGS = subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW


class RateLimited(Exception):
    """YouTube said 429. Stop everything; do not retry."""


def parse_json3(doc: dict) -> list[dict]:
    """json3 caption events -> [{start, text}], the same shape the server uses."""
    lines = []
    for e in doc.get("events") or []:
        segs = e.get("segs")
        if not isinstance(segs, list) or not isinstance(e.get("tStartMs"), (int, float)):
            continue
        text = " ".join("".join(s.get("utf8", "") for s in segs).split())
        if text:
            lines.append({"start": e["tStartMs"] / 1000, "text": text})
    return lines


def pick_caption_file(files: list[str], video_id: str) -> str | None:
    """English as published first, then the auto-generated original."""
    for name in (f"{video_id}.en.json3", f"{video_id}.en-orig.json3", f"{video_id}.en-en.json3"):
        if name in files:
            return name
    return next((f for f in files if f.startswith(f"{video_id}.en") and f.endswith(".json3")), None)


def fetch(video_id: str, timeout: float = 120.0) -> dict | None:
    """Captions plus the channel, or None when the video has no English captions."""
    with tempfile.TemporaryDirectory(prefix="sponsorskip-") as tmp:
        proc = subprocess.run(
            [
                sys.executable, "-m", "yt_dlp",
                "--skip-download", "--no-simulate", "--no-warnings", "--quiet",
                "--write-subs", "--write-auto-subs", "--sub-langs", SUB_LANGS, "--sub-format", "json3",
                "--sleep-requests", "1", "--sleep-subtitles", "3",
                "--print", "%(title)s\t%(duration)s\t%(channel)s\t%(channel_id)s\t%(language)s",
                "-o", "%(id)s.%(ext)s",
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            cwd=tmp, capture_output=True, text=True, timeout=timeout, creationflags=FLAGS,
        )
        if proc.returncode != 0:
            err = (proc.stderr or "").strip()
            if "429" in err or "Too Many Requests" in err:
                raise RateLimited(err[:200])
            raise RuntimeError(err.splitlines()[-1][:160] if err else f"yt-dlp exited {proc.returncode}")

        fields = (proc.stdout.strip().split("\n")[0].split("\t") + [""] * 5)[:5]
        title, duration, channel, channel_id, language = fields
        name = pick_caption_file(os.listdir(tmp), video_id)
        if not name:
            return None
        lines = parse_json3(json.loads(Path(tmp, name).read_text(encoding="utf-8")))
        if not lines:
            return None
        # The video's spoken language, as YouTube reports it. When it is not English, the "en" track we
        # read is a machine translation: the class behind the worst video in every set so far
        # (measured 2026-09-21: bqtppv75MJg, Russian, language "ru", the en track's URL carries tlang=).
        return {
            "videoID": video_id,
            "title": title or None,
            "channel": channel if channel and channel != "NA" else None,
            "channel_id": channel_id if channel_id and channel_id != "NA" else None,
            "duration": float(duration) if duration.replace(".", "").isdigit() else None,
            "language": language if language and language != "NA" else None,
            "captionFile": name,
            "lines": lines,
        }


BACKOFF = (30, 90)
REFUSALS_BEFORE_STOP = 5  # consecutive 429s that mean YouTube is blocking US, not one video


def fetch_with_backoff(video_id: str) -> dict | None:
    """fetch(), but a 429 waits and tries again. Raises RateLimited once the waits run out."""
    for wait in BACKOFF:
        try:
            return fetch(video_id)
        except RateLimited:
            print(f"    429 on {video_id}; waiting {wait} s before trying again")
            time.sleep(wait)
    return fetch(video_id)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=Path, default=HERE / "data" / "candidates.json")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "captions")
    ap.add_argument("--pause", type=float, default=3.0, help="seconds between videos")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many NEW fetches (0 = all)")
    ap.add_argument("--cooldown", type=float, default=2700, help="seconds to wait out a throttle before resuming")
    ap.add_argument("--max-cooldowns", type=int, default=24, help="how many throttles to sit through before giving up")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    failures_path = args.out.parent / "caption_failures.json"
    failures = json.loads(failures_path.read_text(encoding="utf-8")) if failures_path.exists() else {}

    videos = json.loads(args.candidates.read_text(encoding="utf-8"))
    todo = [v for v in videos if not (args.out / f"{v['videoID']}.json").exists() and v["videoID"] not in failures]
    print(f"{len(videos)} candidates, {len(videos) - len(todo)} already done or failed, {len(todo)} to fetch")

    done = no_captions = failed = refused = cooldowns = 0
    started = time.time()
    try:
        for i, v in enumerate(todo, 1):
            vid = v["videoID"]
            try:
                got = fetch_with_backoff(vid)
            except RateLimited:
                failures[vid] = "429 after backing off"
                failed, refused = failed + 1, refused + 1
                print(f"[{i}/{len(todo)}] {vid}  429 even after backing off; moving on")
                if refused >= REFUSALS_BEFORE_STOP:
                    # A long crawl should wait the throttle out rather than die on it.
                    cooldowns += 1
                    if cooldowns > args.max_cooldowns:
                        print("")
                        print(f"!! throttled {cooldowns} times; giving up for now. Run again later to resume.")
                        break
                    print("")
                    print(f"!! throttled ({refused} refusals in a row). Waiting {args.cooldown / 60:.0f} min, "
                          f"then carrying on. Cooldown {cooldowns} of {args.max_cooldowns}.")
                    time.sleep(args.cooldown)
                    refused = 0
                    for vid_retry in [v for v in failures if failures[v] == "429 after backing off"]:
                        failures.pop(vid_retry, None)  # give the skipped ones another chance later
            except subprocess.TimeoutExpired:
                failures[vid], failed = "timeout", failed + 1
                print(f"[{i}/{len(todo)}] {vid}  TIMEOUT")
            except Exception as e:
                failures[vid], failed = str(e), failed + 1
                print(f"[{i}/{len(todo)}] {vid}  failed: {e}")
            else:
                if got is None:
                    failures[vid], no_captions = "no english captions", no_captions + 1
                    print(f"[{i}/{len(todo)}] {vid}  no English captions")
                else:
                    refused = 0
                    (args.out / f"{vid}.json").write_text(json.dumps(got), encoding="utf-8")
                    done += 1
                    mins = (got["duration"] or 0) / 60
                    print(f"[{i}/{len(todo)}] {vid}  {len(got['lines']):5d} lines  {mins:5.1f} min  {(got['channel'] or '?')[:32]}")
            if args.limit and done >= args.limit:
                break
            time.sleep(args.pause)
    except KeyboardInterrupt:
        print("\nstopped by hand")
    finally:
        failures_path.write_text(json.dumps(failures, indent=1), encoding="utf-8")

    have = len(list(args.out.glob("*.json")))
    print(f"\nfetched {done}, no captions {no_captions}, failed {failed}, in {(time.time() - started) / 60:.1f} min")
    print(f"{have} videos with captions in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
