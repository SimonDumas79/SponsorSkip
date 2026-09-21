"""Pull timestamped English captions out of the Webis archive, into our own format.

Why this is the corpus now: SponsorBlock's labels only cover videos its
contributors happened to mark, which is the coverage bias this whole project
exists to beat, and fetching captions from YouTube is rate-limited to about 35
videos an hour. Webis holds 1,657,005 videos whose auto-generated English
captions are already timestamped, on disk, under CC-BY. Claude supplies the
labels, so SponsorBlock is no longer needed for training at all.

The caption files are LRC:

    [re:Lavf56.40.101]
    [00:00.03]what's kinda like everybody walk you
    [00:01.74]back to his nothing to sleep video and

Note MM can exceed 59 on a long video ([75:30.00] is 75 minutes in), so the
minutes field is parsed as a plain integer rather than clamped.

WHAT WE DO NOT GET: a real duration. The last caption line's time is used as a
floor for it, which is close enough for the "how far into the video" feature and
is recorded as approximate rather than passed off as exact.

Output: data/captions_webis/<id>.json, the same shape fetch_captions.py writes,
so everything downstream is unchanged.
"""

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).parent

TIMED = re.compile(r"^\[(\d+):(\d{2})(?:[.:](\d{1,3}))?\](.*)$")
CAPTION_FILE = re.compile(r"/id([A-Za-z0-9_-]{11})\.(?:auto\.)?en\.lrc$")

# The same six cue patterns features.py uses. Here they only decide which videos
# are worth a labelling pass -- inside a chosen video, EVERY window is swept, so
# an unconventional read in that video is still found.
CUES = re.compile(
    r"\bsponsor(ed|s|ship)?\b|\bbrought to you by\b|\b(use|with) (the )?code\b|"
    r"\bpromo code\b|\blink (in|below)\b|\bdescription below\b|\b\d{1,2}% off\b|"
    r"\bfree trial\b|\bsign up (at|for)\b",
    re.I,
)


def parse_lrc(text: str) -> list[dict]:
    """LRC -> [{start, text}], dropping the metadata tags at the top."""
    lines = []
    for raw in text.splitlines():
        m = TIMED.match(raw.strip())
        if not m:
            continue
        minutes, seconds, frac, body = m.groups()
        body = " ".join(body.split())
        if not body:
            continue
        start = int(minutes) * 60 + int(seconds) + (int(frac) / (10 ** len(frac)) if frac else 0.0)
        lines.append({"start": round(start, 2), "text": body})
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, default=HERE / "data" / "webis-captions.zip")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "captions_webis")
    ap.add_argument("--parts", type=int, default=2, help="how many part-archives to work through")
    ap.add_argument("--min-lines", type=int, default=120, help="skip very short videos")
    ap.add_argument("--min-seconds", type=float, default=240, help="skip videos under this length")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    kept = short = 0
    with_cue = 0

    with zipfile.ZipFile(args.archive) as outer:
        parts = sorted(n for n in outer.namelist() if n.endswith(".zip"))[: args.parts]
        for part in parts:
            inner = zipfile.ZipFile(io.BytesIO(outer.read(part)))
            for name in inner.namelist():
                m = CAPTION_FILE.search(name)
                if not m:
                    continue
                vid = m.group(1)
                target = args.out / f"{vid}.json"
                if target.exists():
                    continue
                lines = parse_lrc(inner.read(name).decode("utf-8", "replace"))
                if len(lines) < args.min_lines or not lines or lines[-1]["start"] < args.min_seconds:
                    short += 1
                    continue
                joined = " ".join(l["text"] for l in lines)
                cue = bool(CUES.search(joined))
                with_cue += cue
                target.write_text(
                    json.dumps({
                        "videoID": vid,
                        "title": None,
                        "channel": None,
                        "channel_id": None,
                        # No real duration in this archive; the last line is a floor.
                        "duration": round(lines[-1]["start"] + 5, 2),
                        "durationApproximate": True,
                        "source": "webis",
                        "captionFile": name.rsplit("/", 1)[-1],
                        "hasCue": cue,
                        "lines": lines,
                    }),
                    encoding="utf-8",
                )
                kept += 1
            print(f"  {part}: {kept:,} kept so far ({with_cue:,} with a sponsor cue), {short:,} too short")

    print(f"\n{kept:,} usable transcripts -> {args.out}")
    print(f"{with_cue:,} ({100 * with_cue / max(kept, 1):.0f}%) contain at least one stock sponsor phrase")
    print("\nThat flag picks which VIDEOS are worth a labelling pass, not which windows")
    print("Claude sees. Inside a chosen video every window is swept, so an unconventional")
    print("read in the same video is still found. Label a random slice too, or videos whose")
    print("only read is unconventional never enter the training set at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
