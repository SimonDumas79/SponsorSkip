"""Index the Webis YouTube-8M caption archive, and measure its overlap with SponsorBlock.

Why this archive matters: unlike YouTube-Commons (measured 2026-09-20: no
timestamps, and only 0.18% overlap), these are real caption files --

    [00:00.03]what's kinda like everybody walk you
    [00:01.74]back to his nothing to sleep video and

-- auto-generated English, timestamped per line, which is exactly what the
extension reads live. If enough of them carry SponsorBlock segments, the whole
rate-limited fetching problem goes away.

Layout: captions.zip holds 631 part*.zip, each holding ~3,500 caption files
named  /partN/id<VIDEOID>_captions/id<VIDEOID>.auto.en.lrc.  The "id" prefix is
the archive's own; the YouTube id is the remaining 11 characters.

Output: data/webis_index.json -- {video_id: part name} for every video with an
English auto-caption track, so a later step can pull only the parts it needs.
"""

import argparse
import io
import json
import re
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).parent

# id<11-char youtube id>.auto.en.lrc  (also accept .en.lrc for uploaded tracks)
CAPTION = re.compile(r"/id([A-Za-z0-9_-]{11})\.(auto\.)?en\.lrc$")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, default=HERE / "data" / "webis-captions.zip")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "webis_index.json")
    ap.add_argument(
        "--segments",
        type=Path,
        nargs="+",
        default=[HERE / "data" / "raw_segments.json", HERE / "data" / "raw_segments_wide.json"],
    )
    args = ap.parse_args()

    index: dict[str, str] = {}
    auto_only = 0
    started = time.time()
    with zipfile.ZipFile(args.archive) as outer:
        parts = [n for n in outer.namelist() if n.endswith(".zip")]
        for i, part in enumerate(parts, 1):
            try:
                inner = zipfile.ZipFile(io.BytesIO(outer.read(part)))
            except Exception as e:
                print(f"  {part}: unreadable ({str(e)[:60]})")
                continue
            for name in inner.namelist():
                m = CAPTION.search(name)
                if m:
                    index.setdefault(m.group(1), part)
                    auto_only += 1 if m.group(2) else 0
            if i % 100 == 0 or i == len(parts):
                print(f"  {i}/{len(parts)} parts   {len(index):,} videos with English captions   "
                      f"{(time.time() - started) / 60:.1f} min")

    args.out.write_text(json.dumps(index), encoding="utf-8")
    print(f"\n{len(index):,} videos with an English caption track ({auto_only:,} auto-generated) -> {args.out}")

    sponsor = {}
    for path in args.segments:
        if path.exists():
            for v in json.loads(path.read_text(encoding="utf-8")):
                sponsor[v["videoID"]] = v
    hits = [v for v in sponsor if v in index]
    rate = len(hits) / max(len(sponsor), 1)
    segments = sum(len(sponsor[v]["segments"]) for v in hits)

    print(f"\n{len(sponsor):,} SponsorBlock videos sampled uniformly by hash prefix")
    print(f"overlap: {len(hits)} ({100 * rate:.2f}%), carrying {segments} sponsor segments")
    print(f"\nSponsorBlock holds on the order of a million labelled videos, so {100 * rate:.2f}%")
    print(f"implies roughly {int(rate * 1_000_000):,} videos that could be labelled with NO fetching.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
