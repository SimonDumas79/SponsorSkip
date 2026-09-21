"""How many SponsorBlock-labelled videos have a transcript in YouTube-Commons?

This answers whether the public dataset can replace fetching captions from
YouTube, which is rate-limited to roughly 35 videos an hour.

The SponsorBlock sample is drawn by hash prefix, so it is a uniform random
sample of their database. That means the overlap measured on it IS the overlap
rate for the whole database, and multiplying by SponsorBlock's size gives the
number of labelled videos the dataset could supply.

IMPORTANT CAVEAT, measured 2026-09-20: YouTube-Commons stores each transcript
as one block of prose with NO timestamps. Our labelling needs per-line start
times, because a SponsorBlock segment is a range in seconds. So a high overlap
here is necessary but not sufficient -- see the notes printed at the end.
"""

import argparse
import json
from pathlib import Path

import duckdb

HERE = Path(__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ytc", type=Path, default=HERE / "data" / "ytc_ids.parquet")
    ap.add_argument(
        "--segments",
        type=Path,
        nargs="+",
        default=[HERE / "data" / "raw_segments.json", HERE / "data" / "raw_segments_wide.json"],
    )
    args = ap.parse_args()

    sponsor: dict[str, int] = {}
    for path in args.segments:
        if not path.exists():
            print(f"  (skipping {path.name}, not built yet)")
            continue
        for v in json.loads(path.read_text(encoding="utf-8")):
            sponsor[v["videoID"]] = len(v["segments"])
    print(f"{len(sponsor):,} SponsorBlock videos sampled (uniform, by hash prefix)")

    con = duckdb.connect()
    con.execute("CREATE TABLE sb (video_id VARCHAR)")
    con.executemany("INSERT INTO sb VALUES (?)", [(k,) for k in sponsor])

    ytc = args.ytc.as_posix()
    total = con.execute(f"SELECT count(*) FROM read_parquet('{ytc}')").fetchone()[0]
    print(f"{total:,} videos in YouTube-Commons")

    hits = con.execute(
        f"SELECT y.video_id, y.lang, y.words FROM read_parquet('{ytc}') y JOIN sb ON sb.video_id = y.video_id"
    ).fetchall()
    english = [h for h in hits if (h[1] or "").lower().startswith("en")]

    rate = len(hits) / max(len(sponsor), 1)
    print(f"\noverlap: {len(hits)} of {len(sponsor):,} sampled SponsorBlock videos ({100 * rate:.2f}%)")
    print(f"  of those, {len(english)} have an English transcript")
    print(f"\nSponsorBlock holds on the order of a million videos with sponsor segments,")
    print(f"so a {100 * rate:.2f}% rate implies very roughly {int(rate * 1_000_000):,} joinable videos.")

    print("\nBUT: YouTube-Commons transcripts carry no timestamps -- they are one block")
    print("of prose per video. Labelling needs per-line start times to map a segment")
    print("given in seconds onto text. Without them these transcripts cannot be")
    print("labelled from SponsorBlock at all, whatever the overlap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
