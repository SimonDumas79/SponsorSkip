"""Pull every video id out of YouTube-Commons, so it can be joined to SponsorBlock.

Why this exists: fetching captions from YouTube is rate-limited hard -- about 35
videos an hour before it answers 429 to everything. YouTube-Commons is a public
CC-BY dataset of 22.7M transcripts across ~3.1M YouTube videos, and it carries
the video id for every row. SponsorBlock's segments are keyed by video id too.
So any video in BOTH gives a labelled training example with no fetching at all.

The trick that makes this cheap: parquet stores columns separately, so asking
for only `video_id` downloads a tiny slice of a 162 GB dataset. One shard reads
in about a second.

Output: data/ytc_ids.parquet -- one row per distinct video id, plus the channel
so the channel-stratified split still works on whatever we join.
"""

import argparse
import time
from pathlib import Path

import duckdb

HERE = Path(__file__).parent
BASE = "https://huggingface.co/datasets/PleIAs/YouTube-Commons/resolve/main"
SHARDS = 439


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "data" / "ytc_ids.parquet")
    ap.add_argument("--batch", type=int, default=20, help="shards per query")
    ap.add_argument("--shards", type=int, default=SHARDS)
    args = ap.parse_args()

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("CREATE TABLE ids (video_id VARCHAR, channel_id VARCHAR, lang VARCHAR, words BIGINT)")

    started = time.time()
    for lo in range(0, args.shards, args.batch):
        urls = [f"{BASE}/cctube_{i}.parquet" for i in range(lo, min(lo + args.batch, args.shards))]
        try:
            con.execute(
                "INSERT INTO ids SELECT DISTINCT video_id, channel_id, transcription_language, word_count "
                f"FROM read_parquet({urls!r})"
            )
        except Exception as e:  # one bad shard must not lose the scan
            print(f"  shards {lo}-{lo + len(urls) - 1} failed: {str(e)[:110]}")
            continue
        n = con.execute("SELECT count(*) FROM ids").fetchone()[0]
        print(f"  shards {lo:3d}-{lo + len(urls) - 1:3d}   {n:,} ids so far   {(time.time() - started) / 60:.1f} min")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT DISTINCT video_id, any_value(channel_id) AS channel_id, "
                f"any_value(lang) AS lang, max(words) AS words FROM ids GROUP BY video_id) "
                f"TO '{args.out.as_posix()}' (FORMAT parquet)")
    total, channels = con.execute(
        f"SELECT count(*), count(DISTINCT channel_id) FROM read_parquet('{args.out.as_posix()}')"
    ).fetchone()
    print(f"\n{total:,} distinct videos across {channels:,} channels -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
