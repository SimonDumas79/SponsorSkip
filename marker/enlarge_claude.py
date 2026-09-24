"""The fine-tuned Claude-category detector as an 8th stack input, retested on the enlarged pool. (2026-09-24)

claude_labels_categories.json only covers the original 205 pooled videos, and
claude_labels_categories_holdout5.json covers holdout 5 (46 more) -- 251 of the enlarged pool's 494
videos. Running claude_label.py --categories on the other 243 (holdout 3, the channel set, holdout 4)
would mean fresh Claude API calls on videos nobody has categorised, which is out of scope here (not
requested, and not a small job). Instead:

  1. A true out-of-fold stream on the 251 labelled videos: finetune_minilm.py --target claude-nonsponsor
     --model BAAI/bge-small-en-v1.5 --extra-sets features_holdout5.npz, channel-grouped 5-fold CV
     exactly as category_detector.py's linear version, just fine-tuned and on 251 instead of 205.
  2. For the 243 unlabelled videos: the SAME detector trained on all 251 labelled videos (never on
     their own channels, which are disjoint from the 251's -- verified: 0 shared channels), scored
     with finetune_minilm.py --score once per remaining set. This is honestly out-of-fold for those
     videos (their channels were never in the training pool) but is a different regime from true
     k-fold CV, so it is reported as what it is, not relabelled as identical to the 251's OOF.

This script only ASSEMBLES the 494-row stream from those pieces (by video, not by row-block order,
since the 251-pool's internal set order differs from ENLARGED_SETS) and saves it; it runs no model.

    python marker/enlarge_claude.py
"""

import numpy as np

from build_production import ENLARGED_SETS, SETS as POOLED_SETS, pooled
from train import DATA, load

TAG = "claude_nonsponsor_251"
REMAINING = ["features_holdout3.npz", "features_channels.npz", "features_holdout4.npz"]
OUT = DATA / "claude_nonsponsor_stream_enlarged.npy"


def place_by_video(dst_rows, src_rows, src_scores: np.ndarray) -> np.ndarray:
    """Copy src_scores into a dst-shaped array, matching rows by video id (not by row-block position,
    since two pools can list the same videos in different set orders)."""
    out = np.full(len(dst_rows), np.nan, dtype=np.float32)
    src_index = {str(v): np.flatnonzero(src_rows.video == v) for v in np.unique(src_rows.video)}
    for v in np.unique(dst_rows.video):
        d = np.flatnonzero(dst_rows.video == v)
        s = src_index.get(str(v))
        if s is None:
            continue
        if len(s) != len(d):
            raise SystemExit(f"video {v}: row-count mismatch placing a stream ({len(s)} vs {len(d)})")
        out[d] = src_scores[s]
    return out


def main() -> int:
    rows = pooled(ENLARGED_SETS)[0]
    labelled_pool = pooled(POOLED_SETS + ["features_holdout5.npz"])[0]

    oof_path = DATA / f"finetune_oof_{TAG}_seed0.npy"
    if not oof_path.exists():
        raise SystemExit(f"missing {oof_path}: run finetune_minilm.py --target claude-nonsponsor "
                         f"--model BAAI/bge-small-en-v1.5 --extra-sets features_holdout5.npz --tag {TAG} first")
    stream = place_by_video(rows, labelled_pool, np.load(oof_path))
    covered = int((~np.isnan(stream)).sum())
    print(f"251-video labelled pool (true out-of-fold): {covered:,} of {len(rows):,} rows placed")

    for feats in REMAINING:
        name = feats.replace("features_", "").replace(".npz", "")
        full_path = DATA / f"finetune_full_{TAG}_seed0__{name}.npy"
        if not full_path.exists():
            print(f"  missing {full_path}: run finetune_minilm.py --target claude-nonsponsor "
                 f"--model BAAI/bge-small-en-v1.5 --extra-sets features_holdout5.npz --tag {TAG} "
                 f"--score data/examples_{name}.jsonl data/{feats}  -- SKIPPING, those rows stay unfilled")
            continue
        fresh_rows = load(DATA / feats)
        placed = place_by_video(rows, fresh_rows, np.load(full_path))
        fill = ~np.isnan(placed)
        stream[fill] = placed[fill]
        print(f"  {feats}: {int(fill.sum()):,} rows placed (trained on the 251, scored fresh)")

    missing = int(np.isnan(stream).sum())
    if missing:
        print(f"\n{missing:,} rows ({missing / len(rows):.1%}) still unfilled -- not every set's score was ready")
    stream = np.nan_to_num(stream, nan=0.0)
    np.save(OUT, stream)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
