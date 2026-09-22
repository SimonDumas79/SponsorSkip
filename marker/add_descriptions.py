"""Add the two description columns to feature files already built, without re-embedding anything.

features.py computes them for new builds; this widens the existing .npz files (395 -> 397
columns) from the same examples rows and the same function, so the two paths cannot differ.
Idempotent: a file that already has the columns is recomputed in place, not doubled.

    python marker/add_descriptions.py
"""

import json
import sys
from pathlib import Path

import numpy as np

from features import DESCRIPTION_COLUMNS, description_row, description_tokens

HERE = Path(__file__).parent
DATA = HERE / "data"
PAIRS = {"features.npz": "examples.jsonl", "features_tail.npz": "examples_tail.jsonl",
         "features_holdout2.npz": "examples_holdout2.jsonl", "features_holdout3.npz": "examples_holdout3.jsonl",
         "features_channels.npz": "examples_channels.jsonl", "features_holdout4.npz": "examples_holdout4.jsonl"}


def main() -> int:
    only = set(sys.argv[1:])   # optional: feature file names to limit the run to
    descriptions = json.loads((DATA / "descriptions.json").read_text(encoding="utf-8"))
    for feats, examples in PAIRS.items():
        if not (DATA / feats).exists() or (only and feats not in only):
            continue
        d = dict(np.load(DATA / feats, allow_pickle=False))
        text = {}
        with (DATA / examples).open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                text[(r["videoID"], r["i"])] = (r["text"], r["context"])
        names = [n for n in d["feature_names"].tolist() if n not in DESCRIPTION_COLUMNS]
        X = d["X"][:, :len(names)]
        tokens_of = {v: description_tokens(descriptions.get(str(v))) for v in np.unique(d["video"])}
        desc = np.array([description_row(*text[(str(v), int(i))], tokens_of[v])
                         for v, i in zip(d["video"], d["line"])], dtype=np.float32)
        d["X"] = np.hstack([X, desc]).astype(np.float32)
        d["feature_names"] = np.array(names + DESCRIPTION_COLUMNS)
        np.savez_compressed(DATA / feats, **d)
        with_desc = sum(1 for v in tokens_of if tokens_of[v])
        print(f"{feats}: X {d['X'].shape}; {with_desc}/{len(tokens_of)} videos have description tokens; "
              f"{int(desc[:, 0].sum())} lines hit, {int((desc[:, 0] * d['y']).sum())} of them inside a read")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
