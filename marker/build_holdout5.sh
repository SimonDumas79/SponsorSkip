#!/usr/bin/env bash
# Build holdout 5 (PREREGISTRATION.md, pre-registration 3: the cascade's grade) from whatever the
# rescheduled crawl has fetched. The set is defined by a RULE, not a choice: every caption file whose
# channel appears in none of the six existing sets. Run it only once there are 40+ such videos --
# holdout 3 is 10 videos and 15 reads, and its confidence interval is about +-25 points, which is why
# it cannot stand in for this.
#
# The cascade's thresholds must already be in data/cascade_thresholds.json (cascade_pooled.py, chosen
# on the pooled 205 videos) BEFORE this runs. That ordering is the whole point.
set -e
cd "$(dirname "$0")/.."
D=marker/data

python marker/build_dataset.py --captions $D/captions --candidates $D/candidates.json --out $D/examples_holdout5.jsonl \
  --all-to holdout5 --keep-selfpromo-only \
  --not-in $D/examples.jsonl $D/examples_tail.jsonl $D/examples_holdout2.jsonl $D/examples_holdout3.jsonl \
           $D/examples_channels.jsonl $D/examples_holdout4.jsonl
PYTHONIOENCODING=utf-8 python marker/features.py --device cpu --examples $D/examples_holdout5.jsonl --captions $D/captions --out $D/features_holdout5.npz
python -u marker/fetch_descriptions.py
python marker/add_descriptions.py features_holdout5.npz
PYTHONIOENCODING=utf-8 python marker/features.py --encoder potion --device cpu --examples $D/examples_holdout5.jsonl --captions $D/captions --out $D/features_potion_holdout5.npz
PYTHONIOENCODING=utf-8 python -u marker/fetch_watch_meta.py

python - <<'PY'
import json, sys
sys.path.insert(0, "marker")
from train import DATA, load
r = load(DATA / "features_holdout5.npz")
print(f"holdout 5: {r.videos} videos, {r.reads} reads, {len(set(r.channel))} channels")
if r.videos < 40:
    print("FEWER THAN 40 VIDEOS: too small to decide anything. Let the crawl run longer before grading.")
PY

echo
echo "Built. The fine-tuned streams come from the saved checkpoints now, not a fresh training run:"
echo "  python marker/cascade.py --edges --set holdout5 --model haiku     # the cascade"
echo "  python marker/export_candidate.py --from-checkpoints holdout5     # the candidate, same streams"
echo "Grade ONCE, with data/cascade_thresholds.json, and write the result into PREREGISTRATION.md"
echo "whether it passes or fails."
