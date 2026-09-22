#!/usr/bin/env bash
# Build holdout 4 (PREREGISTRATION.md, pre-registration 2) from whatever the tail crawl has fetched so far.
# CPU and network steps only; the GPU steps (BGE --score x3, sponsorblock_ml.py) run after, one at a time.
set -e
cd "$(dirname "$0")/.."
D=marker/data
python marker/build_dataset.py --captions $D/captions --candidates $D/candidates.json --out $D/examples_holdout4.jsonl \
  --all-to holdout4 --keep-selfpromo-only \
  --not-in $D/examples.jsonl $D/examples_tail.jsonl $D/examples_holdout2.jsonl $D/examples_holdout3.jsonl $D/examples_channels.jsonl
PYTHONIOENCODING=utf-8 python marker/features.py --device cpu --examples $D/examples_holdout4.jsonl --captions $D/captions --out $D/features_holdout4.npz
python -u marker/fetch_descriptions.py
python marker/add_descriptions.py features_holdout4.npz
PYTHONIOENCODING=utf-8 python marker/features.py --encoder potion --device cpu --examples $D/examples_holdout4.jsonl --captions $D/captions --out $D/features_potion_holdout4.npz
PYTHONIOENCODING=utf-8 python -u marker/fetch_watch_meta.py
python -u marker/sb_dates.py
echo "holdout 4 built: now the GPU steps (finetune_minilm.py --score for bge, edge_start, edge_resume; sponsorblock_ml.py)"
