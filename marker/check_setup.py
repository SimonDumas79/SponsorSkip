"""Run this first. It proves your environment works and shows you your data.

It trains nothing. It answers the four questions you would otherwise waste the
first half hour on: is PyTorch seeing the GPU, does the data load, what shape
is it, and what exactly does train.py have to produce.

    python marker/check_setup.py
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
FEATURES = HERE / "data" / "features.npz"


def main() -> int:
    print("=" * 66)
    print("1. PyTorch")
    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"   torch {torch.__version__}   CUDA available: {cuda}")
        if cuda:
            print(f"   device: {torch.cuda.get_device_name(0)}")
        print("   (train on the CPU first -- this data is small and CPU errors read better)")
    except ImportError:
        print("   PyTorch NOT installed")
        return 1

    print("\n2. Your data")
    if not FEATURES.exists():
        print(f"   {FEATURES} missing -- run build_dataset.py then features.py")
        return 1
    d = np.load(FEATURES, allow_pickle=False)
    X, y, split = d["X"], d["y"], d["split"]
    names = list(d["feature_names"])
    train, val = split == "train", split == "val"

    print(f"   X            {X.shape}   float32, one row per caption line")
    print(f"   y            {y.shape}   1 = this line is inside a sponsor read")
    print(f"   train rows   {train.sum():,} across {len(set(d['channel'][train]))} channels")
    print(f"   val rows     {val.sum():,} across {len(set(d['channel'][val]))} channels (never trained on)")
    print(f"   positives    {y.sum():,} ({100 * y.mean():.1f}%)  <- the imbalance that makes accuracy useless")
    print(f"   read starts  {int(d['is_start'].sum())}   (a separate label, unused so far -- see below)")

    print("\n3. What the 395 columns are")
    groups = [("meaning (MiniLM, 9 lines averaged)", 384), ("relatedness / seam", 3),
              ("ad cue patterns", 6), ("position in video", 2)]
    at = 0
    for label, width in groups:
        print(f"   [{at:3d}:{at + width:3d}]  {label}")
        at += width
    print(f"   last two column names: {names[-2]}, {names[-1]}")

    print("\n4. The scale problem you must fix in step 2")
    minutes = names.index("minutes_in")
    print(f"   column 'minutes_in'  range {X[:, minutes].min():.1f} to {X[:, minutes].max():.1f}")
    print(f"   column 'meaning_0'   range {X[:, 0].min():.2f} to {X[:, 0].max():.2f}")
    print("   -> standardise every column on the TRAINING rows only, or the big one drowns the rest")

    pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
    print("\n5. Numbers you will want")
    print(f"   pos_weight for BCEWithLogitsLoss: {pos_weight:.1f}   (negatives / positives)")
    print("   a starting point, not a rule -- raising it buys recall and costs precision")

    print("\n6. What train.py must produce")
    print("   a .npy of one probability per VALIDATION row, in this same order:")
    print(f"     np.save('marker/data/val_scores.npy', probs)   # length {val.sum():,}")
    print("   then:  python marker/score.py --scores marker/data/val_scores.npy")
    print("   score.py reports REGION RECALL -- of the real reads, how many you flagged anywhere inside.")

    print("\n7. The bar")
    print("   cue patterns alone, low-review holdout: 62.9% region recall")
    print("   cue patterns + qwen3 union:             81.4%")
    print("   the gate:                               90%")
    print("   13 of 70 reads are found by neither -- those are the model's actual job")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
