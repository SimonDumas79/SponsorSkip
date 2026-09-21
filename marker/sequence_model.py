"""One model that reads a whole video at once: the sequence marker from Part 7.5 of the build guide.

The free tier is two models in a row: the marker scores each line alone, then the
context model reads those scores 15 lines either side. This is the end-to-end
version: every line's 395 numbers go through a small layer, then two 1-D
convolutions let each line mix with its neighbours (reach lines either side,
twice), and one output per line says "inside a read". It trains one video per
step, so it can learn the SHAPE of a read (a seam, then ad language, then a seam
back) directly from the features instead of from a first model's scores.

Judged exactly like the free tier on the pooled 205 videos: the same
channel-grouped folds (seed 0), the same start/resume heads placing the edges,
the same rule (most ad time skipped with at most B s of show lost per video and
at most 2% of videos over 60 s). Cross-validated only: both holdouts are spent,
so nothing from here ships without a fresh test.

    python marker/sequence_model.py
"""

import argparse

import numpy as np
import torch
from torch import nn

from build_production import OVER_CAP_SHARE, over_cap, pooled
from edge_heads import place
from replay import HEADER, grade_regions, regions_by_video, row
from train import DATA, scaling


class SequenceMarker(nn.Module):
    def __init__(self, features: int = 395, hidden: int = 64, reach: int = 7, dropout: float = 0.2):
        super().__init__()
        self.inp = nn.Linear(features, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=2 * reach + 1, padding=reach)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=2 * reach + 1, padding=reach)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (lines, features), one video
        h = self.drop(torch.relu(self.inp(x)))                       # (lines, hidden)
        h = h.T.unsqueeze(0)                                         # (1, hidden, lines)
        h = torch.relu(self.conv1(h))
        h = torch.relu(self.conv2(h)) + h                            # a residual keeps the first reach
        return self.out(self.drop(h.squeeze(0).T)).squeeze(1)        # (lines,)


def videos_of(rows, mask):
    """(row indices) per video inside the mask, in caption order."""
    return [np.flatnonzero(mask & (rows.video == v)) for v in np.unique(rows.video[mask])]


def train_and_score(rows, train_mask, test_mask, hidden, reach, epochs, seed=0):
    torch.manual_seed(seed)
    mean, std = scaling(rows.X[train_mask])
    X = torch.from_numpy((rows.X - mean) / std).float()
    y = torch.from_numpy(rows.y.astype(np.float32))
    model = SequenceMarker(rows.X.shape[1], hidden, reach)
    ratio = float((len(y[train_mask]) - y[train_mask].sum()) / y[train_mask].sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ratio]))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    train_videos = videos_of(rows, train_mask)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        for k in rng.permutation(len(train_videos)):
            idx = torch.from_numpy(train_videos[k])
            opt.zero_grad()
            loss_fn(model(X[idx]), y[idx]).backward()
            opt.step()
    model.eval()
    out = np.zeros(len(rows), dtype=np.float32)
    with torch.no_grad():
        for r in videos_of(rows, test_mask):
            out[r] = torch.sigmoid(model(X[torch.from_numpy(r)])).numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(6)

    rows, _, _ = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")   # build_production.py's out-of-fold scores
    channels = np.array(sorted(set(rows.channel)))
    np.random.default_rng(0).shuffle(channels)   # the same folds as cross_validate(seed=0)
    folds = np.array_split(channels, 5)

    candidates = {"free tier (marker + context model)": level2}
    for hidden, reach in ((32, 7), (64, 7), (64, 15)):
        oof = np.zeros(len(rows), dtype=np.float32)
        for fold in folds:
            test = np.isin(rows.channel, fold)
            oof += train_and_score(rows, ~test, test, hidden, reach, args.epochs) * test
        candidates[f"sequence model h{hidden} reach {reach}"] = oof
        np.save(DATA / f"sequence_oof_h{hidden}_r{reach}.npy", oof)
        print(f"trained sequence model h{hidden} reach {reach}", flush=True)

    print(f"\npooled cross-validation, {rows.videos} videos, {rows.reads} reads; edges from the start/resume heads;")
    print(f"rule: most ad time with <= B s lost per video and <= {OVER_CAP_SHARE:.0%} of videos over 60 s")
    print(HEADER)
    for name, probs in candidates.items():
        for budget in (5, 10):
            best = None
            for th in np.round(1 / (1 + np.exp(-np.arange(0.0, 7.0, 0.1))), 4):
                kept = place(regions_by_video(probs, rows, th, 1), rows, p_start, p_end)
                g = grade_regions(kept, rows)
                if g["show"] <= budget and over_cap(kept, rows) <= OVER_CAP_SHARE and (
                        best is None or g["coverage"] > best[1]["coverage"]):
                    best = (float(th), g)
            if best:
                print(row(f"B={budget}: {name[:30]} th {best[0]}", best[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
