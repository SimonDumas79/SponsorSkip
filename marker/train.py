"""Train the marker, choose its thresholds honestly, and grade it once on the holdout.

Reads   data/features.npz        train + val rows, 395 numbers per caption line
        data/features_tail.npz   the holdout (only with --holdout): videos whose SponsorBlock
                                 segments few people ever reviewed, i.e. the smaller channels
                                 the marker has to work on, and not one channel shared with training
Writes  data/val_scores.npy      one probability per validation row (score.py --scores)
        data/holdout_scores.npy  one probability per holdout row     (only with --holdout)
        data/marker.pt           weights, scaling and the two chosen thresholds

What happens, in order
----------------------
1. Cross-validation, grouped by channel. The channels are dealt into five
   groups; five markers are trained, each on four groups, and each scores the
   fifth. Every row ends up scored by a marker that never saw its channel, so
   every read in the file becomes tuning data, not just the validation split's.
2. A threshold sweep over those out-of-fold scores picks the two thresholds
   the extension needs, one per way of running it:
     with a checker   a local model or Claude reads each flagged window before
                      anything is skipped, so a false alarm only costs one check:
                      the highest threshold whose recall still clears the gate
     alone            nothing checks the flags, so a false alarm skips real show:
                      the lowest threshold whose lost show stays under the budget
3. The final marker is trained on the training channels (the validation
   channels stay out so score.py still has unseen rows to grade) and saved.
4. With --holdout, the saved marker is graded ONCE on the holdout at the chosen
   thresholds. It is off by default on purpose: a holdout you look at while
   tuning stops being a holdout.

The marker is one linear model over all 395 numbers, or with --stack two linear
models trained separately (the 384 meaning numbers; the 11 seam, cue and
position numbers) whose logits are averaged. Trained apart, each has to find
reads on its own, so they miss different reads and the average covers both.

Every number beside recall is a cost. Region recall on its own is bought by
flagging everything, so the tables always show what the flags would cost:
checker windows opened per video, and seconds of real show flagged per video.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from score import regions, score_video, smooth

HERE = Path(__file__).parent
DATA = HERE / "data"

SLACK = 3    # caption lines of slack when matching a flag to a read, as score.py
LAST_LINE_SECONDS = 3.0   # a video's final caption line has no successor to measure against
# Stamped into marker.pt so whoever loads it can check the rows were built the same way.
# Bump it whenever features.py changes what the 395 numbers are.
FEATURES = "minilm-v1: 384 meaning, 3 seam, 6 cues, 2 position"
MEANING_COLUMNS = 384   # columns 0-383 are MiniLM meaning; the rest are seam, cues, position
# Thresholds are swept evenly in logit space: the useful ones crowd above 0.9, where an even
# 0.01 step would jump several reads at a time.
THRESHOLDS = np.round(1.0 / (1.0 + np.exp(-np.arange(0.0, 8.0, 0.05))), 4).tolist()


# ----------------------------------------------------------------------------- data


@dataclass
class Rows:
    """Part of the dataset: the numbers, the labels, and where each row came from."""

    X: np.ndarray              # (rows, 395) float32
    y: np.ndarray              # (rows,) 0/1: inside a sponsor or self-promo read
    video: np.ndarray          # (rows,) video id, so rows can be grouped into videos
    channel: np.ndarray        # (rows,) channel id, the unit the splits are made of
    start_seconds: np.ndarray  # (rows,) when the caption line appears
    split: np.ndarray          # (rows,) "train", "val" or "holdout"
    feature_names: np.ndarray = field(default_factory=lambda: np.array([]))

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "Rows":
        return Rows(self.X[mask], self.y[mask], self.video[mask], self.channel[mask], self.start_seconds[mask],
                    self.split[mask], self.feature_names)

    @property
    def reads(self) -> int:
        return sum(len(regions(self.y[self.video == v])) for v in np.unique(self.video))

    @property
    def videos(self) -> int:
        return len(np.unique(self.video))


def load(path: Path) -> Rows:
    d = np.load(path, allow_pickle=False)
    rows = Rows(d["X"], d["y"], d["video"], d["channel"], d["start_seconds"], d["split"], d["feature_names"])
    # Region scoring reads each video's rows in caption order, and the cost adds up how long each
    # line is on screen, so both the line numbers and their times must run forwards.
    for v in np.unique(rows.video):
        m = rows.video == v
        if not (np.all(np.diff(d["line"][m]) > 0) and np.all(np.diff(rows.start_seconds[m]) >= 0)):
            raise SystemExit(f"{path.name}: rows of video {v} are not in caption and time order")
    return rows


# ----------------------------------------------------------------------------- model


def scaling(X_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and standard deviation, from TRAINING rows only."""
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std[std == 0] = 1.0   # a constant column would divide by zero
    return mean, std


def fit(X_tr: np.ndarray, y_tr: np.ndarray, epochs: int, seed: int, weight_decay: float = 0.01,
        on_epoch=None) -> nn.Module:
    """Train one linear model on already-scaled rows. on_epoch(epoch, model, train_loss) runs after each epoch."""
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X_tr).float()
    yt = torch.from_numpy(y_tr).float()   # int8 on disk; the loss wants floats
    model = nn.Linear(Xt.shape[1], 1)

    # Only about one line in fourteen is inside a read. Without this the model learns "say no" and
    # is right 93% of the time; with it, a missed read line costs as much as ~13 false ones.
    pos_weight = torch.tensor([(len(yt) - yt.sum()) / yt.sum()])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(Xt))   # a fresh shuffle every epoch
        running = 0.0
        for i in range(0, len(Xt), 256):
            idx = order[i:i + 256]
            optimizer.zero_grad()
            logits = model(Xt[idx]).squeeze(1)   # (256, 1) -> (256,)
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            optimizer.step()
            running += loss.item() * len(idx)
        model.eval()
        if on_epoch:
            on_epoch(epoch, model, running / len(Xt))
    return model


@dataclass
class Branch:
    """One linear model over some of the columns, with the scaling it was trained with."""

    columns: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    model: nn.Module

    def logits(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            scaled = torch.from_numpy((X[:, self.columns] - self.mean) / self.std).float()
            return self.model(scaled).squeeze(1).numpy()


@dataclass
class Marker:
    """One branch over every column, or two trained separately; the score averages their logits."""

    branches: list[Branch]

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = np.mean([b.logits(X) for b in self.branches], axis=0)
        return 1.0 / (1.0 + np.exp(-z))


def column_sets(n_columns: int, stack: bool) -> list[np.ndarray]:
    all_columns = np.arange(n_columns)
    if not stack:
        return [all_columns]
    return [all_columns[:MEANING_COLUMNS], all_columns[MEANING_COLUMNS:]]


def train_marker(rows: Rows, epochs: int, seed: int, weight_decay: float, stack: bool, on_epoch=None) -> Marker:
    branches = []
    for columns in column_sets(rows.X.shape[1], stack):
        X = rows.X[:, columns]
        mean, std = scaling(X)   # from these training rows only
        model = fit((X - mean) / std, rows.y, epochs, seed, weight_decay, on_epoch if not branches else None)
        branches.append(Branch(columns, mean, std, model))
    return Marker(branches)


# ----------------------------------------------------------------------------- grading


@dataclass
class Grade:
    """What a threshold finds, and what it costs. Recall means nothing without the last two."""

    found: int = 0          # real reads with a flag inside (give or take SLACK lines)
    total: int = 0          # real reads
    windows: int = 0        # flagged regions: each one is a checker window...
    right: int = 0          # ...and this many of them were real
    show_seconds: float = 0.0   # seconds of real show flagged: what is SKIPPED with no checker
    videos: int = 0

    @property
    def recall(self) -> float:
        return self.found / max(self.total, 1)

    @property
    def precision(self) -> float:
        return self.right / max(self.windows, 1)

    @property
    def windows_per_video(self) -> float:
        return self.windows / max(self.videos, 1)

    @property
    def show_per_video(self) -> float:
        return self.show_seconds / max(self.videos, 1)

    def row(self, label: str) -> str:
        return (f"  {label:<11} {self.found:3d} / {self.total:<3d} {self.recall:6.1%}  "
                f"{self.precision:6.1%}  {self.windows_per_video:9.1f}  {self.show_per_video:8.1f} s")

    @staticmethod
    def header() -> str:
        return ("  threshold   reads found  recall   precis   windows/vid  lost show/vid\n"
                "  (windows/vid = checker windows opened per video; lost show/vid = seconds of real show\n"
                "   flagged per video, which is what gets skipped when nothing checks the flags)")


def grade(probs: np.ndarray, rows: Rows, threshold: float, smooth_window: int) -> Grade:
    """Flag, bridge gaps, and score every video the way score.py does."""
    g = Grade()
    for vid in np.unique(rows.video):
        r = np.flatnonzero(rows.video == vid)
        flags = smooth((probs[r] >= threshold).astype(np.int8), smooth_window)
        found, total, windows, right = score_video(rows.y[r], flags, SLACK)
        g.found += found
        g.total += total
        g.windows += windows
        g.right += right
        # Each caption line is on screen until the next one starts.
        s = rows.start_seconds[r]
        on_screen = np.diff(np.append(s, s[-1] + LAST_LINE_SECONDS))
        g.show_seconds += float(on_screen[(flags == 1) & (rows.y[r] == 0)].sum())
        g.videos += 1
    return g


def cue_baseline(rows: Rows, smooth_window: int) -> Grade:
    """The six hand-written cue patterns, no model: the number a trained marker has to beat."""
    cue_columns = [i for i, n in enumerate(rows.feature_names) if n.startswith("cue_")]
    fired = (rows.X[:, cue_columns].sum(axis=1) > 0).astype(np.float32)
    return grade(fired, rows, 0.5, smooth_window)


# ----------------------------------------------------------------------------- cross-validation


def cross_validate(rows: Rows, folds: int, epochs: int, seed: int, weight_decay: float, stack: bool) -> np.ndarray:
    """One out-of-fold probability per row, each from a marker that never saw the row's channel."""
    channels = np.array(sorted(set(rows.channel)))
    np.random.default_rng(seed).shuffle(channels)
    oof = np.full(len(rows), np.nan, dtype=np.float32)
    for k, fold in enumerate(np.array_split(channels, folds), start=1):
        test = np.isin(rows.channel, fold)
        marker = train_marker(rows.subset(~test), epochs, seed, weight_decay, stack)
        oof[test] = marker.predict(rows.X[test])
        held = rows.subset(test)
        print(f"  fold {k}: held out {len(fold):2d} channels, {held.videos:2d} videos, {held.reads:3d} reads")
    assert not np.isnan(oof).any(), "a row was never predicted: a fold produced no scores"
    return oof


def choose_thresholds(oof: np.ndarray, rows: Rows, gate: float, budget: float,
                      smooth_window: int) -> tuple[float | None, float | None, dict[float, Grade]]:
    """with_checker: highest threshold with recall >= gate. alone: lowest threshold with lost show <= budget."""
    sweep = {th: grade(oof, rows, th, smooth_window) for th in THRESHOLDS}
    clears_gate = [th for th, g in sweep.items() if g.recall >= gate]
    under_budget = [th for th, g in sweep.items() if g.show_per_video <= budget]
    with_checker = float(max(clears_gate)) if clears_gate else None
    alone = float(min(under_budget)) if under_budget else None
    return with_checker, alone, sweep


def neighbours(th: float | None) -> list[float]:
    """The chosen threshold and the grid points either side, so the reader sees what the rule traded."""
    if th is None:
        return []
    i = THRESHOLDS.index(th)
    return THRESHOLDS[max(0, i - 1):i + 2]


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--stack", action="store_true", help="two separately trained models: meaning; seam+cues+position")
    ap.add_argument("--smooth", type=int, default=3, help="bridge gaps of this many lines (1 = off); score.py --smooth")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", type=float, default=0.90, help="recall the with-checker threshold must keep")
    ap.add_argument("--budget", type=float, default=10.0,
                    help="seconds of real show per video the alone threshold may flag (it gets skipped)")
    ap.add_argument("--holdout", action="store_true",
                    help="grade the holdout at the chosen thresholds; do this once, at the end")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rows = load(DATA / "features.npz")
    n_channels = len(set(rows.channel))
    if not 2 <= args.folds <= n_channels:
        raise SystemExit(f"--folds must be between 2 and the number of channels ({n_channels})")
    marker_kind = "two separately trained linear models (stacked)" if args.stack else "one linear model"
    print(f"features.npz: {len(rows)} rows, {rows.videos} videos, {n_channels} channels, {rows.reads} reads")
    print(f"marker: {marker_kind}, {args.epochs} epochs, weight decay {args.weight_decay}, "
          f"gaps of {args.smooth} line(s) bridged")

    # 1. Cross-validation grouped by channel.
    print(f"\n{args.folds}-fold cross-validation by channel:")
    oof = cross_validate(rows, args.folds, args.epochs, args.seed, args.weight_decay, args.stack)

    # 2. Choose the thresholds on the out-of-fold scores, and show the sweep around them.
    with_checker, alone, sweep = choose_thresholds(oof, rows, args.gate, args.budget, args.smooth)
    print(f"\nout-of-fold, {rows.reads} reads on channels each marker never saw.")
    print(f"  with checker = highest threshold keeping recall >= {args.gate:.0%} "
          f"(miss at most {int(rows.reads * (1 - args.gate))} reads)")
    print(f"  alone        = lowest threshold losing <= {args.budget:.0f} s of real show per video")
    print(Grade.header())
    floor = grade(rows.y.astype(np.float32), rows, 0.5, args.smooth)
    print(floor.row("perfect") + "   <- a marker that flags exactly the reads: the floor bridging adds")
    landmarks = {min(THRESHOLDS, key=lambda t: abs(t - x)) for x in (0.5, 0.9, 0.95)}   # nearest grid points
    shown = sorted(landmarks | set(neighbours(with_checker)) | set(neighbours(alone)))
    for th in shown:
        tags = [name for t, name in ((with_checker, "with checker"), (alone, "alone")) if t == th]
        print(sweep[th].row(f"{th:.4f}") + ("  <- " + " & ".join(tags) if tags else ""))
    if with_checker is None:
        print(f"\nno threshold keeps recall >= {args.gate:.0%}; the with-checker threshold is unset")
    if alone is None:
        print(f"\nno threshold loses <= {args.budget:.0f} s of show per video; the alone threshold is unset")

    # 3. The final marker: train on the training channels, sanity-check on the validation channels.
    train = rows.subset(rows.split == "train")
    val = rows.subset(rows.split == "val")
    # Nothing cleared the gate: watch at a middle threshold so the check still shows something.
    watch = with_checker if with_checker is not None else 0.95
    print(f"\nfinal marker on {train.videos} training videos. Sanity check on {val.reads} validation reads "
          f"at {watch:.4f}\n(these reads were already in the tuning pool above, so this is a check, not a grade):")

    def report(epoch: int, model: nn.Module, train_loss: float) -> None:
        if epoch in (1, args.epochs):
            print(f"  epoch {epoch:2d}  train loss {train_loss:.4f}")

    marker = train_marker(train, args.epochs, args.seed, args.weight_decay, args.stack, on_epoch=report)
    val_probs = marker.predict(val.X)
    g = grade(val_probs, val, watch, args.smooth)
    print(f"  validation: {g.found}/{g.total} reads, {g.precision:.1%} precision, "
          f"{g.windows_per_video:.1f} windows/vid, {g.show_per_video:.1f} s lost show/vid")
    np.save(DATA / "val_scores.npy", val_probs)
    torch.save(
        {
            "branches": [{"columns": torch.from_numpy(b.columns), "state": b.model.state_dict(),
                          "mean": torch.from_numpy(b.mean), "std": torch.from_numpy(b.std)} for b in marker.branches],
            "thresholds": {"withChecker": with_checker, "alone": alone},
            "smooth": args.smooth,
            "features": FEATURES,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
        },
        DATA / "marker.pt",
    )
    print(f"saved val_scores.npy and marker.pt (thresholds: with checker {with_checker}, alone {alone})")

    # 4. The holdout, once.
    if not args.holdout:
        print("\nholdout not graded (pass --holdout when the tuning is finished)")
        return 0
    holdout = load(DATA / "features_tail.npz")
    shared = np.isin(holdout.channel, rows.channel)
    if shared.any():
        print(f"\nWARNING: {holdout.subset(shared).videos} holdout video(s) come from channels in training; "
              "grading without them (rebuild with build_dataset.py --not-in to fix it at the source)")
        holdout = holdout.subset(~shared)
    holdout_probs = marker.predict(holdout.X)
    np.save(DATA / "holdout_scores.npy", holdout_probs)
    print(f"\nHOLDOUT (few-review tail): {holdout.videos} videos, {holdout.reads} reads")
    print(Grade.header())
    print(cue_baseline(holdout, args.smooth).row("cues"))
    for th, label in ((with_checker, "with checker"), (alone, "alone")):
        if th is not None:
            print(grade(holdout_probs, holdout, th, args.smooth).row(f"{th:.4f}") + f"  <- {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
