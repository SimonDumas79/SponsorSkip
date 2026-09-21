"""A second-stage model that reads the marker's scores in CONTEXT, to find where reads start and end.

The marker scores each caption line on its own. A sponsor read is a run of 20 to 60
lines, so a line whose neighbours all score high is probably inside one, and a
line where the scores fall away is probably an edge, even if its own score is
middling. The line model cannot see that. This one can: for every line it takes
the marker's out-of-fold logits for the 15 lines either side (31 numbers), plus
running summaries of them, and learns which lines are really inside a read.

It is stacking, level 2 over level 1: trained only on OUT-OF-FOLD marker scores
(each from a marker that never saw the row's channel), with the same five
channel-grouped folds, so the second stage never grades a channel it trained on.

It is judged the way the extension is used: for each budget of real show lost
per video (with no video losing more than 60 s), the share of all AD TIME that
gets skipped, and the share of reads touched. Ad time is the headline, because
touching one line of a read counts it as found but leaves the rest playing.

    python marker/context_stack.py              # cross-validated comparison
    python marker/context_stack.py --holdout    # plus the one-time holdout grade
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from replay import grade_regions, row, HEADER, regions_by_video
from train import DATA, load

REACH = 15   # lines either side the second stage may look at
THRESHOLDS = np.round(1.0 / (1.0 + np.exp(-np.arange(-4.0, 8.0, 0.1))), 4)
SEAM, CUES, FRACTION = 384, slice(387, 393), 394


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def context_features(rows, scores: np.ndarray) -> np.ndarray:
    """Per line: the marker's logits at offsets -REACH..+REACH, and running summaries of them."""
    out = np.zeros((len(rows), 2 * REACH + 1 + 8), dtype=np.float32)
    for vid in np.unique(rows.video):
        r = np.flatnonzero(rows.video == vid)
        z = logit(scores[r])
        padded = np.pad(z, REACH, constant_values=-6.0)   # beyond the video: "certainly not an ad"
        window = np.lib.stride_tricks.sliding_window_view(padded, 2 * REACH + 1)   # (lines, 31)
        before, after = window[:, :REACH], window[:, REACH + 1:]
        X = rows.X[r]
        out[r] = np.column_stack([
            window,
            before.mean(1), after.mean(1), before.max(1), after.max(1),
            X[:, SEAM], (X[:, CUES].sum(1) > 0).astype(np.float32), X[:, FRACTION],
            np.minimum(before.mean(1), after.mean(1)),   # high only when BOTH sides look like an ad
        ])
    return out


def fit_stage2(F: np.ndarray, y: np.ndarray, hidden: int, seed: int = 0, epochs: int = 15) -> tuple:
    mean, std = F.mean(0), F.std(0) + 1e-6
    torch.manual_seed(seed)
    Xt, yt = torch.from_numpy((F - mean) / std).float(), torch.from_numpy(y).float()
    model = nn.Linear(F.shape[1], 1) if hidden == 0 else nn.Sequential(
        nn.Linear(F.shape[1], hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(yt) - yt.sum()) / yt.sum()]))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            idx = order[i:i + 256]
            opt.zero_grad()
            loss_fn(model(Xt[idx]).squeeze(1), yt[idx]).backward()
            opt.step()
    model.eval()

    def predict(G: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(model(torch.from_numpy((G - mean) / std).float()).squeeze(1)).numpy()
    return predict


def out_of_fold(rows, F: np.ndarray, hidden: int, folds: int = 5, seed: int = 0) -> np.ndarray:
    channels = np.array(sorted(set(rows.channel)))
    np.random.default_rng(seed).shuffle(channels)   # the same folds the marker's own CV used
    p = np.zeros(len(rows), dtype=np.float32)
    for fold in np.array_split(channels, folds):
        test = np.isin(rows.channel, fold)
        p[test] = fit_stage2(F[~test], rows.y[~test].astype(np.float32), hidden)(F[test])
    return p


def best_under(probs: np.ndarray, rows, budget: float, min_lines: int = 1):
    """The threshold that skips the most ad time with <= budget s lost per video and no video over 60 s."""
    best = None
    for th in THRESHOLDS:
        kept = regions_by_video(probs, rows, th, 1)
        if min_lines > 1:
            kept = {v: [(a, b) for a, b in rs if b - a >= min_lines] for v, rs in kept.items()}
        g = grade_regions(kept, rows)
        if g["show"] <= budget and g["worst"] <= 60 and (
                best is None or (g["coverage"], g["recall"]) > (best[1]["coverage"], best[1]["recall"])):
            best = (float(th), g)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level1", type=Path, default=None,
                    help="a zoo folder whose oof_seed0.npy is the level-1 score (default: the baseline marker)")
    ap.add_argument("--holdout", action="store_true", help="apply the CV-chosen setup to the holdout, once")
    ap.add_argument("--apply", nargs=3, metavar=("FEATURES", "LEVEL1_SCORES", "OUT"),
                    help="train the second stage (32 hidden) on all CV rows and write its probabilities for "
                         "another set, from that set's marker scores; grades nothing")
    args = ap.parse_args()
    torch.set_num_threads(4)

    rows = load(DATA / "features.npz")
    if args.apply:
        from experiments import BASE, cross_validate
        features, scores, out = (Path(a) for a in args.apply)
        F = context_features(rows, cross_validate(BASE, rows, seed=0))
        other = load(features)
        probs = fit_stage2(F, rows.y.astype(np.float32), hidden=32)(context_features(other, np.load(scores)))
        np.save(out, probs)
        print(f"wrote {len(probs)} second-stage probabilities to {out}")
        return 0
    if args.level1:
        level1 = np.load(args.level1 / "oof_seed0.npy")
        name1 = args.level1.name
    else:
        from experiments import BASE, cross_validate
        level1 = cross_validate(BASE, rows, seed=0)
        name1 = "baseline marker"
    F = context_features(rows, level1)
    print(f"level 1: {name1}; level 2 sees {F.shape[1]} numbers per line ({2 * REACH + 1} neighbouring logits)")

    print("\ncross-validated, best threshold per budget (objective: ad time skipped):")
    print(HEADER)
    results = {"level 1 alone": level1}
    for hidden in (0, 32):
        results[f"level 2, {'linear' if hidden == 0 else f'{hidden} hidden'}"] = out_of_fold(rows, F, hidden)
    chosen = {}
    for name, probs in results.items():
        for budget in (5, 10, 15):
            best = best_under(probs, rows, budget)
            if best:
                chosen[(name, budget)] = best[0]
                print(row(f"{name} <= {budget}s (th {best[0]})", best[1]))
    np.save(DATA / "context_stack_oof.npy", results["level 2, 32 hidden"])
    (DATA / "context_stack_thresholds.json").write_text(json.dumps(
        {f"{k[0]} @ {k[1]}s": v for k, v in chosen.items()}, indent=1), encoding="utf-8")

    if not args.holdout:
        print("\nholdout not graded (pass --holdout once the choice is final)")
        return 0
    manifest = json.loads((DATA / "confirm_verdicts_holdout.manifest.json").read_text(encoding="utf-8"))
    h_rows = load(Path(manifest["features"]))
    h_level1 = np.load(manifest["scores"])   # the baseline marker trained on every training row
    HF = context_features(h_rows, h_level1)
    print(f"\nHOLDOUT: {h_rows.videos} videos, {h_rows.reads} reads; thresholds chosen on CV above")
    print(HEADER)
    stage2 = {hidden: fit_stage2(F, rows.y.astype(np.float32), hidden) for hidden in (0, 32)}
    for (name, budget), th in chosen.items():
        if name == "level 1 alone":
            probs = h_level1
        else:
            probs = stage2[0 if "linear" in name else 32](HF)
        print(row(f"{name}, CV-chosen for <= {budget}s", grade_regions(regions_by_video(probs, h_rows, th, 1), h_rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
