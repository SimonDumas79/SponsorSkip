"""A second model with its own job: judge each flagged REGION, not each caption line.

The marker scores caption lines one at a time. Whether a flagged region is a
real read is a different question, with evidence the line model never sees
together: how long the region runs, how confident the marker is across all of
it, whether ad phrasing turns up anywhere inside, how sharp the change of
subject is where it begins, where in the video it sits, and (when the local
model was asked) what qwen said about it. This script trains a small logistic
model on exactly those facts.

It is stacking done honestly. Every input is something already recorded: the
marker's out-of-fold scores, the features, and qwen's verdicts from
confirm_check.py. The judge is trained with five channel-grouped folds over
the regions, so each region's verdict comes from a judge that never saw its
channel. Then a skip policy is chosen from those out-of-fold verdicts (keep a
region when the judge is sure enough; skip only its high-confidence core), and
with --holdout that policy is applied, once, to the holdout's regions.

    python marker/region_judge.py              # cross-validated comparison
    python marker/region_judge.py --holdout    # plus the one-time holdout grade
"""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from replay import grade_regions, load_verdicts, regions_by_video, row, HEADER
from train import DATA, load

CV_RUN = DATA / "confirm_verdicts.jsonl"
HOLDOUT_RUN = DATA / "confirm_verdicts_holdout.jsonl"
SEAM, CUES, FRACTION = 384, slice(387, 393), 394   # feature columns (see features.py)


def load_run(verdicts_path: Path):
    manifest_path = verdicts_path.with_suffix(".manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = load(Path(manifest["features"]))
        scores = np.load(manifest["scores"])
        threshold, smooth_w = manifest["threshold"], manifest["config"]["smooth"]
    else:   # the first CV run predates manifests: the baseline marker, out-of-fold, at 0.84
        from experiments import BASE, cross_validate
        rows = load(DATA / "features.npz")
        scores = cross_validate(BASE, rows, seed=0)
        threshold, smooth_w = 0.84, 3
    return rows, scores, threshold, smooth_w, load_verdicts(verdicts_path)


def region_table(rows, scores, threshold, smooth_w, verdicts, strict: bool = True):
    """One row of facts per flagged region, plus whether it really held a read."""
    facts, info = [], []
    for vid, spans in regions_by_video(scores, rows, threshold, smooth_w).items():
        r = np.flatnonzero(rows.video == vid)
        X, s, t = rows.X[r], scores[r], rows.start_seconds[r]
        for lo, hi in spans:
            v = verdicts.get(f"{vid}:{lo}-{hi}")
            if v is None:
                if strict:
                    raise SystemExit(f"no recorded verdict for region {vid}:{lo}-{hi}; the replay would not be honest")
                v = {"said": None, "p_yes": None}   # qwen never asked: only the free judge may use this row
            seconds = float(t[min(hi, len(r) - 1)] - t[lo]) + 1.0
            facts.append([
                s[lo:hi].max(), s[lo:hi].mean(), (s[lo:hi] >= 0.95).mean(), (s[lo:hi] >= 0.99).mean(),
                np.log1p(hi - lo), np.log1p(seconds),
                (X[lo:hi, CUES].sum(axis=1) > 0).mean(), X[lo:hi, CUES].max(),
                X[max(0, lo - 3):lo + 3, SEAM].max(), X[lo, FRACTION],
                float(v["said"] is True), v["p_yes"] if v.get("p_yes") is not None else -1.0,
            ])
            info.append({"video": vid, "channel": str(rows.channel[r][0]), "lo": lo, "hi": hi,
                         "is_read": bool(rows.y[r][max(0, lo - 3):hi + 3].any())})   # from the labels, as score.py
    return np.array(facts, dtype=np.float64), info


NAMES = ["peak", "mean", "share>=0.95", "share>=0.99", "log lines", "log seconds", "cue share", "any cue",
         "seam at start", "position", "qwen yes", "qwen P(yes)"]
WITHOUT_QWEN = list(range(10))
WITH_QWEN = list(range(11))   # P(yes) is left out: the holdout run did not record it


def fit_judge(F: np.ndarray, y: np.ndarray, columns: list[int]):
    mean, std = F[:, columns].mean(axis=0), F[:, columns].std(axis=0) + 1e-9
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    model.fit((F[:, columns] - mean) / std, y)
    return lambda G: model.predict_proba((G[:, columns] - mean) / std)[:, 1]


def out_of_fold(F, y, info, columns, folds=5, seed=0):
    channels = np.array(sorted({i["channel"] for i in info}))
    np.random.default_rng(seed).shuffle(channels)
    ch = np.array([i["channel"] for i in info])
    p = np.zeros(len(y))
    for fold in np.array_split(channels, folds):
        test = np.isin(ch, fold)
        p[test] = fit_judge(F[~test], y[~test], columns)(F[test])
    return p


def kept_regions(info, p, t):
    kept = {}
    for i, pi in zip(info, p):
        if pi >= t:
            kept.setdefault(i["video"], []).append((i["lo"], i["hi"]))
    return kept


CORES = [0.0, 0.95, 0.99]           # 0 = skip the whole flagged region
GROWS = [None, 0.95, 0.9, 0.84, 0.7]  # grow each core outwards while the score stays >= this


def better(g: dict, best: dict | None) -> bool:
    """The objective: most AD TIME skipped; then most reads; then least show lost."""
    return best is None or (g["coverage"], g["recall"], -g["show"]) > (best["coverage"], best["recall"], -best["show"])


def best_policy(info, p, rows, scores, budget):
    """Most ad time skipped with <= budget s of show lost per video and no video over 60 s."""
    best = None
    for t, c, low in product(np.round(np.arange(0.05, 1.0, 0.05), 2), CORES, GROWS):
        if c == 0.0 and low is not None:
            continue
        g = grade_regions(kept_regions(info, p, t), rows, scores, c, 1, low)
        if g["show"] <= budget and g["worst"] <= 60 and better(g, best and best[3]):
            best = (float(t), c, low, g)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", action="store_true", help="apply the CV-chosen policies to the holdout, once")
    args = ap.parse_args()
    torch.set_num_threads(4)

    rows, scores, threshold, smooth_w, verdicts = load_run(CV_RUN)
    F, info = region_table(rows, scores, threshold, smooth_w, verdicts)
    y = np.array([i["is_read"] for i in info], dtype=int)
    print(f"{len(info)} flagged regions at threshold {threshold} ({y.sum()} hold a read) on {rows.videos} videos")

    judges = {"judge without qwen": WITHOUT_QWEN, "judge with qwen": WITH_QWEN}
    chosen = {}
    print("\ncross-validated (region judges trained by channel folds), best policy per budget:")
    print(HEADER)
    for name, columns in judges.items():
        p = out_of_fold(F, y, info, columns)
        for budget in (5, 10):
            best = best_policy(info, p, rows, scores, budget)
            if best:
                t, c, low, g = best
                chosen[(name, budget)] = (t, c, low)
                print(row(f"{name} <={budget}s: p>={t} core {c} grow {low}", g))
    for budget in (5, 10):   # the qwen-only rule from replay.py, for comparison on the same footing
        best = None
        for b, c, low in product([0.995, 0.998, 1.01], CORES, GROWS):
            if c == 0.0 and low is not None:
                continue
            policy_kept = {}
            for i, f in zip(info, F):
                if f[10] == 1.0 or f[0] >= b:
                    policy_kept.setdefault(i["video"], []).append((i["lo"], i["hi"]))
            g = grade_regions(policy_kept, rows, scores, c, 1, low)
            if g["show"] <= budget and g["worst"] <= 60 and better(g, best and best[3]):
                best = (b, c, low, g)
        if best:
            chosen[("qwen rule", budget)] = best[:3]
            print(row(f"qwen rule <={budget}s: yes|peak>={best[0]} core {best[1]} grow {best[2]}", best[3]))

    full = fit_judge(F, y, WITH_QWEN)
    coef = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(
        (F[:, WITH_QWEN] - F[:, WITH_QWEN].mean(0)) / (F[:, WITH_QWEN].std(0) + 1e-9), y).coef_[0]
    print("\nwhat the judge with qwen leans on (standardised weights):")
    for n, w in sorted(zip([NAMES[i] for i in WITH_QWEN], coef), key=lambda nw: -abs(nw[1])):
        print(f"  {n:<14} {w:+.2f}")
    (DATA / "region_judge_policies.json").write_text(json.dumps(
        {f"{k[0]} @ {k[1]}s": v for k, v in chosen.items()}, indent=1), encoding="utf-8")

    if not args.holdout:
        print("\nholdout not graded (pass --holdout once the choice is final)")
        return 0
    h_rows, h_scores, h_th, h_smooth, h_verdicts = load_run(HOLDOUT_RUN)
    HF, h_info = region_table(h_rows, h_scores, h_th, h_smooth, h_verdicts)
    print(f"\nHOLDOUT: {len(h_info)} flagged regions on {h_rows.videos} videos; policies chosen on CV above")
    print(HEADER)
    print(row("marker alone at the checker threshold", grade_regions(kept_regions(h_info, np.ones(len(h_info)), 0.5), h_rows)))
    for (name, budget), (t, c, low) in chosen.items():
        if name == "qwen rule":
            kept = {}
            for i, f in zip(h_info, HF):
                if f[10] == 1.0 or f[0] >= t:
                    kept.setdefault(i["video"], []).append((i["lo"], i["hi"]))
        else:
            judge = fit_judge(F, y, judges[name])
            kept = kept_regions(h_info, judge(HF), t)
        print(row(f"{name}, CV-chosen for <= {budget} s", grade_regions(kept, h_rows, h_scores, c, 1, low)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
