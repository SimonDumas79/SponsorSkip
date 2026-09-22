"""Grade the community SponsorBlock model (sponsorblock_ml.py's predictions) by our rule. (2026-09-22)

Its finds are (start, end, category, classifier probabilities) in seconds; a caption line counts as
skipped when its start time falls inside a find whose classifier category is sponsor or selfpromo.
Reported two ways: at the project's default (the classifier's 0.5 cut), and with its one knob (the
probability cut) chosen by the same pooled rule ours face (most ad time with at most B s lost per
video, at most 2% of videos over 60 s), then applied unchanged to the fresh sets.

Leakage, on the record: it was trained on SponsorBlock labels up to early 2022, the source of our
labels too, so on the pooled videos it may have seen the answers; the channel set and holdout 4 are
the clean comparisons (recent uploads).

    python marker/sbml_eval.py
"""

import json
import sys

import numpy as np

from build_production import OVER_CAP_SHARE, over_cap, pooled
from replay import grade_regions
from experiments import runs
from train import DATA, load

sys.stdout.reconfigure(encoding="utf-8")
CUTS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
READS = ("sponsor", "selfpromo")


def load_preds() -> dict:
    out = {}
    with open(DATA / "sbml_predictions.jsonl", encoding="utf-8") as f:
        for rec in map(json.loads, f):
            out.setdefault(rec["video"], [])
            if rec.get("start") is not None:
                out[rec["video"]].append(rec)
    return out


def regions(rows, preds: dict, cut: float) -> tuple[dict, set]:
    kept, covered = {}, set()
    for v in np.unique(rows.video):
        v = str(v)
        if v not in preds:
            continue
        covered.add(v)
        r = np.flatnonzero(rows.video == v)
        s = rows.start_seconds[r]
        flags = np.zeros(len(r), np.int8)
        for p in preds[v]:
            probs = p["probs"]
            cat = max(probs, key=probs.get)
            if cat in READS and probs[cat] >= cut:
                flags[(s >= p["start"]) & (s < p["end"])] = 1
        spans = runs(flags)
        if spans:
            kept[v] = spans
    return kept, covered


def only(rows, videos: set):
    from train import Rows
    m = np.isin(rows.video, list(videos))
    return Rows(rows.X[m], rows.y[m], rows.video[m], rows.channel[m], rows.start_seconds[m], rows.split[m], rows.feature_names)


def main() -> int:
    preds = load_preds()
    rows, _, _ = pooled()
    kept_all, covered = regions(rows, preds, 0.5)
    P = only(rows, covered)
    print(f"predictions for {len(preds)} videos; pooled videos covered: {len(covered)} of {rows.videos}\n")
    best = {}
    for cut in CUTS:
        kept, _ = regions(P, preds, cut)
        g = grade_regions(kept, P)
        oc = over_cap(kept, P)
        print(f"  pooled (covered) cut {cut:.2f}: ad time {g['coverage']:6.1%}  show lost {g['show']:4.1f} s/video  over 60 s {oc:4.1%}")
        for b in (5, 10):
            if g["show"] <= b and oc <= OVER_CAP_SHARE and (b not in best or g["coverage"] > best[b][1]):
                best[b] = (cut, g["coverage"])
    print(f"\n  cut chosen by the pooled rule: B=5 {best.get(5)}, B=10 {best.get(10)}")
    for name in ("features_channels.npz", "features_holdout4.npz"):
        if not (DATA / name).exists():
            continue
        T = load(DATA / name)
        _, cov = regions(T, preds, 0.5)
        if not cov:
            continue
        T = only(T, cov)
        print(f"\n{name}: {len(cov)} videos covered")
        for label, cut in (("default 0.5", 0.5), ("rule B=5", best.get(5, (0.5,))[0]), ("rule B=10", best.get(10, (0.5,))[0])):
            kept, _ = regions(T, preds, cut)
            g = grade_regions(kept, T)
            print(f"  {label:<12} cut {cut:.2f}: ad time {g['coverage']:6.1%}  show lost {g['show']:4.1f} s/video  "
                  f"over 60 s {over_cap(kept, T):4.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
