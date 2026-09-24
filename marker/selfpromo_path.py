"""The self-promotion path: its own detector, its own regions, its own budget. (Simon, 2026-09-24: "I want
self promo detection now")

Every way of attaching the self-promo detector to v3 failed (veto, union, eighth input; README), because
v3 never touches ~73% of own-product reads. So this path runs BESIDE v3: the fine-tuned BGE-small trained
on Claude's non-sponsor promotion reads (own product, channel plugs, other promo; "read" style only)
marks its own regions, labelled "selfpromo", and the extension's "Skip self-promotion too" switch decides
whether they are skipped. v3's sponsor answer is untouched.

The rule (smoothing, threshold, bridging, minimum length) is chosen here on the TRUE out-of-fold stream of
the 251 videos Claude labelled (channel-grouped folds, finetune_minilm.py --target claude-nonsponsor), under
a budget on real show lost: seconds flagged where Claude saw no promotion of any kind AND SponsorBlock has
no sponsor. Its fresh grade is holdout 6, once.

    python marker/selfpromo_path.py
"""

import json
import sys

import numpy as np

from train import DATA, LAST_LINE_SECONDS

sys.stdout.reconfigure(encoding="utf-8")

TAG = "claude_nonsponsor_251"
LOST_BUDGET = 2.0      # seconds of real show lost per video, averaged: this path is opt-out, so keep it cheap
OVER_CAP_S = 60.0      # a video losing more than this counts against the same 2% rule as v3
MAX_OVER = 0.02


def seconds_per_line(rows) -> np.ndarray:
    sec = np.zeros(len(rows.video), dtype=np.float32)
    for v in np.unique(rows.video):
        r = np.flatnonzero(rows.video == v)
        s = rows.start_seconds[r]
        sec[r] = np.diff(np.append(s, s[-1] + LAST_LINE_SECONDS))
    return sec


def regions(p: np.ndarray, sec: np.ndarray, th: float, smooth_w: int, bridge: int, min_s: float) -> np.ndarray:
    """One video's scores -> a 0/1 flag per line: smoothed, thresholded, gaps of <= bridge lines closed,
    runs shorter than min_s seconds dropped."""
    q = np.convolve(p, np.ones(smooth_w) / smooth_w, mode="same") if smooth_w > 1 else p
    f = (q >= th).astype(np.int8)
    if bridge:
        on = np.flatnonzero(f)
        for a, b in zip(on[:-1], on[1:]):
            if 1 < b - a <= bridge + 1:
                f[a:b] = 1
    padded = np.concatenate(([0], f, [0]))
    d = np.diff(padded)
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if sec[a:b].sum() < min_s:
            f[a:b] = 0
    return f


def grade(flags: np.ndarray, rows, sec, promo, show) -> dict:
    caught = float((sec * flags * promo).sum() / max((sec * promo).sum(), 1e-9))
    lost = []
    for v in np.unique(rows.video):
        r = np.flatnonzero(rows.video == v)
        lost.append(float((sec[r] * flags[r] * show[r]).sum()))
    lost = np.array(lost)
    return {"caught": caught, "lost": float(lost.mean()), "over": float((lost > OVER_CAP_S).mean()),
            "worst": float(lost.max())}


def apply(stream, rows, sec, cfg) -> np.ndarray:
    out = np.zeros(len(stream), dtype=np.int8)
    for v in np.unique(rows.video):
        r = np.flatnonzero(rows.video == v)
        out[r] = regions(stream[r], sec[r], *cfg)
    return out


def main() -> int:
    # Imported here, not at the top: serving imports regions() and seconds_per_line() from this file.
    from build_production import SETS as POOLED_SETS, pooled
    from category_detector import claude_targets
    rows = pooled(POOLED_SETS + ["features_holdout5.npz"])[0]
    stream = np.load(DATA / f"finetune_oof_{TAG}_seed0.npy")
    if len(stream) != len(rows.video):
        raise SystemExit(f"stream has {len(stream)} rows, pool has {len(rows.video)}")
    t = claude_targets(rows, ["claude_labels_categories_holdout5.json"])
    promo = t["non-sponsor"].astype(np.float32)
    show = ((t["any promo"] == 0) & (rows.y == 0)).astype(np.float32)
    sec = seconds_per_line(rows)
    n_vid = len(np.unique(rows.video))
    print(f"{n_vid} videos; non-sponsor promotion reads {promo.sum():,.0f} lines / {(sec * promo).sum() / 60:,.0f} min "
          f"in {len(np.unique(rows.video[promo > 0]))} videos\n")

    results = []
    for smooth_w in (1, 3, 5, 9):
        for bridge in (0, 2, 4):
            for min_s in (5.0, 10.0, 15.0):
                for th in np.round(np.arange(0.05, 0.96, 0.05), 2):
                    cfg = (float(th), smooth_w, bridge, min_s)
                    results.append((cfg, grade(apply(stream, rows, sec, cfg), rows, sec, promo, show)))

    def best(budget):
        ok = [(c, g) for c, g in results if g["lost"] <= budget and g["over"] <= MAX_OVER]
        return max(ok, key=lambda x: (x[1]["caught"], -x[1]["lost"])) if ok else None

    print(f"{'budget':>8}  {'threshold':>9} {'smooth':>6} {'bridge':>6} {'min s':>5}   caught  show lost/video  over 60 s  worst")
    for budget in (1.0, LOST_BUDGET, 3.0, 5.0):
        b = best(budget)
        if not b:
            print(f"{budget:>7.1f}s  nothing fits")
            continue
        (th, w, br, ms), g = b
        print(f"{budget:>7.1f}s  {th:>9.2f} {w:>6} {br:>6} {ms:>5.0f}   {g['caught']:6.1%}  {g['lost']:10.1f} s     "
              f"{g['over']:6.1%}  {g['worst']:5.0f} s")
    chosen = best(LOST_BUDGET)
    if chosen:
        (th, w, br, ms), g = chosen
        json.dump({"threshold": th, "smooth": w, "bridge": br, "min_seconds": ms, "budget_s": LOST_BUDGET,
                   "cv": g, "stream": f"finetune_oof_{TAG}_seed0.npy"},
                  open(DATA / "selfpromo_rule.json", "w"), indent=1)
        print(f"\nchosen at {LOST_BUDGET} s -> data/selfpromo_rule.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
