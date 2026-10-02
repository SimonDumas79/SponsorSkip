"""Hysteresis for candidate v3: find a read at the tight threshold, then grow it outwards while the
stack's score stays >= a lower bar. Chosen on pooled CV (the same rule that fixed v3's thresholds),
never on a holdout. (2026-10-02, after the Fireship video: 65 s read, 5 s skipped.)

Variants, each graded on the v3 CV videos (pooled videos the community model never trained on):
  heads        v3 as shipped: regions at th, edges by the heads             (reference)
  grow>heads   grow the found regions, then let the heads place the edges
  heads>grow   heads place the edges, then grow the placed region
  grow-only    grown regions are the skip; no heads
For each variant the detect threshold is swept the usual way and `low` over a grid; the rule picks
(th, low) jointly: show lost <= B, <= 2% of videos over 60 s, most ad time. The fixed-th rows
(v3's own threshold, only `low` varied) are printed too.

    python marker/grow_sweep.py
"""
import datetime
import json
import sys

import numpy as np

from build_production import OVER_CAP_SHARE, over_cap, pooled
from candidate import sub
from detector_bakeoff import graded, pick
from edge_heads import merged, place
from replay import grade_regions, per_read, regions_by_video
from stack_check import SHARES
from train import DATA

LOWS = [None, 0.98, 0.95, 0.9, 0.85, 0.8, 0.7]
MIN_TH, MAX_TH = 0.97, 0.9990   # the rule has picked 0.9959 (B=10) and 0.9984 (B=5); the full 90-step grid took too long


def grow(found: dict, rows, scores: np.ndarray, low: float | None) -> dict:
    if low is None:
        return found
    out = {}
    for vid, spans in found.items():
        r = np.flatnonzero(rows.video == vid)
        s = scores[r]
        g = []
        for a, b in spans:
            a, b = int(a), int(b)
            while a > 0 and s[a - 1] >= low:
                a -= 1
            while b < len(r) and s[b] >= low:
                b += 1
            g.append((a, b))
        out[vid] = merged(g)
    return out


def variants(scores, rows, ps, pe, th, low):
    found = regions_by_video(scores, rows, th, 1)
    return {
        "heads": place(found, rows, ps, pe) if low is None else None,
        "grow>heads": place(grow(found, rows, scores, low), rows, ps, pe),
        "heads>grow": grow(place(found, rows, ps, pe), rows, scores, low),
        "grow-only": grow(found, rows, scores, low),
    }


def stats(kept, rows):
    g = grade_regions(kept, rows)
    idx = {str(v): np.flatnonzero(rows.video == v) for v in np.unique(rows.video)}
    false = sum(1 for v, spans in kept.items() for lo, hi in spans if not rows.y[idx[v]][lo:hi].any())
    pr = np.array([s for *_, s in per_read(kept, rows)])
    return dict(coverage=g["coverage"], show=g["show"], over=over_cap(kept, rows), false=false,
                full=float(np.mean(pr >= .95)), partial=float(np.mean((pr > 0) & (pr < .95))), missed=float(np.mean(pr == 0)))


def fmt(name, th, low, s):
    return (f"  {name:<11} th {th:.4f} low {('-' if low is None else f'{low:.2f}'):>4}  ad time {s['coverage']:6.1%}  "
            f"show lost {s['show']:4.1f} s  over 60 s {s['over']:4.1%}  false {s['false']:3d}  "
            f"reads full {s['full']:.0%} / partial {s['partial']:.0%} / missed {s['missed']:.0%}")


def main() -> int:
    P, is_start, is_resume = pooled()
    _, _, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    fs = np.load(DATA / "finetune_oof_edge_start_seed0.npy")
    fe = np.load(DATA / "finetune_oof_edge_resume_seed0.npy")
    d = json.load(open(DATA / "sb_dates.json"))
    cutoff = datetime.datetime(2022, 4, 1).timestamp() * 1000
    unseen = np.array([not (d.get(str(v), {}).get("first") and d[str(v)]["first"] < cutoff) for v in P.video])
    U = sub(P, unseen)
    oof3 = np.load(DATA / "stack_sbml_oof_s0.npy")[1]
    assert len(oof3) == len(U), (len(oof3), len(U))
    ps, pe = ((p_start + fs) / 2)[unseen], ((p_end + fe) / 2)[unseen]
    print(f"v3 CV videos: {U.videos} videos, {U.reads} reads, {len(U)} lines", flush=True)

    ths = [float(t) for t in np.unique(np.quantile(oof3, 1 - SHARES)) if MIN_TH <= t <= MAX_TH]
    print(f"{len(ths)} thresholds >= {MIN_TH} x {len(LOWS)} lows", flush=True)
    table = {}   # (variant, th, low) -> (kept, stats)
    log = open(DATA / "grow_sweep_rows.jsonl", "w", encoding="utf-8")
    for i, th in enumerate(ths):
        print(f"  th {i + 1}/{len(ths)} = {th:.4f}", flush=True)
        for low in LOWS:
            for name, kept in variants(oof3, U, ps, pe, th, low).items():
                if kept is not None:
                    table[(name, th, low)] = (kept, stats(kept, U))
                    log.write(json.dumps({"variant": name, "th": th, "low": low, **table[(name, th, low)][1]}) + "\n")
                    log.flush()
    ref = {}
    for b in (5, 10):
        print(f"\n== B = {b} s ==")
        rule = lambda s: s["show"] <= b and s["over"] <= OVER_CAP_SHARE
        best_ref = max(((k, v) for k, v in table.items() if k[0] == "heads" and rule(v[1])), key=lambda kv: kv[1][1]["coverage"])
        th_ref = best_ref[0][1]
        ref[b] = th_ref
        print(f"reference, v3 as shipped (rule picks th alone):")
        print(fmt("heads", th_ref, None, best_ref[1][1]))
        print(f"v3's threshold held fixed, only low varied:")
        for name in ("grow>heads", "heads>grow", "grow-only"):
            for low in LOWS[1:]:
                s = table[(name, th_ref, low)][1]
                print(fmt(name, th_ref, low, s) + ("" if rule(s) else "   (fails the rule)"))
        print(f"rule picks th and low jointly:")
        for name in ("grow>heads", "heads>grow", "grow-only"):
            ok = [(k, v) for k, v in table.items() if k[0] == name and rule(v[1])]
            if not ok:
                print(f"  {name:<11} nothing meets the rule"); continue
            k, v = max(ok, key=lambda kv: kv[1][1]["coverage"])
            print(fmt(name, k[1], k[2], v[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
