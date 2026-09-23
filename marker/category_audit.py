"""How much of each KIND of promotion v3 skips, by Claude's category labels (2026-09-24).

Claude labelled every window of the 205 pooled videos into sponsor / own_product / channel_plug /
other_promo, each a "read" (the show stops for a pitch) or a "mention" (claude_label.py --categories).
This grades v3's out-of-fold regions (seed 0, B = 10, as in miss_audit.py) against those segments on the
160 videos v3 was tuned on: the share of each category's lines that v3 skips. v3 was trained on
SponsorBlock's sponsor + selfpromo labels, so this also shows what that training target left out.

    python marker/category_audit.py
"""

import collections
import datetime
import json
import sys

import numpy as np

from build_production import pooled
from detector_bakeoff import graded, pick
from stack_check import sweep_fine
from train import DATA, Rows

sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    rows, _, _ = pooled()
    _, _, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    fs, fe = np.load(DATA / "finetune_oof_edge_start_seed0.npy"), np.load(DATA / "finetune_oof_edge_resume_seed0.npy")
    d = json.load(open(DATA / "sb_dates.json"))
    cut = datetime.datetime(2022, 4, 1).timestamp() * 1000
    unseen = np.array([not (d.get(str(v), {}).get("first") and d[str(v)]["first"] < cut) for v in rows.video])
    S = Rows(rows.X[unseen], rows.y[unseen], rows.video[unseen], rows.channel[unseen], rows.start_seconds[unseen],
             rows.split[unseen], rows.feature_names)
    heads = ((p_start + fs) / 2)[unseen], ((p_end + fe) / 2)[unseen]
    sc = np.load(DATA / "stack_sbml_oof_s0.npy")[1]
    th, kept, g = pick(graded(sweep_fine(sc, S, *heads), S), 10)
    print(f"v3 seed 0 at B = 10: SponsorBlock ad time {g['coverage']:.1%}, show lost {g['show']:.1f} s\n")
    skipped = np.zeros(len(S.y), dtype=bool)
    local = {str(v): np.flatnonzero(S.video == v) for v in np.unique(S.video)}
    for v, spans in kept.items():
        for a, z in spans:
            skipped[local[v][a:z]] = True
    labels = {r["videoID"]: r["segments"] for r in json.load(open(DATA / "claude_labels_categories.json", encoding="utf-8"))}
    lines, hit, segs, whole_missed = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    sb_lines = sb_hit = 0
    for v, r in local.items():
        t = S.start_seconds[r]
        for s in labels.get(v, []):
            k = (s.get("category"), s.get("style"))
            inside = (t >= s["start"]) & (t < max(s["end"], s["start"] + 0.1))
            n = int(inside.sum())
            if n == 0:
                continue
            h = int(skipped[r][inside].sum())
            lines[k] += n
            hit[k] += h
            segs[k] += 1
            whole_missed[k] += h == 0
        sb_lines += int(S.y[r].sum())
        sb_hit += int((skipped[r] & (S.y[r] == 1)).sum())
    print(f"{'category':14s}{'style':9s}{'segments':>9s}{'lines':>7s}{'skipped':>9s}{'never touched':>15s}")
    for k in sorted(lines, key=lambda k: -lines[k]):
        print(f"{k[0]:14s}{k[1]:9s}{segs[k]:9d}{lines[k]:7d}{hit[k] / lines[k]:9.1%}{whole_missed[k] / segs[k]:15.1%}")
    print(f"\n(SponsorBlock's own labels on the same videos, by lines: {sb_hit / sb_lines:.1%} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
