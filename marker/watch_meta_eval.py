"""What the watch page's free hints are worth: the paid-promotion flag and creator chapters. (2026-09-22)

Reads data/watch_meta.json (fetch_watch_meta.py) for the pooled 205 videos only (the channel set is
the held-out test and is not looked at here).

  coverage   how often the flag and chapters appear, split by what the video's labels contain
  chapters   chapters whose title names a read ("Sponsor", "Ad", "Promo"...): where they sit against
             the labelled reads, in seconds, at both edges
  gate       two thresholds instead of one: a looser one on flagged videos, a stricter one elsewhere,
             both chosen together by the pooled rule
  edges      where a sponsor-titled chapter overlaps a region, the chapter's edges replace the model's;
             a sponsor chapter with no region is skipped anyway (the creator said so)

Every system here is the shipped free tier's context model + edge heads, pooled out-of-fold scores.

    python marker/watch_meta_eval.py
"""

import json
import re
import sys

import numpy as np

from build_production import OVER_CAP_SHARE, over_cap, pooled
from detector_bakeoff import line
from edge_heads import merged, place
from experiments import runs
from replay import grade_regions, regions_by_video
from stack_check import SHARES
from train import DATA, LAST_LINE_SECONDS

sys.stdout.reconfigure(encoding="utf-8")
PROMO_TITLE = re.compile(r"\b(sponsor\w*|ad|ads|advert\w*|promo\w*|brought to you|partner\w*|thanks to|merch|patreon)\b", re.I)


def chapter_spans(meta: dict, duration: float) -> list[tuple[float, float, str]]:
    ch = sorted(meta.get("chapters") or [], key=lambda c: c["start"])
    return [(c["start"], ch[k + 1]["start"] if k + 1 < len(ch) else duration, c["title"]) for k, c in enumerate(ch)]


def main() -> int:
    rows, _, _ = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    meta = json.loads((DATA / "watch_meta.json").read_text(encoding="utf-8"))
    videos = [str(v) for v in np.unique(rows.video)]
    idx = {v: np.flatnonzero(rows.video == v) for v in videos}
    have = [v for v in videos if meta.get(v, {}).get("ok")]
    print(f"pooled: {len(videos)} videos, watch page read for {len(have)}\n")

    # --- coverage
    cats = {}
    for d in [np.load(DATA / f) for f in ("features.npz", "features_tail.npz", "features_holdout2.npz")]:
        for v, c in zip(d["video"], d["category"]):
            if c:
                cats.setdefault(str(v), set()).add(str(c))
    groups = {"sponsor read(s)": [v for v in have if "sponsor" in cats.get(v, set())],
              "self-promo only": [v for v in have if cats.get(v) == {"selfpromo"}]}
    for name, vs in groups.items():
        if vs:
            print(f"  {name:<18} {len(vs):3d} videos: paid flag on {np.mean([meta[v]['paid'] for v in vs]):.0%}, "
                  f"chapters on {np.mean([bool(meta[v]['chapters']) for v in vs]):.0%}")

    # --- chapters that name a read
    starts, ends, hits, promo_ch = [], [], 0, 0
    secs_of = {}
    for v in have:
        r = idx[v]
        s = rows.start_seconds[r]
        dur = float(s[-1] + LAST_LINE_SECONDS)
        secs_of[v] = s
        reads = [(s[a], s[b] if b < len(s) else dur) for a, b in runs(rows.y[r])]
        for cs, ce, title in chapter_spans(meta[v], dur):
            if not PROMO_TITLE.search(title):
                continue
            promo_ch += 1
            over = [(a, b) for a, b in reads if a < ce and cs < b]
            if over:
                hits += 1
                a, b = min(over, key=lambda ab: abs(ab[0] - cs))
                starts.append(cs - a)
                ends.append(ce - b)
    print(f"\n  chapters whose title names a read: {promo_ch}; {hits} overlap a labelled read")
    if starts:
        st, en = np.abs(starts), np.abs(ends)
        print(f"    start error median {np.median(st):.1f} s ({np.mean(st <= 2):.0%} within 2 s); "
              f"end error median {np.median(en):.1f} s ({np.mean(en <= 2):.0%} within 2 s)")
    n_reads = int(sum(len(runs(rows.y[idx[v]])) for v in have))
    print(f"    reads with such a chapter: {hits} of {n_reads}")

    # --- the flag as a gate, and chapters as edges
    ths = np.unique(np.quantile(level2, 1 - SHARES))
    placed = {float(t): place(regions_by_video(level2, rows, float(t), 1), rows, p_start, p_end) for t in ths}
    flagged = {v for v in have if meta[v]["paid"]}
    coarse = [float(t) for t in np.unique(np.quantile(level2, 1 - np.geomspace(0.004, 0.25, 30)))]
    placed.update({t: place(regions_by_video(level2, rows, t, 1), rows, p_start, p_end) for t in coarse if t not in placed})

    def rule(cands, budget):
        best = None
        for label, kept in cands:
            g = grade_regions(kept, rows)
            if g["show"] <= budget and over_cap(kept, rows) <= OVER_CAP_SHARE and (best is None or g["coverage"] > best[2]["coverage"]):
                best = (label, kept, g)
        return best

    def with_chapters(kept):
        out = {}
        for v in videos:
            spans = list(kept.get(v, []))
            if v in secs_of:
                s = secs_of[v]
                dur = float(s[-1] + LAST_LINE_SECONDS)
                for cs, ce, title in chapter_spans(meta[v], dur):
                    if not PROMO_TITLE.search(title):
                        continue
                    lo, hi = int(np.searchsorted(s, cs)), int(np.searchsorted(s, ce))
                    if hi <= lo:
                        continue
                    spans = [sp for sp in spans if not (sp[0] < hi and lo < sp[1])] + [(lo, hi)]
            if spans:
                out[v] = merged(spans)
        return out

    shipped = [(t, k) for t, k in placed.items()]
    gated = [((t1, t2), {v: (placed[t1] if v in flagged else placed[t2]).get(v, []) for v in videos})
             for t1 in coarse for t2 in coarse if t1 <= t2]
    chap = [(t, with_chapters(k)) for t, k in placed.items()]
    print()
    for budget in (5, 10):
        print(line(f"B={budget:>2} shipped free tier", rule(shipped, budget)))
        b = rule(gated, budget)
        print(line(f"B={budget:>2}   + paid flag gate (looser when flagged)", b))
        print(line(f"B={budget:>2}   + sponsor chapters as edges", rule(chap, budget)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
