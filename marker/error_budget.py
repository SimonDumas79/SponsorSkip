"""Where the free tier's errors come from: NOT MARKING a read, or getting its LENGTH wrong? (2026-09-21)

Measured on the pooled out-of-fold scores that build_production.py caches (data/pooled_oof.npy:
training + holdout 1 + holdout 2, 205 videos, 280 reads), with the pooled rule (most ad time
under B s of lost show per video, at most 2% of videos over 60 s). Nothing is trained here and
no holdout-3 video is read.

It prints: what share of all ad time is lost to reads never marked, late starts, early ends and
holes; why the unmarked reads were missed (the marker never saw them, or the context model's
threshold threw them away); how lost show splits between overshoot and false alarms, with the
worst false-alarm videos; the ceilings if one stage were perfect; where late starts come from;
and a sweep of WIDER start searches (only narrower ones had been measured before).

    python marker/error_budget.py
"""
import ctypes
import glob
import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
import torch  # noqa: E402

torch.set_num_threads(4)
from build_production import OVER_CAP_SHARE, over_cap, pooled  # noqa: E402
from edge_heads import AFTER, BEFORE, INSIDE, TAIL, place  # noqa: E402
from experiments import runs  # noqa: E402
from replay import grade_regions, regions_by_video  # noqa: E402
from short_reads import place_reach  # noqa: E402
from train import DATA, LAST_LINE_SECONDS  # noqa: E402

rows, _, _ = pooled()
level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
assert level1.shape[0] == len(rows)
titles = {}
for f in glob.glob(str(DATA / "captions*" / "*.json")):
    try:
        c = json.load(open(f, encoding="utf-8"))
        titles[c["videoID"]] = (c.get("title") or "", c.get("channel") or "")
    except Exception:
        pass
videos = [str(v) for v in np.unique(rows.video)]
IDX = {v: np.flatnonzero(rows.video == v) for v in videos}
SECS = {}
for v, r in IDX.items():
    s = rows.start_seconds[r]
    SECS[v] = np.diff(np.append(s, s[-1] + LAST_LINE_SECONDS))
TH = np.round(1 / (1 + np.exp(-np.arange(2.0, 6.0, 0.1))), 4)
NV = len(videos)


def tier(th):
    return place(regions_by_video(level2, rows, th, 1), rows, p_start, p_end)


def reads_of(v):
    return runs(rows.y[IDX[v]])


def classify(kept, near=5):
    """Split every region into on-a-read / adjacent / false; attribute lost show and missed ad time."""
    out = dict(ad_total=0.0, ad_skipped=0.0, miss_untouched=0.0, miss_late_start=0.0, miss_early_end=0.0,
               miss_holes=0.0, lost_false=0.0, lost_adjacent=0.0, lost_overshoot=0.0, n_false=0, n_true=0, n_adj=0)
    per_video_false, per_read = {}, []
    for v in videos:
        r, sec, y = IDX[v], SECS[v], rows.y[IDX[v]]
        flags = np.zeros(len(r), np.int8)
        rd = reads_of(v)
        for lo, hi in kept.get(v, []):
            flags[lo:hi] = 1
            lost = float(sec[lo:hi][y[lo:hi] == 0].sum())
            if y[lo:hi].any():
                out["lost_overshoot"] += lost
                out["n_true"] += 1
            elif any(a - near <= hi and lo <= b + near for a, b in rd):
                out["lost_adjacent"] += lost
                out["n_adj"] += 1
            else:
                out["lost_false"] += lost
                out["n_false"] += 1
                per_video_false[v] = per_video_false.get(v, 0.0) + lost
        for k, (a, b) in enumerate(rd):
            L = float(sec[a:b].sum())
            f = flags[a:b]
            got = float(sec[a:b][f == 1].sum())
            out["ad_total"] += L
            out["ad_skipped"] += got
            if not f.any():
                out["miss_untouched"] += L
                kind = "untouched"
            else:
                on = np.flatnonzero(f)
                late = float(sec[a:a + on[0]].sum())
                early = float(sec[a + on[-1] + 1:b].sum())
                out["miss_late_start"] += late
                out["miss_early_end"] += early
                out["miss_holes"] += L - got - late - early
                kind = "full" if got >= 0.95 * L else "partial"
            per_read.append(dict(video=v, k=k, seconds=L, got=got, kind=kind,
                                 m_peak=float(level1[r][a:b].max()), c_peak=float(level2[r][a:b].max())))
    return out, per_video_false, per_read


def pick(candidates, budget):
    best = None
    for th, kept in candidates:
        g = grade_regions(kept, rows)
        if g["show"] <= budget and over_cap(kept, rows) <= OVER_CAP_SHARE and (
                best is None or g["coverage"] > best[2]["coverage"]):
            best = (th, kept, g)
    return best


def oracle_edges(kept):
    """Every region that overlaps a read is replaced by exactly that read's span; others stay as they are."""
    out = {}
    for v, spans in kept.items():
        rd = reads_of(v)
        new = []
        for lo, hi in spans:
            hit = [(a, b) for a, b in rd if a < hi and lo < b]
            new.extend(hit if hit else [(lo, hi)])
        out[v] = sorted(set(new))
    return out


def oracle_verifier(kept):
    """Drop every region that overlaps no read (a perfect yes/no checker); keep the edges as placed."""
    return {v: [(lo, hi) for lo, hi in spans if rows.y[IDX[v]][lo:hi].any()] for v, spans in kept.items()}


ad_min = sum(SECS[v][rows.y[IDX[v]] == 1].sum() for v in videos) / 60
print(f"pooled: {NV} videos, {rows.reads} reads, {ad_min:.0f} min of ad")
base = [(th, tier(th)) for th in TH]
for budget in (5, 10):
    th, kept, g = pick(base, budget)
    c, pvf, pr = classify(kept)
    A = c["ad_total"]
    lost_total = c["lost_false"] + c["lost_adjacent"] + c["lost_overshoot"]
    print(f"\n=== free tier, B={budget}: threshold {th}; skips {g['coverage']:.1%} of ad time, "
          f"loses {g['show']:.1f} s/video ===")
    print("AD TIME NOT SKIPPED, as a share of all ad time:")
    for k, name in (("miss_untouched", "reads never marked at all"), ("miss_late_start", "found, skip starts late"),
                    ("miss_early_end", "found, skip ends early"), ("miss_holes", "found, holes inside")):
        print(f"   {name:<30} {c[k] / A:6.1%}")
    kinds = {k: [p for p in pr if p["kind"] == k] for k in ("untouched", "partial", "full")}
    print(f"READS: {len(kinds['full'])} skipped fully (>=95%), {len(kinds['partial'])} partly, "
          f"{len(kinds['untouched'])} never touched (of {len(pr)})")
    for k in ("full", "partial", "untouched"):
        s = np.array([p["seconds"] for p in kinds[k]]) if kinds[k] else np.array([0.0])
        print(f"   {k:<9} median length {np.median(s):5.0f} s")
    part = kinds["partial"]
    if part:
        share = np.array([p["got"] / p["seconds"] for p in part])
        print(f"   partial reads: median share skipped {np.median(share):.0%}")
    un = kinds["untouched"]
    m = np.array([p["m_peak"] for p in un])
    L = np.array([p["seconds"] for p in un])
    print(f"UNTOUCHED READS, why ({len(un)}):")
    print(f"   marker's best line < 0.5 (does not recognise the text)   {np.sum(m < 0.5):3d} reads, "
          f"{L[m < 0.5].sum() / A:5.1%} of ad time")
    w = (m >= 0.5) & (m < 0.84)
    print(f"   marker 0.5-0.84 (weak)                                   {np.sum(w):3d} reads, {L[w].sum() / A:5.1%}")
    print(f"   marker >= 0.84 but context model under threshold         {np.sum(m >= 0.84):3d} reads, "
          f"{L[m >= 0.84].sum() / A:5.1%}")
    print(f"   under 30 s: {np.sum(L < 30)} of them; median length {np.median(L):.0f} s")
    print(f"LOST SHOW, {lost_total / NV:.1f} s/video:")
    for k, name in (("lost_overshoot", "overshoot around real reads"), ("lost_adjacent", "region beside a read"),
                    ("lost_false", "false alarms (no read nearby)")):
        print(f"   {name:<32} {c[k] / NV:5.1f} s/video  ({c[k] / max(lost_total, 1e-9):4.0%})")
    print(f"   regions: {c['n_true']} on reads, {c['n_adj']} beside one, {c['n_false']} false, in {len(pvf)} videos")
    print("   worst false-alarm videos:")
    for v, s in sorted(pvf.items(), key=lambda kv: -kv[1])[:8]:
        t, ch = titles.get(v, ("?", "?"))
        print(f"     {s:5.0f} s  {ch[:22]:<22} {t[:62]}")

print("\n=== CEILINGS: threshold rechosen by the same rule with one stage made perfect ===")
for name, fn in (("as shipped", lambda k: k), ("perfect edges on found reads", oracle_edges),
                 ("perfect yes/no checker (no false alarms)", oracle_verifier),
                 ("both", lambda k: oracle_edges(oracle_verifier(k)))):
    cands = [(th, fn(k)) for th, k in base]
    for budget in (5, 10):
        b = pick(cands, budget)
        if b:
            print(f"   {name:<42} B={budget:>2}: th {b[0]}  ad time {b[2]['coverage']:6.1%}  "
                  f"lost {b[2]['show']:4.1f} s/video  reads touched {b[2]['recall']:.0%}")
        else:
            print(f"   {name:<42} B={budget:>2}: nothing under the rule")
lowest = base[0][1]
pr = classify(lowest)[2]
allm = np.array([p["m_peak"] for p in pr])
print(f"\nDETECTION CEILING: at the loosest context threshold swept ({TH[0]}), "
      f"{sum(p['kind'] != 'untouched' for p in pr)} of {len(pr)} reads are touched.")
print(f"   marker's best line >= 0.5 in {np.mean(allm >= 0.5):.0%} of reads, >= 0.84 in {np.mean(allm >= 0.84):.0%}, "
      f">= 0.95 in {np.mean(allm >= 0.95):.0%}")


# ----------------------------------------------------------------- where late starts come from


def oracle_side(kept, side):
    out = {}
    for v, spans in kept.items():
        rd = reads_of(v)
        new = []
        for lo, hi in spans:
            hit = [(a, b) for a, b in rd if a < hi and lo < b]
            if not hit:
                new.append((lo, hi))
                continue
            a, b = hit[0][0], hit[-1][1]
            new.append((a, hi) if side == "start" else (lo, b))
        out[v] = new
    return out


print("CEILINGS with one edge made perfect (threshold rechosen by the rule):")
for name, fn in (("perfect starts only", lambda k: oracle_side(k, "start")),
                 ("perfect ends only", lambda k: oracle_side(k, "end"))):
    cands = [(th, fn(k)) for th, k in base]
    for budget in (5, 10):
        b = pick(cands, budget)
        print(f"   {name:<22} B={budget:>2}: th {b[0]}  ad time {b[2]['coverage']:6.1%}  lost {b[2]['show']:4.1f} s/video")

th = 0.982
ctx = regions_by_video(level2, rows, th, 1)
placed = place(ctx, rows, p_start, p_end)
late_in = late_out = early_in = early_out = on_time = 0
late_s_in = late_s_out = 0.0
over_s = under_s = 0.0
for v in videos:
    sec, y = SECS[v], rows.y[IDX[v]]
    for (lo, hi), (pa, pb) in zip(ctx.get(v, []), placed.get(v, [])):
        hit = [(a, b) for a, b in reads_of(v) if a < pb and pa < b]
        if not hit:
            continue
        a, b = hit[0][0], hit[-1][1]
        if pa > a:
            reach = lo - BEFORE <= a <= lo + INSIDE
            s = float(sec[a:pa].sum())
            if reach:
                late_in += 1; late_s_in += s
            else:
                late_out += 1; late_s_out += s
        elif pa < a:
            over_s += float(sec[pa:a].sum())
            on_time += 0
        else:
            on_time += 1
        if pb < b:
            if hi - TAIL <= b <= hi + AFTER:
                early_in += 1
            else:
                early_out += 1
n = late_in + late_out
print("");print(f"STARTS at th {th}, regions on real reads:")
print(f"   exactly right line: {on_time}")
print(f"   late, true start INSIDE the head's search window (head picked the wrong line): {late_in} regions, {late_s_in/60:.1f} min of ad")
print(f"   late, true start OUTSIDE the window (context region began too deep in the read): {late_out} regions, {late_s_out/60:.1f} min")
print(f"   early (skip began in show before the read): {over_s/60:.1f} min of show")
print(f"ENDS: early with the true end inside the window {early_in}, outside {early_out}")


# ----------------------------------------------------------------- wider start searches
print("WIDER START SEARCH (pooled CV, same rule: most ad time, <= B s lost/video, <= 2% of videos over 60 s)")
for before, after in ((20, 20), (30, 20), (40, 20), (60, 20), (40, 30)):
    cands = [(th, place_reach(regions_by_video(level2, rows, th, 1), rows, p_start, p_end, before, after)) for th in TH]
    for budget in (5, 10):
        b = pick(cands, budget)
        if b:
            print(f"   before {before:>2} after {after:>2}  B={budget:>2}: th {b[0]}  ad time {b[2]['coverage']:6.1%}  lost {b[2]['show']:4.1f} s/video  worst {b[2]['worst']:.0f} s")
        else:
            print(f"   before {before:>2} after {after:>2}  B={budget:>2}: nothing under the rule")
