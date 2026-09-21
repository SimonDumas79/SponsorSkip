"""Why do short reads get skipped so little, and what fixes it? Measured on cross-validation only.

Reads under 30 s are 80 of the 280 labelled reads, and the free tier skips only
13-28% of their time (2026-09-21, both holdouts). This script asks the question
that decides the fix: are they MISSED (no stage fires on them at all) or FOUND
AND CUT SHORT (the context model or the edge heads trim them to nothing)?

Everything here is out-of-fold with the marker's own channel folds, on the
training set only. The holdouts are not touched: they have been graded, and
holdout 3 is still being crawled.

    python marker/short_reads.py            # the diagnosis: every read by length bucket
    python marker/short_reads.py --levers   # plus each candidate fix, chosen by the same rule as system_eval.py
"""

import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from context_stack import REACH, context_features, fit_stage2, logit
from edge_heads import AFTER, BEFORE, INSIDE, TAIL, edge_oof, place, soft
from experiments import BASE, cross_validate, runs
from replay import HEADER, grade_regions, per_read, regions_by_video, row
from system_eval import WORST_CAP, marker_alone
from train import DATA, LAST_LINE_SECONDS, load

FREE_THRESHOLD = 0.9781   # predict.py: the free tier's CV-chosen context threshold (B = 5 and 10 s)
BUCKETS = (("<15 s", 0, 15), ("15-30 s", 15, 30), ("30-90 s", 30, 90), (">=90 s", 90, 1e9))
CTX_THRESHOLDS = np.round(1 / (1 + np.exp(-np.arange(2.0, 6.0, 0.1))), 4)


def low_priority() -> None:
    """Simon may be using the PC: never fight him for the CPU."""
    if sys.platform == "win32":
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)   # BELOW_NORMAL_PRIORITY_CLASS
    torch.set_num_threads(4)


def read_table(rows, d) -> list[dict]:
    """One record per labelled read: where it is, how long, its category."""
    out = []
    for vid in np.unique(rows.video):
        r = np.flatnonzero(rows.video == vid)
        s = rows.start_seconds[r]
        on_screen = np.diff(np.append(s, s[-1] + LAST_LINE_SECONDS))
        for k, (a, b) in enumerate(runs(rows.y[r])):
            cats = set(d["category"][r][a:b].tolist()) - {""}
            out.append({"video": str(vid), "k": k, "lo": a, "hi": b, "rows": r,
                        "seconds": float(on_screen[a:b].sum()), "lines": b - a,
                        "category": "+".join(sorted(cats)) or "?"})
    return out


def peaks(reads: list[dict], scores: np.ndarray) -> np.ndarray:
    return np.array([scores[rd["rows"][rd["lo"]:rd["hi"]]].max() for rd in reads])


def bucket_of(seconds: float) -> str:
    return next(name for name, lo, hi in BUCKETS if lo <= seconds < hi)


def coverage_by_bucket(kept: dict, rows, reads: list[dict]) -> dict[str, tuple[int, float, float]]:
    """Per length bucket: (reads, share touched at all, mean share of the read's time skipped)."""
    pr = per_read(kept, rows)   # same order as read_table: by video, then read number
    out = {}
    for name, lo, hi in BUCKETS:
        idx = [i for i, rd in enumerate(reads) if lo <= rd["seconds"] < hi]
        if idx:
            cov = np.array([pr[i][3] for i in idx])
            out[name] = (len(idx), float((cov > 0).mean()), float(cov.mean()))
    return out


def bucket_line(by_bucket: dict) -> str:
    return " | ".join(f"{name} n={n} touched {t:.0%} skipped {c:.0%}" for name, (n, t, c) in by_bucket.items())


def free_tier(level2: np.ndarray, rows, p_start, p_end, threshold: float, placer=place) -> dict:
    return placer(regions_by_video(level2, rows, threshold, 1), rows, p_start, p_end)


def choose(name: str, candidates, rows, reads, budget: float, over_cap=None):
    """system_eval.py's rule: the most ad time skipped with <= budget s lost per video, no video over 60 s.
    With over_cap (build_production.py's rule for big sets): at most 2% of videos over 60 s instead."""
    best = None
    for label, kept in candidates:
        g = grade_regions(kept, rows)
        within_cap = over_cap(kept, rows) <= 0.02 if over_cap else g["worst"] <= WORST_CAP
        if g["show"] <= budget and within_cap and (
                best is None or (g["coverage"], g["recall"], -g["show"]) > (
                    best[2]["coverage"], best[2]["recall"], -best[2]["show"])):
            best = (label, kept, g)
    if best is None:
        print(f"  B={budget}: {name}: nothing stays under the budget")
        return None
    label, kept, g = best
    print(row(f"B={budget} {name} {label}", g))
    print(f"      {bucket_line(coverage_by_bucket(kept, rows, reads))}")
    return best


# ----------------------------------------------------------------------------- the levers


def place_scaled(kept, rows, p_start, p_end):
    """edge_heads.place, but a short region may not push its start deep inside itself or its end back past
    its own middle: the reach INTO the region is capped at half the region's length."""
    placed = {}
    for vid, spans in kept.items():
        r = np.flatnonzero(rows.video == vid)
        ps, pe = p_start[r], p_end[r]
        for lo, hi in spans:
            inside = min(INSIDE, max(1, (hi - lo) // 2))
            a0, a1 = max(0, lo - BEFORE), min(len(r), lo + inside)
            start = a0 + int(np.argmax(ps[a0:a1]))
            b0, b1 = max(start + 1, hi - min(TAIL, max(1, (hi - lo) // 2))), min(len(r), hi + AFTER + 1)
            end = b0 + int(np.argmax(pe[b0:b1])) if b1 > b0 else hi
            placed.setdefault(vid, []).append((start, max(end, start + 1)))
    return placed


def multiscale_features(rows, scores: np.ndarray) -> np.ndarray:
    """context_features plus the same summaries at short reaches (3 and 7 lines), so a bump the width of
    a 10-second read is visible to the second stage instead of being averaged away over 15 lines."""
    base = context_features(rows, scores)
    extra = []
    for reach in (3, 7):
        out = np.zeros((len(rows), 5), dtype=np.float32)
        for vid in np.unique(rows.video):
            r = np.flatnonzero(rows.video == vid)
            z = logit(scores[r])
            padded = np.pad(z, reach, constant_values=-6.0)
            window = np.lib.stride_tricks.sliding_window_view(padded, 2 * reach + 1)
            before, after = window[:, :reach], window[:, reach + 1:]
            out[r] = np.column_stack([before.mean(1), after.mean(1), window.mean(1), window.max(1),
                                      np.minimum(before.mean(1), after.mean(1))])
        extra.append(out)
    return np.column_stack([base] + extra)


def fit_weighted(F: np.ndarray, y: np.ndarray, w: np.ndarray, hidden: int = 32, seed: int = 0, epochs: int = 15):
    """fit_stage2 with a weight per row, so each READ can count the same however many lines it has."""
    mean, std = F.mean(0), F.std(0) + 1e-6
    torch.manual_seed(seed)
    Xt, yt, wt = (torch.from_numpy((F - mean) / std).float(), torch.from_numpy(y).float(),
                  torch.from_numpy(w.astype(np.float32)))
    model = nn.Sequential(nn.Linear(F.shape[1], hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1))
    pos_weight = torch.tensor([(len(yt) - yt.sum()) / yt.sum()])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            idx = order[i:i + 256]
            opt.zero_grad()
            ((loss_fn(model(Xt[idx]).squeeze(1), yt[idx]) * wt[idx]).sum() / wt[idx].sum()).backward()
            opt.step()
    model.eval()

    def predict(G: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(model(torch.from_numpy((G - mean) / std).float()).squeeze(1)).numpy()
    return predict


def oof(rows, F: np.ndarray, y: np.ndarray, fitter, folds: int = 5, seed: int = 0) -> np.ndarray:
    channels = np.array(sorted(set(rows.channel)))
    np.random.default_rng(seed).shuffle(channels)
    p = np.zeros(len(rows), dtype=np.float32)
    for fold in np.array_split(channels, folds):
        test = np.isin(rows.channel, fold)
        p[test] = fitter(F[~test], y[~test], test)(F[test])
    return p


def read_weights(rows, reads: list[dict], floor: float = 0.05) -> np.ndarray:
    """1 for lines outside a read; inside, 1 / (lines in that read) scaled so the mean positive weight is 1.
    A 6-line read then weighs as much in the loss as a 60-line one, instead of a tenth."""
    w = np.ones(len(rows), dtype=np.float32)
    for rd in reads:
        w[rd["rows"][rd["lo"]:rd["hi"]]] = 1.0 / rd["lines"]
    pos = rows.y == 1
    w[pos] = np.maximum(w[pos] / w[pos].mean(), floor)
    return w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levers", action="store_true", help="also measure each candidate fix on CV")
    args = ap.parse_args()
    low_priority()

    feats = DATA / "features.npz"
    rows, d = load(feats), np.load(feats)
    reads = read_table(rows, d)
    cache = DATA / "level1_oof.npy"
    if cache.exists() and len(np.load(cache)) == len(rows):
        level1 = np.load(cache)
    else:
        level1 = cross_validate(BASE, rows, seed=0)
        np.save(cache, level1)
    level2 = np.load(DATA / "context_stack_oof.npy")
    p_start, p_end = np.load(DATA / "edge_heads_oof.npy")
    F = context_features(rows, level1)

    seconds = np.array([rd["seconds"] for rd in reads])
    print(f"CV set: {rows.videos} videos, {len(reads)} reads; {(seconds < 30).sum()} under 30 s "
          f"({(seconds < 15).sum()} under 15 s). Median read {np.median(seconds):.0f} s, "
          f"{np.median([rd['lines'] for rd in reads]):.0f} lines; the context model's reach is {REACH} lines.")
    print("\nby bucket: reads, category mix, median lines, and the highest OUT-OF-FOLD score any line inside")
    print("the read gets from the marker and from the context model (a read whose peak is under the")
    print(f"threshold {FREE_THRESHOLD} cannot be skipped by the free tier, whatever the edge heads do):")
    pk1, pk2 = peaks(reads, level1), peaks(reads, level2)
    for name, lo, hi in BUCKETS:
        idx = np.flatnonzero((seconds >= lo) & (seconds < hi))
        if not len(idx):
            continue
        cats = {}
        for i in idx:
            cats[reads[i]["category"]] = cats.get(reads[i]["category"], 0) + 1
        mix = ", ".join(f"{c} {n}" for c, n in sorted(cats.items(), key=lambda kv: -kv[1]))
        print(f"  {name:<8} n={len(idx):3d}  lines med {np.median([reads[i]['lines'] for i in idx]):3.0f}  "
              f"marker peak med {np.median(pk1[idx]):.3f} (>=0.84: {(pk1[idx] >= 0.84).mean():.0%})  "
              f"context peak med {np.median(pk2[idx]):.3f} (>={FREE_THRESHOLD}: "
              f"{(pk2[idx] >= FREE_THRESHOLD).mean():.0%})   [{mix}]")

    print("\nthe shipped free tier on CV (context model at its threshold, edge heads), read by read:")
    print(HEADER)
    regions = regions_by_video(level2, rows, FREE_THRESHOLD, 1)
    for label, kept in (("context regions, own edges", regions),
                        ("context regions, edge heads (= free tier)", place(regions, rows, p_start, p_end))):
        print(row(label, grade_regions(kept, rows)))
        print(f"      {bucket_line(coverage_by_bucket(kept, rows, reads))}")
    alone = marker_alone(rows, level1, 10)
    if alone:
        kept = regions_by_video(level1, rows, alone[0], alone[1])
        print(row(f"marker alone, B=10 rule (th {alone[0]}, smooth {alone[1]})", grade_regions(kept, rows)))
        print(f"      {bucket_line(coverage_by_bucket(kept, rows, reads))}")

    # Where does a found short read lose its time: before the heads, or after?
    short = [i for i, rd in enumerate(reads) if rd["seconds"] < 30]
    before_heads, after_heads = per_read(regions, rows), per_read(place(regions, rows, p_start, p_end), rows)
    cut = sum(1 for i in short if before_heads[i][3] > 0 and after_heads[i][3] < before_heads[i][3] - 0.1)
    grown = sum(1 for i in short if after_heads[i][3] > before_heads[i][3] + 0.1)
    never = sum(1 for i in short if before_heads[i][3] == 0)
    print(f"\nshort reads (<30 s): {len(short)} in all; {never} never touched by the context model at "
          f"{FREE_THRESHOLD}; of the rest the edge heads cut {cut} by more than 10 points and grew {grown}.")

    if not args.levers:
        return 0

    # ------------------------------------------------------------------ levers, each chosen by the rule on CV
    print("\nLEVERS, each chosen by system_eval.py's rule on CV (most ad time under B s lost per video, worst <= 60 s):")
    print(HEADER)
    is_start, is_resume = soft(d["is_start"], rows.video), soft(d["is_resume"], rows.video)
    y = rows.y.astype(np.float32)
    plain = lambda Ftr, ytr, test: fit_stage2(Ftr, ytr, hidden=32)

    variants = {}
    variants["A baseline: context + heads"] = (level2, p_start, p_end, place)
    variants["B heads reach capped by region length"] = (level2, p_start, p_end, place_scaled)

    print("  training the multi-scale context model out of fold...", flush=True)
    F_ms = multiscale_features(rows, level1)
    level2_ms = oof(rows, F_ms, y, plain)
    variants["C multi-scale context (+3/+7 line summaries)"] = (level2_ms, p_start, p_end, place)
    ps_ms, pe_ms = oof(rows, F_ms, is_start, plain), oof(rows, F_ms, is_resume, plain)
    variants["D multi-scale context AND multi-scale heads"] = (level2_ms, ps_ms, pe_ms, place)

    print("  training the read-weighted context model out of fold...", flush=True)
    w = read_weights(rows, reads)
    level2_w = oof(rows, F, y, lambda Ftr, ytr, test: fit_weighted(Ftr, ytr, w[~test]))
    variants["E read-weighted context (each read counts once)"] = (level2_w, p_start, p_end, place)
    level2_msw = oof(rows, F_ms, y, lambda Ftr, ytr, test: fit_weighted(Ftr, ytr, w[~test]))
    variants["F multi-scale AND read-weighted"] = (level2_msw, p_start, p_end, place)

    chosen = {}
    for name, (l2, ps, pe, placer) in variants.items():
        for budget in (5, 10):
            cands = [(f"th {t}", free_tier(l2, rows, ps, pe, t, placer)) for t in CTX_THRESHOLDS]
            best = choose(name, cands, rows, reads, budget)
            if best:
                chosen[(name, budget)] = best
    np.save(DATA / "short_reads_oof.npy", np.stack([level2_ms, level2_w, level2_msw, ps_ms, pe_ms]))
    print("\n(out-of-fold scores of the variants saved to data/short_reads_oof.npy; nothing exported)")
    return 0



# ----------------------------------------------------------------------------- the cap, and guards


def video_facts(rows, kept: dict) -> list[dict]:
    """Per video: seconds of real show lost, share of the video flagged, the longest flagged region."""
    out = []
    for vid in np.unique(rows.video):
        r = np.flatnonzero(rows.video == vid)
        s = rows.start_seconds[r]
        on_screen = np.diff(np.append(s, s[-1] + LAST_LINE_SECONDS))
        flags = np.zeros(len(r), dtype=np.int8)
        longest = 0.0
        for a, b in kept.get(str(vid), []):
            flags[a:b] = 1
            longest = max(longest, float(on_screen[a:b].sum()))
        lost = float(on_screen[(flags == 1) & (rows.y[r] == 0)].sum())
        out.append({"video": str(vid), "lost": lost, "share": float(on_screen[flags == 1].sum() / on_screen.sum()),
                    "longest": longest, "ad_seconds": float(on_screen[rows.y[r] == 1].sum()),
                    "length": float(on_screen.sum()), "regions": len(kept.get(str(vid), []))})
    return out


def guard(kept: dict, rows, max_region: float = 0.0, max_share: float = 0.0) -> dict:
    """Structural guards, as claude_label.py uses on Claude: a region longer than max_region seconds is
    not skipped (no read is that long), and a video where the flagged time exceeds max_share of its length
    is left alone entirely (the extension falls back to SponsorBlock)."""
    out = {}
    for vid, spans in kept.items():
        r = np.flatnonzero(rows.video == vid)
        s = rows.start_seconds[r]
        on_screen = np.diff(np.append(s, s[-1] + LAST_LINE_SECONDS))
        keep = [(a, b) for a, b in spans if not max_region or on_screen[a:b].sum() <= max_region]
        if max_share and keep and sum(on_screen[a:b].sum() for a, b in keep) / on_screen.sum() > max_share:
            keep = []
        if keep:
            out[vid] = keep
    return out


def foreign_title(title: str) -> bool:
    """A title with letters outside the Latin scripts: the English caption track is then almost certainly a
    machine translation, the class behind the worst video in every set so far."""
    return any(ch.isalpha() and ord(ch) > 0x024F for ch in (title or ""))


def leave_alone(kept: dict, videos: set) -> dict:
    """These videos get no regions: the extension would fall back to SponsorBlock for them."""
    return {v: rs for v, rs in kept.items() if v not in videos}


def sweep(rows, reads, level2, p_start, p_end, titles: dict, skip_videos: set = frozenset()) -> None:
    print("\nTHE CAP: the free tier at every threshold. Which video is the worst, and is it the cap or the budget")
    print("that stops the threshold from dropping (the sweep stops at 0.85):")
    if skip_videos:
        print(f"  ({len(skip_videos)} videos with non-Latin titles are left to SponsorBlock: no regions in them)")
        base_tier = free_tier
        free_tier_local = lambda *a, **k: leave_alone(base_tier(*a, **k), skip_videos)
    else:
        free_tier_local = free_tier
    print(HEADER)
    longest_read = max(rd["seconds"] for rd in reads)
    print(f"  (the longest labelled read in this set is {longest_read:.0f} s)")
    for t in CTX_THRESHOLDS[::-1]:
        if t < 0.85:
            break
        kept = free_tier_local(level2, rows, p_start, p_end, t)
        g = grade_regions(kept, rows)
        facts = max(video_facts(rows, kept), key=lambda f: f["lost"])
        flag = "" if g["worst"] <= WORST_CAP else "  <- over the cap"
        print(row(f"th {t}", g) + flag)
        print(f"      worst: {facts['video']} loses {facts['lost']:.0f} s; {facts['share']:.0%} of the video flagged, "
              f"longest region {facts['longest']:.0f} s, {facts['ad_seconds']:.0f} s of real ad in it; "
              f"{titles.get(facts['video'], '?')[:60]}")
        b = coverage_by_bucket(kept, rows, reads)
        print(f"      short reads: <15 s skipped {b['<15 s'][2]:.0%}, 15-30 s skipped {b['15-30 s'][2]:.0%}")

    print("\nGUARDS: the same sweep with structural guards, chosen by the rule (worst <= 60 s, B = 5 and 10):")
    print(HEADER)
    for max_region, max_share in ((0, 0), (180, 0), (240, 0), (0, 0.35), (180, 0.35), (240, 0.35), (240, 0.5)):
        name = f"region<={max_region or 'any'}s share<={max_share or 'any'}"
        cands = [(f"th {t}", guard(free_tier_local(level2, rows, p_start, p_end, t), rows, max_region, max_share))
                 for t in CTX_THRESHOLDS]
        for budget in (5, 10):
            choose(name, cands, rows, reads, budget)


def place_reach(kept, rows, p_start, p_end, before: int, after: int, proportional: bool = False):
    """edge_heads.place with the OUTWARD reach as a parameter: how far before a region the start may move,
    how far past it the end may move. The heads were built to fix true reads' edges; on a false region
    that same reach widens the damage (measured: 44 s -> 144 s on one video). With proportional=True the
    outward reach is also capped at half the region's own length."""
    placed = {}
    for vid, spans in kept.items():
        r = np.flatnonzero(rows.video == vid)
        ps, pe = p_start[r], p_end[r]
        for lo, hi in spans:
            b_reach, a_reach = before, after
            if proportional:
                b_reach, a_reach = min(before, max(2, (hi - lo) // 2)), min(after, max(2, (hi - lo) // 2))
            a0, a1 = max(0, lo - b_reach), min(len(r), lo + INSIDE)
            start = a0 + int(np.argmax(ps[a0:a1]))
            b0, b1 = max(start + 1, hi - TAIL), min(len(r), hi + a_reach + 1)
            end = b0 + int(np.argmax(pe[b0:b1])) if b1 > b0 else hi
            placed.setdefault(vid, []).append((start, max(end, start + 1)))
    return placed


def reach_main() -> int:
    low_priority()
    feats = DATA / "features.npz"
    rows, d = load(feats), np.load(feats)
    reads = read_table(rows, d)
    level2 = np.load(DATA / "context_stack_oof.npy")
    p_start, p_end = np.load(DATA / "edge_heads_oof.npy")
    print("OUTWARD REACH of the edge heads, chosen by the rule on CV (the shipped heads: before 20, after 20):")
    print(HEADER)
    settings = [(20, 20, False), (10, 20, False), (20, 10, False), (10, 10, False), (5, 5, False), (3, 3, False),
                (0, 0, False), (20, 20, True), (10, 10, True)]
    for before, after, prop in settings:
        name = f"before {before} after {after}{' prop' if prop else ''}"
        cands = [(f"th {t}", place_reach(regions_by_video(level2, rows, t, 1), rows, p_start, p_end, before, after,
                                          prop)) for t in CTX_THRESHOLDS]
        for budget in (5, 10):
            choose(name, cands, rows, reads, budget)
    return 0


def pooled_main() -> int:
    """The levers again on the pooled 205-video set under build_production.py's rule: 2.3x the reads,
    so a real effect has a better chance of standing out from the noise."""
    from build_production import over_cap, pooled
    low_priority()
    rows, is_start, is_resume = pooled()
    cache = DATA / "pooled_oof.npy"
    if not cache.exists():
        raise SystemExit("run build_production.py first: it caches the pooled out-of-fold scores")
    level1, level2, p_start, p_end = np.load(cache)
    F = context_features(rows, level1)
    y = rows.y.astype(np.float32)
    # read_table needs each row's category; the pooled rows carry none, so mark every read "?"
    class D(dict):
        def __getitem__(self, k):
            return np.array([""] * len(rows)) if k == "category" else super().__getitem__(k)
    reads = read_table(rows, D())
    seconds = np.array([rd["seconds"] for rd in reads])
    print(f"POOLED: {rows.videos} videos, {len(reads)} reads, {(seconds < 30).sum()} under 30 s. Rule: most ad time "
          f"under B s lost per video, at most 2% of videos over 60 s.")
    print(HEADER)
    plain = lambda Ftr, ytr, test: fit_stage2(Ftr, ytr, hidden=32)
    variants = {"A baseline: context + heads": (level2, p_start, p_end)}
    print("  training the multi-scale and read-weighted context models out of fold...", flush=True)
    F_ms = multiscale_features(rows, level1)
    variants["C multi-scale context"] = (oof(rows, F_ms, y, plain), p_start, p_end)
    w = read_weights(rows, reads)
    variants["E read-weighted context"] = (oof(rows, F, y, lambda Ftr, ytr, test: fit_weighted(Ftr, ytr, w[~test])),
                                          p_start, p_end)
    variants["F multi-scale AND read-weighted"] = (
        oof(rows, F_ms, y, lambda Ftr, ytr, test: fit_weighted(Ftr, ytr, w[~test])), p_start, p_end)
    for name, (l2, ps, pe) in variants.items():
        for budget in (5, 10):
            cands = [(f"th {t}", free_tier(l2, rows, ps, pe, t)) for t in CTX_THRESHOLDS]
            choose(name, cands, rows, reads, budget, over_cap=over_cap)
    return 0


def titles_by_video() -> dict:
    """Video titles, from the caption files (candidates.json has none)."""
    import json
    out = {}
    for folder in ("captions", "captions_tail"):
        for path in (DATA / folder).glob("*.json"):
            with path.open(encoding="utf-8") as f:
                head = f.read(2000)   # the title is in the first few hundred characters; the file is long
            try:
                title = json.loads(head[:head.index('"channel"')].rstrip().rstrip(",") + "}").get("title") or ""
            except (ValueError, KeyError):
                title = ""
            out[path.stem] = title
    return out


def cap_main() -> int:
    low_priority()
    feats = DATA / "features.npz"
    rows, d = load(feats), np.load(feats)
    reads = read_table(rows, d)
    level2 = np.load(DATA / "context_stack_oof.npy")
    p_start, p_end = np.load(DATA / "edge_heads_oof.npy")
    titles = titles_by_video()
    skip = set()
    if "--no-translated" in sys.argv:
        skip = {str(v) for v in np.unique(rows.video) if foreign_title(titles.get(str(v), ""))}
    sweep(rows, reads, level2, p_start, p_end, titles, skip)
    return 0


if __name__ == "__main__":
    modes = {"--cap": cap_main, "--reach": reach_main, "--pooled": pooled_main}
    raise SystemExit(next((fn for flag, fn in modes.items() if flag in sys.argv), main)())
