"""Every free detector we built, side by side on the pooled 205 videos, then combined. (2026-09-21)

Simon's question: compare the marker with the other options we prepared for it, and combine them all.

The detectors, each scored out of fold with the same channel folds (seed 0) as everything else:
  cues         the six hand-written phrase patterns, nothing learned
  description  the video description names the sponsor (the two label-free description columns)
  structure    a linear model on seam + cues + position only, no sentence meaning
  meaning      a linear model on the MiniLM sentence meaning only
  marker       the shipped linear marker (meaning + structure)
  potion       the same linear marker on potion-base-8M embeddings (the in-browser candidate)
  sequence     the end-to-end sequence model (sequence_model.py, its best saved variant)

Each one goes through the free tier's own pipeline: a context model trained on ITS scores, the
edge heads, and the pooled rule (the most ad time with at most B s of real show lost per video
and at most 2% of videos over 60 s). Thresholds are swept by the share of lines flagged, so
detectors with different calibrations are compared fairly.

Then the combinations:
  union    every detector's chosen regions, all skipped
  average  the level-one detectors' logits averaged, then the usual context model and heads
  stack    ONE context model reading every detector's scores in its window, plus cues and description
  agree    the shipped free tier's regions, kept only when a second detector also fires inside
           them (aimed at false alarms, which is what pins the threshold: error_budget.py)

Cross-validated only; nothing here reads holdout 3, and nothing ships from here.

    python marker/detector_bakeoff.py            # scores are cached in data/bakeoff_streams.npz
"""

import ctypes
import sys

import numpy as np
import torch

from build_production import OVER_CAP_SHARE, over_cap, pooled
from context_stack import context_features, logit, out_of_fold
from edge_heads import edge_oof, merged, place, soft
from experiments import BASE, cross_validate, fit
from replay import grade_regions, per_read, regions_by_video
from train import DATA, load, scaling

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(4)

CACHE = DATA / "bakeoff_streams.npz"
POTION = ["features_potion_cv.npz", "features_potion_tail.npz", "features_potion_holdout2.npz"]
SHARES = np.geomspace(0.004, 0.25, 36)   # share of all lines flagged, the threshold sweep
BUDGETS = (5, 10)


def potion_rows(rows) -> np.ndarray:
    """The potion feature files, concatenated in the pooled order and checked line by line against it."""
    parts = [load(DATA / f) for f in POTION]
    video = np.concatenate([p.video for p in parts])
    assert len(video) == len(rows) and (video == rows.video).all(), "potion rows are not in the pooled order"
    assert (np.concatenate([p.y for p in parts]) == rows.y).all()
    X = np.concatenate([p.X for p in parts])
    return X[:, :X.shape[1] - 2]   # the embedding + seam + cues + position; the description columns dropped


def cv_linear(X: np.ndarray, rows, seed: int = 0, folds: int = 5) -> np.ndarray:
    """experiments.cross_validate for an arbitrary column set (the same folds, the same BASE training)."""
    channels = np.array(sorted(set(rows.channel)))
    np.random.default_rng(seed).shuffle(channels)
    oof = np.zeros(len(rows), dtype=np.float32)
    for fold in np.array_split(channels, folds):
        test = np.isin(rows.channel, fold)
        mean, std = scaling(X[~test])
        model = fit(BASE, (X[~test] - mean) / std, rows.y[~test], seed)
        with torch.no_grad():
            z = model(torch.from_numpy((X[test] - mean) / std).float()).squeeze(1).numpy()
        oof[test] = 1 / (1 + np.exp(-z))
    return oof


def level_one(rows) -> dict[str, np.ndarray]:
    cached = dict(np.load(CACHE)) if CACHE.exists() else {}
    names = [str(n) for n in rows.feature_names]
    cue_cols = [i for i, n in enumerate(names) if n.startswith("cue_")]
    streams = {}
    level1, level2, _, _ = np.load(DATA / "pooled_oof.npy")
    streams["cues"] = np.where(rows.X[:, cue_cols].sum(1) > 0, 0.95, 0.05).astype(np.float32)
    jobs = {
        "description": lambda: cross_validate(
            dict(BASE, drop=["meaning", "seam", "cues", "position"], use_description=True), rows, seed=0),
        "structure": lambda: cross_validate(dict(BASE, drop=["meaning"]), rows, seed=0),
        "meaning": lambda: cross_validate(dict(BASE, drop=["seam", "cues", "position"]), rows, seed=0),
        "potion": lambda: cv_linear(potion_rows(rows), rows),
    }
    for name, job in jobs.items():
        if name not in cached:
            print(f"  scoring {name} out of fold...", flush=True)
            cached[name] = job()
            np.savez(CACHE, **cached)
        streams[name] = cached[name]
    streams["marker"] = level1
    best, best_cov = None, -1.0
    for variant in ("h32_r7", "h64_r7", "h64_r15"):   # the saved sequence runs; keep the one the rule likes best
        s = np.load(DATA / f"sequence_oof_{variant}.npy")
        b = choose(sweep(s, rows), rows, 10)
        if b and b[2]["coverage"] > best_cov:
            best, best_cov = (variant, s), b[2]["coverage"]
    print(f"  sequence model: variant {best[0]} (the rule's best at B=10)")
    streams["sequence"] = best[1]
    return streams, level2


def thresholds(scores: np.ndarray) -> np.ndarray:
    return np.unique(np.quantile(scores, 1 - SHARES))


def sweep(scores: np.ndarray, rows, p_start=None, p_end=None, filt=None) -> list:
    """(threshold, regions) at every threshold: own edges, or the edge heads when given."""
    out = []
    for th in thresholds(scores):
        kept = regions_by_video(scores, rows, float(th), 1)
        if p_start is not None:
            kept = place(kept, rows, p_start, p_end)
        if filt is not None:
            kept = filt(kept)
        out.append((float(th), kept))
    return out


def graded(candidates: list, rows) -> list:
    """Grade every candidate once: (threshold, regions, grade, share of videos over 60 s)."""
    return [(th, kept, grade_regions(kept, rows), over_cap(kept, rows)) for th, kept in candidates]


def pick(graded_list: list, budget: float):
    best = None
    for th, kept, g, oc in graded_list:
        if g["show"] <= budget and oc <= OVER_CAP_SHARE and (best is None or g["coverage"] > best[2]["coverage"]):
            best = (th, kept, g)
    return best


def choose(candidates: list, rows, budget: float):
    return pick(graded(candidates, rows), budget)


def line(name: str, best) -> str:
    if best is None:
        return f"  {name:<46} nothing meets the rule"
    g = best[2]
    return (f"  {name:<46} ad time {g['coverage']:6.1%}   show lost {g['show']:4.1f} s   "
            f"reads touched {g['found']:3d}/{g['total']}")


def false_regions(kept: dict, rows) -> set:
    """(video, first line) of every region that overlaps no real read."""
    out = set()
    for vid, spans in kept.items():
        y = rows.y[rows.video == vid]
        out |= {(vid, lo) for lo, hi in spans if not y[lo:hi].any()}
    return out


def false_videos(kept: dict, rows) -> set:
    return {v for v, _ in false_regions(kept, rows)}


def touched(kept: dict, rows) -> np.ndarray:
    return np.array([share > 0 for _, _, _, share in per_read(kept, rows)])


def main() -> int:
    rows, is_start, is_resume = pooled()
    print(f"pooled: {rows.videos} videos, {rows.reads} reads\n")
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    streams, _ = level_one(rows)

    # ---------------------------------------------------------------- how different are they?
    names = list(streams)
    Z = np.column_stack([logit(streams[n]) for n in names])
    C = np.corrcoef(Z.T)
    print("HOW ALIKE (correlation of the detectors' line scores; 1.00 = the same detector):")
    print("  " + " " * 12 + "".join(f"{n[:9]:>10}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:<12}" + "".join(f"{C[i, j]:10.2f}" for j in range(len(names))))

    # ---------------------------------------------------------------- each detector through the pipeline
    print("\nEACH DETECTOR ALONE, through its own context model + the edge heads, pooled rule:")
    second = {}
    chosen = {}
    for name, s in streams.items():
        key = f"ctx_{name}"
        cached = dict(np.load(CACHE))
        if name == "marker":
            l2 = level2
        elif key in cached:
            l2 = cached[key]
        else:
            print(f"  training the context model on {name}...", flush=True)
            l2 = out_of_fold(rows, context_features(rows, s), hidden=32)
            cached[key] = l2
            np.savez(CACHE, **cached)
        second[name] = l2
        raw = graded(sweep(s, rows), rows)
        full = graded(sweep(l2, rows, p_start, p_end), rows)
        for budget in BUDGETS:
            b_raw, b_full = pick(raw, budget), pick(full, budget)
            print(line(f"B={budget:>2} {name}: raw scores, own edges", b_raw))
            print(line(f"B={budget:>2} {name}: + context model + heads", b_full))
            if budget == 10:
                chosen[name] = b_full or b_raw

    # ---------------------------------------------------------------- do they find DIFFERENT reads?
    print("\nDIFFERENT READS, DIFFERENT FALSE ALARMS (each at its own B=10 choice, against the marker's free tier):")
    base = chosen["marker"][1]
    t_base, f_base = touched(base, rows), false_videos(base, rows)
    print(f"  marker free tier: touches {t_base.sum()} reads, false alarms in {len(f_base)} videos")
    for name, b in chosen.items():
        if name == "marker" or b is None:
            continue
        t, f = touched(b[1], rows), false_videos(b[1], rows)
        print(f"  {name:<12} touches {t.sum():3d}; finds {np.sum(t & ~t_base):3d} the marker misses, "
              f"misses {np.sum(t_base & ~t):3d} it finds; false alarms in {len(f):2d} videos, "
              f"{len(f & f_base):2d} shared with the marker")

    # ---------------------------------------------------------------- combinations
    print("\nCOMBINED, pooled rule:")
    everything = [b[1] for b in chosen.values() if b is not None]
    union = {v: merged([s for k in everything for s in k.get(v, [])]) for v in set().union(*everything)}
    g = grade_regions(union, rows)
    print(f"  {'union of every detector at its own choice':<46} ad time {g['coverage']:6.1%}   show lost "
          f"{g['show']:4.1f} s   over 60 s: {over_cap(union, rows):.1%} of videos (the rule allows 2%)")
    level_one_names = ["marker", "meaning", "structure", "potion", "sequence"]

    cached = dict(np.load(CACHE))
    if "ctx_average" not in cached:
        print("  training the context model on the averaged logits...", flush=True)
        avg = 1 / (1 + np.exp(-np.mean([logit(streams[n]) for n in level_one_names], axis=0)))
        cached["ctx_average"] = out_of_fold(rows, context_features(rows, avg), hidden=32)
        np.savez(CACHE, **cached)
    g_avg = graded(sweep(cached["ctx_average"], rows, p_start, p_end), rows)
    for budget in BUDGETS:
        print(line(f"B={budget:>2} average of 5 detectors -> context -> heads", pick(g_avg, budget)))

    names_all = [str(n) for n in rows.feature_names]
    extra = [i for i, n in enumerate(names_all) if n.startswith("cue_") or n.startswith("desc_")]
    F_all = np.column_stack([context_features(rows, streams[n]) for n in level_one_names] + [rows.X[:, extra]])
    if "ctx_stack" not in cached:
        print(f"  training ONE context model on every detector ({F_all.shape[1]} inputs)...", flush=True)
        cached["ctx_stack"] = out_of_fold(rows, F_all, hidden=32)
        np.savez(CACHE, **cached)
    if "stack_start" not in cached:
        print("  training edge heads on the stacked inputs...", flush=True)
        cached["stack_start"] = edge_oof(rows, F_all, soft(is_start, rows.video))
        cached["stack_end"] = edge_oof(rows, F_all, soft(is_resume, rows.video))
        np.savez(CACHE, **cached)
    stack = cached["ctx_stack"]
    g_sh = graded(sweep(stack, rows, p_start, p_end), rows)
    g_ss = graded(sweep(stack, rows, cached["stack_start"], cached["stack_end"]), rows)
    for budget in BUDGETS:
        print(line(f"B={budget:>2} stacked context -> shipped heads", pick(g_sh, budget)))
        print(line(f"B={budget:>2} stacked context -> stacked heads", pick(g_ss, budget)))

    print("\n  AGREEMENT: the marker's free tier, a region skipped only if a second detector's context model")
    print("  also reaches its own threshold somewhere inside it (both thresholds chosen by the rule):")
    placed = sweep(level2, rows, p_start, p_end)   # the shipped free tier at every threshold, placed once
    for other in ["potion", "sequence", "meaning", "structure", "description", "cues"]:
        s2 = second[other]
        pool = []
        for t2 in np.unique(np.quantile(s2, 1 - np.geomspace(0.01, 0.3, 10))):
            for th, kept in placed:
                out = {}
                for vid, spans in kept.items():
                    r = np.flatnonzero(rows.video == vid)
                    keep = [(lo, hi) for lo, hi in spans if s2[r][lo:hi].max() >= t2]
                    if keep:
                        out[vid] = keep
                pool.append((th, out))
        g_pool = graded(pool, rows)
        for budget in BUDGETS:
            print(line(f"B={budget:>2} marker tier, {other} must agree", pick(g_pool, budget)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
