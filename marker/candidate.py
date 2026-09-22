"""The candidate system, trained on ALL pooled videos and applied to videos it has never seen. (2026-09-22)

Everything so far was out of fold on the pooled 205 videos, and every system was chosen on those same
folds (the context-and-stacking room's caveat). This builds the candidate for a fresh set:

  level 1   the five bake-off detectors refit on all pooled rows (frozen marker, meaning-only,
            structure-only, potion marker, conv sequence model) and scored on the fresh rows, plus
            fine-tuned BGE-small trained on all pooled rows (finetune_minilm.py --score, on the GPU)
  level 2   the stack context model, trained on the pooled OUT-OF-FOLD level-1 scores (standard
            stacking), applied to the fresh set's full-data level-1 scores
  edges     the start/resume heads trained on pooled rows, applied the same way
  rule      the thresholds the pooled CV rule chose for this system at B = 5 and B = 10, fixed in
            advance (never chosen on the fresh set)
  chapters  creator chapters titled for a read become regions / edges (watch_meta.json), if fetched

It grades against the fresh set's labels: ad time skipped, show lost per video, share of videos over
60 s, false-alarm regions, and English videos separately.

    python marker/candidate.py --fresh features_channels.npz --bge data/finetune_full_bge_seed0__channels.npy
    python marker/candidate.py --dry-run 4      # mechanics check: pooled fold 4 plays "fresh"
"""

import argparse
import ctypes
import json
import re
import sys

import numpy as np
import torch

from build_production import OVER_CAP_SHARE, over_cap, pooled
from context_stack import context_features, fit_stage2
from detector_bakeoff import CACHE, graded, line, pick
from edge_heads import merged, place, soft
from experiments import BASE, fit, fit_full, runs
from replay import grade_regions, per_read, regions_by_video
from sequence_model import train_and_score
from stack_check import sweep_fine
from train import DATA, LAST_LINE_SECONDS, Rows, load, scaling

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(8)
POTION = {"features.npz": "features_potion_cv.npz", "features_tail.npz": "features_potion_tail.npz",
          "features_holdout2.npz": "features_potion_holdout2.npz", "features_channels.npz": "features_potion_channels.npz",
          "features_holdout4.npz": "features_potion_holdout4.npz", "features_negatives.npz": "features_potion_negatives.npz"}
PROMO_TITLE = re.compile(r"\b(sponsor\w*|ad|ads|advert\w*|promo\w*|brought to you|partner\w*|thanks to|merch|patreon)\b", re.I)
SEQ = dict(hidden=32, reach=7, epochs=12)


def sub(rows, m) -> Rows:
    return Rows(rows.X[m], rows.y[m], rows.video[m], rows.channel[m], rows.start_seconds[m], rows.split[m], rows.feature_names)


def cat(a, b) -> Rows:
    return Rows(np.concatenate([a.X, b.X]), np.concatenate([a.y, b.y]), np.concatenate([a.video, b.video]),
                np.concatenate([a.channel, b.channel]), np.concatenate([a.start_seconds, b.start_seconds]),
                np.concatenate([a.split, b.split]), a.feature_names)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def level1_fresh(P: Rows, T: Rows, potP: np.ndarray, potT: np.ndarray) -> dict:
    """Each level-1 detector refit on P and scored on T."""
    out = {}
    for name, cfg in (("marker", BASE), ("meaning", dict(BASE, drop=["seam", "cues", "position"])),
                      ("structure", dict(BASE, drop=["meaning"]))):
        parts = fit_full(cfg, P, seed=0)
        with torch.no_grad():
            z = sum(p["model"](torch.from_numpy((T.X[:, p["columns"]] - p["mean"]) / p["std"]).float()).squeeze(1).numpy()
                    for p in parts) / len(parts)
        out[name] = sigmoid(z)
    mean, std = scaling(potP)
    m = fit(BASE, (potP - mean) / std, P.y, 0)
    with torch.no_grad():
        out["potion"] = sigmoid(m(torch.from_numpy((potT - mean) / std).float()).squeeze(1).numpy())
    C = cat(P, T)
    C395 = Rows(C.X[:, :395], C.y, C.video, C.channel, C.start_seconds, C.split, C.feature_names[:395])
    tr = np.zeros(len(C), bool)
    tr[:len(P)] = True
    out["sequence"] = train_and_score(C395, tr, ~tr, SEQ["hidden"], SEQ["reach"], SEQ["epochs"])[len(P):]
    return out


def chapter_regions(kept: dict, T: Rows, meta: dict) -> dict:
    out = dict(kept)
    for v in np.unique(T.video):
        v = str(v)
        m = meta.get(v)
        if not m or not m.get("chapters"):
            continue
        r = np.flatnonzero(T.video == v)
        s = T.start_seconds[r]
        dur = float(s[-1] + LAST_LINE_SECONDS)
        ch = sorted(m["chapters"], key=lambda c: c["start"])
        spans = list(out.get(v, []))
        for k, c in enumerate(ch):
            if not PROMO_TITLE.search(c["title"]):
                continue
            ce = ch[k + 1]["start"] if k + 1 < len(ch) else dur
            lo, hi = int(np.searchsorted(s, c["start"])), int(np.searchsorted(s, ce))
            if hi > lo:
                spans = [sp for sp in spans if not (sp[0] < hi and lo < sp[1])] + [(lo, hi)]
        if spans:
            out[v] = merged(spans)
    return out


def report(name: str, kept: dict, T: Rows, lang: dict | None = None) -> None:
    if T.y.sum() == 0:   # a sponsor-free set: every skipped second is lost show
        lost = []
        for v in np.unique(T.video):
            r = np.flatnonzero(T.video == v)
            s = T.start_seconds[r]
            sec = np.diff(np.append(s, s[-1] + LAST_LINE_SECONDS))
            lost.append(sum(float(sec[lo:hi].sum()) for lo, hi in kept.get(str(v), [])))
        x = np.array(lost)
        print(f"  {name:<44} no skip {np.mean(x == 0):5.0%}  mean {x.mean():4.1f} s  >10 s {np.mean(x > 10):4.0%}  "
              f">60 s {np.mean(x > 60):4.0%}  ({len(x)} sponsor-free videos)", flush=True)
        return
    g = grade_regions(kept, T)
    idx = {str(v): np.flatnonzero(T.video == v) for v in np.unique(T.video)}
    false = sum(1 for v, spans in kept.items() for lo, hi in spans if not T.y[idx[v]][lo:hi].any())
    pr = np.array([s for *_, s in per_read(kept, T)])
    print(f"  {name:<44} ad time {g['coverage']:6.1%}  show lost {g['show']:4.1f} s/video  over 60 s "
          f"{over_cap(kept, T):4.1%}  false regions {false:3d}  reads: full {np.mean(pr >= .95):.0%} / "
          f"partial {np.mean((pr > 0) & (pr < .95)):.0%} / missed {np.mean(pr == 0):.0%}", flush=True)
    if lang:
        en = np.array([lang.get(str(v), {}).get("lang") == "en" for v in T.video])
        if 0 < en.sum() < len(en):
            E = sub(T, en)
            ke = {v: s for v, s in kept.items() if lang.get(v, {}).get("lang") == "en"}
            ge = grade_regions(ke, E)
            print(f"  {'   English videos only':<44} ad time {ge['coverage']:6.1%}  show lost {ge['show']:4.1f} s/video", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", help="a feature file (in marker/data) none of the pooled sets contain")
    ap.add_argument("--bge", help="fine-tuned BGE scores for the fresh rows (finetune_minilm.py --score)")
    ap.add_argument("--dry-run", type=int, help="pooled fold k plays the fresh set (mechanics check only)")
    ap.add_argument("--edges", nargs=2, metavar=("START", "RESUME"),
                    help="candidate v2: fine-tuned start/resume scores for the fresh rows, averaged with the MLP heads")
    args = ap.parse_args()

    rows, is_start, is_resume = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    bake = dict(np.load(CACHE))
    ft = dict(np.load(DATA / "finetune_streams.npz"))
    bge_oof = np.load(DATA / "finetune_oof_bge_seed0.npy")
    pot_all = np.concatenate([np.load(DATA / POTION[f])["X"] for f in ("features.npz", "features_tail.npz", "features_holdout2.npz")])
    pot_all = pot_all[:, :pot_all.shape[1] - 2]
    streams_oof = {"marker": level1, "meaning": bake["meaning"], "structure": bake["structure"], "potion": bake["potion"],
                   "sequence": np.load(DATA / "sequence_oof_h32_r7.npy"), "bge": bge_oof}
    names = [str(n) for n in rows.feature_names]
    extra_cols = [i for i, n in enumerate(names) if n.startswith("cue_") or n.startswith("desc_")]

    # the thresholds, fixed by the pooled CV rule on this system's out-of-fold scores
    g_cv = graded(sweep_fine(ft["stack_bge_seed0"], rows, p_start, p_end), rows)
    th = {b: pick(g_cv, b)[0] for b in (5, 10)}
    print(f"thresholds fixed on pooled CV: B=5 {th[5]:.4f}, B=10 {th[10]:.4f}")
    fs_oof = np.load(DATA / "finetune_oof_edge_start_seed0.npy")
    fe_oof = np.load(DATA / "finetune_oof_edge_resume_seed0.npy")
    g_v2 = graded(sweep_fine(ft["stack_bge_seed0"], rows, (p_start + fs_oof) / 2, (p_end + fe_oof) / 2), rows)
    th2 = {b: pick(g_v2, b)[0] for b in (5, 10)}
    print(f"candidate v2 thresholds fixed on pooled CV: B=5 {th2[5]:.4f}, B=10 {th2[10]:.4f}")

    if args.dry_run is not None:
        channels = np.array(sorted(set(rows.channel)))
        np.random.default_rng(0).shuffle(channels)
        test = np.isin(rows.channel, np.array_split(channels, 5)[args.dry_run])
        P, T = sub(rows, ~test), sub(rows, test)
        potP, potT = pot_all[~test], pot_all[test]
        sP = {k: v[~test] for k, v in streams_oof.items()}
        bge_T = bge_oof[test]
        isP, irP = is_start[~test], is_resume[~test]
        fs_T, fe_T = fs_oof[test], fe_oof[test]
        ref = {b: pick(g_cv, b)[1] for b in (5, 10)}
        print(f"DRY RUN: pooled fold {args.dry_run} plays fresh ({T.videos} videos, {T.reads} reads); "
              f"its out-of-fold stack at the same thresholds, for reference:")
        for b in (5, 10):
            report(f"B={b:>2} out-of-fold stack (reference)", {v: s for v, s in ref[b].items() if v in set(map(str, T.video))}, T)
        meta, lang = {}, None
    else:
        T = load(DATA / args.fresh)
        potT = np.load(DATA / POTION[args.fresh])["X"]
        potT = potT[:, :potT.shape[1] - 2]
        bge_T = np.load(args.bge)
        assert len(bge_T) == len(T) and len(potT) == len(T)
        P, potP, sP = rows, pot_all, streams_oof
        isP, irP = is_start, is_resume
        fs_T, fe_T = (np.load(args.edges[0]), np.load(args.edges[1])) if args.edges else (None, None)
        meta = json.loads((DATA / "watch_meta.json").read_text(encoding="utf-8")) if (DATA / "watch_meta.json").exists() else {}
        lang_path = DATA / "video_language.json"
        lang = json.loads(lang_path.read_text(encoding="utf-8")) if lang_path.exists() else None
        if lang is not None:   # sets crawled after video_language.json: fall back to the caption file's language
            for v in np.unique(T.video):
                lang.setdefault(str(v), {"lang": None})
        print(f"FRESH SET {args.fresh}: {T.videos} videos, {T.reads} reads, "
              f"{len(set(T.channel))} channels; none of them in the pooled training data")
        assert not set(T.channel) & set(P.channel), "fresh set shares channels with the pooled data"

    fresh = level1_fresh(P, T, potP, potT)
    fresh["bge"] = bge_T
    order = ["marker", "meaning", "structure", "potion", "sequence", "bge"]
    FP = np.column_stack([context_features(P, sP[k]) for k in order] + [P.X[:, extra_cols]])
    FT = np.column_stack([context_features(T, fresh[k]) for k in order] + [T.X[:, extra_cols]])
    ctx = fit_stage2(FP, P.y.astype(np.float32), hidden=32, seed=0)(FT)
    EP, ET = context_features(P, sP["marker"]), context_features(T, fresh["marker"])
    ps_T = fit_stage2(EP, soft(isP, P.video), hidden=32)(ET)
    pe_T = fit_stage2(EP, soft(irP, P.video), hidden=32)(ET)

    print()
    for b in (5, 10):
        kept = place(regions_by_video(ctx, T, th[b], 1), T, ps_T, pe_T)
        report(f"B={b:>2} candidate (stack + fine-tuned BGE)", kept, T, lang)
        if meta:
            report(f"B={b:>2}   + creator chapters", chapter_regions(kept, T, meta), T, lang)
    if fs_T is not None:
        print()
        for b in (5, 10):
            kept = place(regions_by_video(ctx, T, th2[b], 1), T, (ps_T + fs_T) / 2, (pe_T + fe_T) / 2)
            report(f"B={b:>2} candidate v2 (averaged edge heads)", kept, T, lang)
            if meta:
                report(f"B={b:>2}   + creator chapters", chapter_regions(kept, T, meta), T, lang)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
