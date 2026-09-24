"""Enlarging the stack's training pool (2026-09-24): the three comparisons Simon asked for.

candidate v3's context model (stack_sbml.py) was trained on only the 160 pooled videos whose
SponsorBlock labels postdate the community model's training, because every OTHER input tried since
has overfit that small a pool. This enlarges the pool with every set that has already been graded or
used as a tripwire (PREREGISTRATION.md results 1-4), so none of it is a live holdout any more:
features_holdout3.npz (10), features_channels.npz (169), features_holdout4.npz (64),
features_holdout5.npz (46) -- 494 videos, 326 channels total, verified channel-disjoint, and
sb_dates.json / sbml_predictions.jsonl now cover all of them (enlarge_pool.py, sb_dates.py,
sponsorblock_ml.py). 434 of the 494 are stack-eligible (post-2022-04 SponsorBlock labels): 160 from
the original pool + 274 newly added.

Every stream feeding the stack is out-of-fold with channel-grouped folds over the ENLARGED pool
(enlarge_pool.py: the linear marker, meaning/structure/potion, the sequence model;
finetune_minilm.py --extra-sets ...: fine-tuned BGE-small inside/start/resume). Nothing here is
graded on a holdout that has not already been used; holdout 6 (not yet built) is the fresh test.

Comparison 1 (like-for-like): both systems use the SAME level-1 detector streams (the enlarged pool's
honest OOF, computed once) and are graded on the SAME videos -- the 274 newly added stack-eligible
ones -- so the only thing that differs is how much data trained the STACKING step itself:
  enlarged   the context model out-of-fold over all 434 stack-eligible videos (5-fold, this run's
             own channel shuffle), graded on the 274 new ones' OOF scores
  orig-160   the context model fit ONCE on the original 160 (never touching the 274), applied to
             them directly -- honest because their channels never appear in the 160

Comparison 2: the fine-tuned Claude-category detector (enlarge_claude.py) as an 8th input, and qwen's
recorded answers (qwen_sweep.jsonl covers only the original 205 -- no fresh sweep was run on the 274
new videos, so that reinput is SKIPPED and reported, not faked).

Comparison 3: hidden=0 (linear) vs hidden=32 for the stacking step, and stronger weight decay/dropout
(context_stack.fit_stage2 now takes both), on the enlarged pool.

    python marker/enlarge_stack.py
"""

import argparse
import datetime
import json
import sys

import numpy as np
import torch

from build_production import ENLARGED_SETS, SETS as POOLED_SETS, pooled
from context_stack import context_features, fit_stage2
from detector_bakeoff import graded, line, pick
from enlarge_pool import BAKEOFF_CACHE, POOLED_OOF
from sbml_eval import load_preds
from stack_check import oof_seeded, sweep_fine
from stack_sbml import sbml_stream
from train import DATA

sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(6)

CUTOFF = datetime.datetime(2022, 4, 1).timestamp() * 1000
ORDER = ["marker", "meaning", "structure", "potion", "sequence", "bge"]
SEEDS = (0, 1, 2)


def masks(rows):
    d = json.load(open(DATA / "sb_dates.json", encoding="utf-8"))
    unseen = np.array([not (d.get(str(v), {}).get("first") and d[str(v)]["first"] < CUTOFF) for v in rows.video])
    orig_videos = set()
    for f in POOLED_SETS:
        orig_videos |= {str(v) for v in np.unique(np.load(DATA / f)["video"])}
    orig = np.array([str(v) in orig_videos for v in rows.video])
    return unseen, orig


def load_streams(rows):
    level1, _, p_start, p_end = np.load(POOLED_OOF)
    bake = dict(np.load(BAKEOFF_CACHE))
    sequence = np.load(DATA / "sequence_oof_h32_r7_enlarged.npy")
    bge_path = DATA / "finetune_oof_bge_enlarged_seed0.npy"
    fs_path = DATA / "finetune_oof_edge_start_enlarged_seed0.npy"
    fe_path = DATA / "finetune_oof_edge_resume_enlarged_seed0.npy"
    missing = [p.name for p in (bge_path, fs_path, fe_path) if not p.exists()]
    if missing:
        raise SystemExit(f"missing fine-tuned enlarged-pool streams: {missing}; run finetune_minilm.py "
                         f"--model BAAI/bge-small-en-v1.5 --extra-sets features_holdout3.npz "
                         f"features_channels.npz features_holdout4.npz features_holdout5.npz "
                         f"--target inside|start|resume --tag bge_enlarged|edge_start_enlarged|edge_resume_enlarged first")
    bge, fs, fe = np.load(bge_path), np.load(fs_path), np.load(fe_path)
    streams = {"marker": level1, "meaning": bake["meaning"], "structure": bake["structure"],
              "potion": bake["potion"], "sequence": sequence, "bge": bge}
    sb = sbml_stream(rows, load_preds())
    heads_start, heads_end = (p_start + fs) / 2, (p_end + fe) / 2
    names = [str(n) for n in rows.feature_names]
    extra_idx = [i for i, n in enumerate(names) if n.startswith("cue_") or n.startswith("desc_")]
    return streams, sb, rows.X[:, extra_idx], heads_start, heads_end


def build_F(rows, mask, streams, sb, extra, claude=None):
    S = rows.subset(mask)
    parts = [context_features(S, streams[k][mask]) for k in ORDER]
    parts.append(context_features(S, sb[mask]))
    if claude is not None:
        parts.append(context_features(S, claude[mask]))
    parts.append(extra[mask])
    return S, np.column_stack(parts)


def comparison1(rows, streams, sb, extra, heads_start, heads_end, unseen, orig, new_unseen) -> dict:
    print("=" * 100)
    print(f"COMPARISON 1: v3 recipe, enlarged pool ({int(unseen.sum())} stack-eligible videos) vs the "
          f"original 160 -- graded on the {int(new_unseen.sum())} newly added stack-eligible videos")
    print("=" * 100)
    train160 = unseen & orig
    S_full, F_full = build_F(rows, unseen, streams, sb, extra)
    hs_full, he_full = heads_start[unseen], heads_end[unseen]
    within_new = new_unseen[unseen]
    S_new = S_full.subset(within_new)
    hs_new, he_new = hs_full[within_new], he_full[within_new]
    F_new = F_full[within_new]
    S_160, F_160 = build_F(rows, train160, streams, sb, extra)
    print(f"  ({S_new.videos} videos, {S_new.reads} reads graded; {S_160.videos} videos trained the orig-160 system)\n")

    out = {"enlarged": {}, "orig160": {}}
    for seed in SEEDS:
        oof_enl = oof_seeded(S_full, F_full, seed)[within_new]
        g_enl = graded(sweep_fine(oof_enl, S_new, hs_new, he_new), S_new)
        out["enlarged"][seed] = g_enl
        model = fit_stage2(F_160, S_160.y.astype(np.float32), hidden=32, seed=seed)
        probs_160 = model(F_new)
        g_160 = graded(sweep_fine(probs_160, S_new, hs_new, he_new), S_new)
        out["orig160"][seed] = g_160
        for b in (5, 10):
            print(line(f"seed{seed} B={b:>2} enlarged (434, retrained)", pick(g_enl, b)))
            print(line(f"seed{seed} B={b:>2} orig-160 (fit once, scored fresh)", pick(g_160, b)))
        print()
    return out


def comparison2(rows, streams, sb, extra, heads_start, heads_end, unseen) -> None:
    print("=" * 100)
    print("COMPARISON 2: retest the failed inputs as additions, on the enlarged pool")
    print("=" * 100)
    S_full, F_full = build_F(rows, unseen, streams, sb, extra)
    hs_full, he_full = heads_start[unseen], heads_end[unseen]
    print(f"  baseline: v3 (7 inputs) over all {S_full.videos} stack-eligible videos\n")
    baseline = {}
    for seed in SEEDS:
        g = graded(sweep_fine(oof_seeded(S_full, F_full, seed), S_full, hs_full, he_full), S_full)
        baseline[seed] = g
        for b in (5, 10):
            print(line(f"seed{seed} B={b:>2} v3 (7 inputs), enlarged pool", pick(g, b)))
    print()

    claude_path = DATA / "claude_nonsponsor_stream_enlarged.npy"
    if claude_path.exists():
        claude = np.load(claude_path)
        S_c, F_c = build_F(rows, unseen, streams, sb, extra, claude=claude)
        print(f"  + fine-tuned Claude-category detector (8th input; coverage: 251/494 videos have true "
              f"Claude labels, the rest score from a model trained on those 251 -- see enlarge_claude.py)\n")
        for seed in SEEDS:
            g = graded(sweep_fine(oof_seeded(S_c, F_c, seed), S_c, hs_full, he_full), S_c)
            for b in (5, 10):
                print(line(f"seed{seed} B={b:>2} v3 + Claude-category detector", pick(g, b)))
    else:
        print(f"  SKIPPED: {claude_path.name} not built (run enlarge_claude.py after its finetune_minilm.py jobs)")

    print(f"\n  qwen's recorded answers: SKIPPED. qwen_sweep.jsonl covers only the original 205 pooled videos; "
          f"no fresh qwen sweep exists for the 274 newly added videos, and running one is out of scope here. "
          f"Adding it honestly would need qwen3:8b answers recorded for those videos first.")


def comparison3(rows, streams, sb, extra, heads_start, heads_end, unseen, which=None) -> None:
    print("=" * 100)
    print("COMPARISON 3: simpler / more regularised stacking step, on the enlarged pool")
    print("=" * 100)
    S_full, F_full = build_F(rows, unseen, streams, sb, extra)
    hs_full, he_full = heads_start[unseen], heads_end[unseen]
    variants = [
        ("hidden=32, wd=0.01, dropout=0.2 (recipe of record)", dict(hidden=32)),
        ("hidden=0 (linear)", dict(hidden=0)),
        ("hidden=32, wd=0.1", dict(hidden=32, weight_decay=0.1)),
        ("hidden=32, wd=0.3", dict(hidden=32, weight_decay=0.3)),
        ("hidden=32, dropout=0.5", dict(hidden=32, dropout=0.5)),
    ]
    for i, (label, kwargs) in enumerate(variants):
        if which is not None and i not in which:
            continue
        for seed in SEEDS:
            oof = oof_seeded(S_full, F_full, seed, **kwargs)
            g = graded(sweep_fine(oof, S_full, hs_full, he_full), S_full)
            for b in (5, 10):
                print(line(f"seed{seed} B={b:>2} {label}", pick(g, b)))
        print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="+", default=[1, 2, 3], help="which comparisons to run")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--variants", type=int, nargs="+", help="comparison 3 variant indices (0 = recipe of record)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rows = pooled(ENLARGED_SETS)
    if isinstance(rows, tuple):
        rows = rows[0]
    print(f"enlarged pool: {rows.videos} videos, {len(set(rows.channel))} channels, {rows.reads} reads")
    unseen, orig = masks(rows)
    new_unseen = unseen & ~orig
    n_videos = lambda m: len(np.unique(rows.video[m]))
    print(f"stack-eligible (post-2022-04): {n_videos(unseen)} of {rows.videos} videos "
         f"({n_videos(unseen & orig)} original + {n_videos(new_unseen)} newly added)\n")
    streams, sb, extra, heads_start, heads_end = load_streams(rows)

    if 1 in args.only:
        comparison1(rows, streams, sb, extra, heads_start, heads_end, unseen, orig, new_unseen)
    if 2 in args.only:
        comparison2(rows, streams, sb, extra, heads_start, heads_end, unseen)
    if 3 in args.only:
        comparison3(rows, streams, sb, extra, heads_start, heads_end, unseen, args.variants)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
