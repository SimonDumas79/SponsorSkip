"""Save the graded candidate as ONE bundle predict.py can serve. (2026-09-22)

Everything measured since 2026-09-21 lived only in evaluation code: candidate.py refits the detectors
on all pooled rows, scores a fresh feature file and prints a table. Nothing was ever written to disk
that could read a NEW video. So the extension still serves free_tier.pt, the system from before the
fine-tuned detector and the stack, and the gap is not small -- on holdout 4, the same 64 fresh videos:

    free_tier.pt (shipped)            40.3% of ad time, 10.8 s of show lost per video, worst 177 s
    candidate v3 B=10 + chapters      67.2% of ad time,  5.5 s of show lost per video, none over 60 s

This script closes that gap. It trains every part on ALL pooled videos (the same fit candidate.py
does, same seeds) and saves the weights:

    marker, meaning, structure   linear detectors over the 395 MiniLM features (experiments.fit_full)
    potion                       a linear detector over the potion-base-8M embedding
    sequence                     the 1-D conv over a whole video (sequence_model)
    bge                          the fine-tuned BGE-small checkpoint from finetune_minilm.py
                                 --train-only (the GPU part; run marker/data/export_chain.sh first)
    context                      the stack's context model over all six, trained on the pooled
                                 OUT-OF-FOLD scores, which is what stacking requires
    start, end                   the MLP edge heads, and the fine-tuned edge checkpoints they are
                                 averaged with (candidate v2)
    thresholds                   B = 5 and B = 10, fixed by the pooled CV rule, never re-chosen here

    python marker/export_candidate.py                       write data/production/candidate.pt
    python marker/export_candidate.py --verify holdout4     score holdout 4 THROUGH the bundle and
                                                            check it reproduces the recorded numbers

--verify is the part that matters: a bundle that does not reproduce the grade is not the system that
was graded, and shipping it would quietly throw the measurement away.
"""

import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np
import torch

from build_production import pooled
from candidate import POTION, SEQ, report
from context_stack import context_features, fit_stage2
from detector_bakeoff import CACHE, graded, pick
from edge_heads import place, soft
from experiments import BASE, fit, fit_full
from predict import BUNDLE, stage_state
from replay import regions_by_video
from sequence_model import train_and_score
from stack_check import sweep_fine
from train import DATA, Rows, load, scaling

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(8)

OUT = BUNDLE.parent / "candidate.pt"
MODELS = BUNDLE.parent / "models"
ORDER = ["marker", "meaning", "structure", "potion", "sequence", "bge"]
CONFIGS = {"marker": BASE, "meaning": dict(BASE, drop=["seam", "cues", "position"]), "structure": dict(BASE, drop=["meaning"])}


def potion_pooled() -> np.ndarray:
    pot = np.concatenate([np.load(DATA / POTION[f])["X"]
                          for f in ("features.npz", "features_tail.npz", "features_holdout2.npz")])
    return pot[:, :pot.shape[1] - 2]


def linear_state(parts) -> list[dict]:
    return [{"columns": torch.from_numpy(p["columns"]), "mean": torch.from_numpy(p["mean"]),
             "std": torch.from_numpy(p["std"]), "state": p["model"].state_dict()} for p in parts]


def build() -> int:
    rows, is_start, is_resume = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    bake = dict(np.load(CACHE))
    ft = dict(np.load(DATA / "finetune_streams.npz"))
    bge_oof = np.load(DATA / "finetune_oof_bge_seed0.npy")
    fs_oof = np.load(DATA / "finetune_oof_edge_start_seed0.npy")
    fe_oof = np.load(DATA / "finetune_oof_edge_resume_seed0.npy")
    pot = potion_pooled()
    names = [str(n) for n in rows.feature_names]
    extra_cols = [i for i, n in enumerate(names) if n.startswith("cue_") or n.startswith("desc_")]
    print(f"pooled: {rows.videos} videos, {rows.reads} reads, {len(rows.y):,} lines", flush=True)

    # The thresholds, fixed by the pooled CV rule on the out-of-fold scores. Copied, never re-chosen.
    g_cv = graded(sweep_fine(ft["stack_bge_seed0"], rows, p_start, p_end), rows)
    g_v2 = graded(sweep_fine(ft["stack_bge_seed0"], rows, (p_start + fs_oof) / 2, (p_end + fe_oof) / 2), rows)
    th = {b: float(pick(g_cv, b)[0]) for b in (5, 10)}
    th2 = {b: float(pick(g_v2, b)[0]) for b in (5, 10)}
    print(f"thresholds from pooled CV: v1 B=5 {th[5]:.4f} B=10 {th[10]:.4f}; "
          f"v2 B=5 {th2[5]:.4f} B=10 {th2[10]:.4f}", flush=True)

    detectors = {}
    for name, cfg in CONFIGS.items():
        print(f"  fitting {name} on all pooled rows", flush=True)
        detectors[name] = linear_state(fit_full(cfg, rows, seed=0))
    print("  fitting potion", flush=True)
    pmean, pstd = scaling(pot)
    pmodel = fit(BASE, (pot - pmean) / pstd, rows.y, 0)
    print(f"  fitting the sequence model ({SEQ})", flush=True)
    r395 = Rows(rows.X[:, :395], rows.y, rows.video, rows.channel, rows.start_seconds, rows.split,
                rows.feature_names[:395])
    allm = np.ones(len(rows.y), bool)
    _, seq_model, smean, sstd = train_and_score(r395, allm, allm, SEQ["hidden"], SEQ["reach"], SEQ["epochs"],
                                                return_model=True)

    # The stack's context model is trained on the pooled OUT-OF-FOLD level-1 scores. Training it on
    # in-fold scores instead would hand it detectors that have already seen the answer, and it would
    # learn to trust them far more than it should on a video it has never seen.
    streams_oof = {"marker": level1, "meaning": bake["meaning"], "structure": bake["structure"],
                   "potion": bake["potion"], "sequence": np.load(DATA / "sequence_oof_h32_r7.npy"), "bge": bge_oof}
    FP = np.column_stack([context_features(rows, streams_oof[k]) for k in ORDER] + [rows.X[:, extra_cols]])
    print(f"  fitting the context model over {FP.shape[1]} stacked features", flush=True)
    ctx = fit_stage2(FP, rows.y.astype(np.float32), hidden=32, seed=0)
    EP = context_features(rows, streams_oof["marker"])
    start_head = fit_stage2(EP, soft(is_start, rows.video), hidden=32)
    end_head = fit_stage2(EP, soft(is_resume, rows.video), hidden=32)

    missing = [p for p in ("bge.pt", "edge_start.pt", "edge_resume.pt") if not (MODELS / p).exists()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "detectors": detectors,
        "potion": {"mean": torch.from_numpy(pmean), "std": torch.from_numpy(pstd), "state": pmodel.state_dict()},
        "sequence": {"state": seq_model.state_dict(), "mean": torch.from_numpy(smean), "std": torch.from_numpy(sstd),
                     **SEQ},
        "stages": {"context": stage_state(ctx), "start": stage_state(start_head), "end": stage_state(end_head)},
        "context_inputs": FP.shape[1], "edge_inputs": EP.shape[1], "extra_cols": extra_cols,
        "order": ORDER, "thresholds": th, "thresholds_v2": th2,
        "fine_tuned": {k: str((MODELS / f"{k}.pt").relative_to(BUNDLE.parent)) for k in ("bge", "edge_start", "edge_resume")},
        "trained_on": f"{rows.videos} pooled videos, {rows.reads} reads",
    }, OUT)
    print(f"\nsaved {OUT}")
    if missing:
        print(f"  NOTE: the fine-tuned checkpoints are not on disk yet ({', '.join(missing)}); "
              f"run marker/data/export_chain.sh. The bundle is incomplete until they are there.")
    return 0


def verify(which: str) -> int:
    """Score a fresh set through the SAVED bundle and check it against the recorded grade."""
    from serve_candidate import Candidate
    fresh = f"features_{which}.npz"
    T = load(DATA / fresh)
    potT = np.load(DATA / POTION[fresh])["X"]
    potT = potT[:, :potT.shape[1] - 2]
    bge_T = np.load(DATA / f"finetune_full_bge_seed0__{which}.npy")
    fs_T = np.load(DATA / f"finetune_full_edge_start_seed0__{which}.npy")
    fe_T = np.load(DATA / f"finetune_full_edge_resume_seed0__{which}.npy")
    c = Candidate(OUT)
    print(f"{fresh}: {T.videos} videos, {T.reads} reads -- scored through {OUT.name}\n")
    ctx, ps, pe = c.score(T, potT, bge_T)
    for b in (5, 10):
        kept = place(regions_by_video(ctx, T, c.thresholds[b], 1), T, ps, pe)
        report(f"B={b:>2} bundle (stack + fine-tuned BGE)", kept, T)
        kept2 = place(regions_by_video(ctx, T, c.thresholds_v2[b], 1), T, (ps + fs_T) / 2, (pe + fe_T) / 2)
        report(f"B={b:>2} bundle v2 (averaged edge heads)", kept2, T)
    print("\ncompare with marker/data/prereg_holdout4.log: the bundle is only the graded system if these match.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", metavar="SET", help="a fresh set name, e.g. holdout4")
    args = ap.parse_args()
    return verify(args.verify) if args.verify else build()


if __name__ == "__main__":
    raise SystemExit(main())
