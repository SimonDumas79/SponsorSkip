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


def smooth(x: np.ndarray, video: np.ndarray, w: int) -> np.ndarray:
    """Rolling maximum over w lines either side, within a video: the cheap layers' reach."""
    out = np.zeros(len(x), dtype=np.float32)
    for v in np.unique(video):
        r = np.flatnonzero(video == v)
        a = x[r]
        out[r] = [a[max(0, i - w):i + w + 1].max() for i in range(len(a))]
    return out


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

    # Simon's first layer, fixed here rather than per video: the cheap five sweep, and only the
    # lines they see something near go to the three fine-tuned models. Measured on holdout 4
    # (cascade.py --gate), all three gated together: at 30% of lines the grade does not move at all
    # (68.4% of ad time, 7.9 s lost, the same figures to the decimal), at 20% it drops to 66.9%
    # because `place` searches 20 lines past a region while the gate reaches 15, and at 10% the
    # detector itself starts missing reads (62.2%). So 30%: a third of the expensive work, nothing
    # given up. The cut is the reach score's 70th percentile on the POOLED data, so a video is never
    # judged against itself and a video with no ad in it simply sends fewer lines through.
    cheap = np.max(np.column_stack([streams_oof[k] for k in
                                    ("marker", "meaning", "structure", "potion", "sequence")]), axis=1)
    reach = smooth(cheap, rows.video, 15)
    gate = {"cut": float(np.quantile(reach, 0.70)), "neutral": float(np.median(bge_oof)), "window": 15, "share": 0.30,
        "neutral_start": float(np.median(fs_oof)), "neutral_end": float(np.median(fe_oof))}
    print(f"  gate: the fine-tuned model reads a line only within 15 lines of a cheap score "
          f"over {gate['cut']:.4f}; elsewhere it takes {gate['neutral']:.4f}", flush=True)

    missing = [p for p in ("bge.pt", "edge_start.pt", "edge_resume.pt") if not (MODELS / p).exists()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "detectors": detectors,
        "potion": {"mean": torch.from_numpy(pmean), "std": torch.from_numpy(pstd), "state": pmodel.state_dict()},
        "sequence": {"state": seq_model.state_dict(), "mean": torch.from_numpy(smean), "std": torch.from_numpy(sstd),
                     **SEQ},
        "stages": {"context": stage_state(ctx), "start": stage_state(start_head), "end": stage_state(end_head)},
        "context_inputs": FP.shape[1], "edge_inputs": EP.shape[1], "extra_cols": extra_cols,
        "order": ORDER, "thresholds": th, "thresholds_v2": th2, "gate": gate,
        "fine_tuned": {k: str((MODELS / f"{k}.pt").relative_to(BUNDLE.parent)) for k in ("bge", "edge_start", "edge_resume")},
        "trained_on": f"{rows.videos} pooled videos, {rows.reads} reads",
    }, OUT)
    print(f"\nsaved {OUT}")
    if missing:
        print(f"  NOTE: the fine-tuned checkpoints are not on disk yet ({', '.join(missing)}); "
              f"run marker/data/export_chain.sh. The bundle is incomplete until they are there.")
    return 0


def add_v3() -> int:
    """Add candidate v3 to the saved bundle: a context model that also reads the community model.

    The same fit candidate.py --sbml grades: the six pooled out-of-fold streams plus the community
    model's line score, trained ONLY on the pooled videos whose SponsorBlock labels postdate that
    model's training (it was trained on SponsorBlock up to early 2022, so it may have seen the rest).
    Thresholds from that model's own out-of-fold scores (stack_sbml.py), never re-chosen here.
    """
    import datetime
    import json

    from sbml_eval import load_preds
    from stack_sbml import sbml_stream

    rows, _, _ = pooled()
    level1, _, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    bake = dict(np.load(CACHE))
    fs_oof = np.load(DATA / "finetune_oof_edge_start_seed0.npy")
    fe_oof = np.load(DATA / "finetune_oof_edge_resume_seed0.npy")
    streams_oof = {"marker": level1, "meaning": bake["meaning"], "structure": bake["structure"],
                   "potion": bake["potion"], "sequence": np.load(DATA / "sequence_oof_h32_r7.npy"),
                   "bge": np.load(DATA / "finetune_oof_bge_seed0.npy")}
    names = [str(n) for n in rows.feature_names]
    extra_cols = [i for i, n in enumerate(names) if n.startswith("cue_") or n.startswith("desc_")]
    d = json.load(open(DATA / "sb_dates.json"))
    cutoff = datetime.datetime(2022, 4, 1).timestamp() * 1000
    unseen = np.array([not (d.get(str(v), {}).get("first") and d[str(v)]["first"] < cutoff) for v in rows.video])
    U = Rows(rows.X[unseen], rows.y[unseen], rows.video[unseen], rows.channel[unseen],
             rows.start_seconds[unseen], rows.split[unseen], rows.feature_names)
    sb = sbml_stream(rows, load_preds())
    FU = np.column_stack([context_features(U, streams_oof[k][unseen]) for k in ORDER]
                         + [context_features(U, sb[unseen]), U.X[:, extra_cols]])
    print(f"v3 context model: {U.videos} pooled videos after the community model's training, "
          f"{FU.shape[1]} features", flush=True)
    ctx3 = fit_stage2(FU, U.y.astype(np.float32), hidden=32, seed=0)
    oof3 = np.load(DATA / "stack_sbml_oof_s0.npy")[1]
    g3 = graded(sweep_fine(oof3, U, ((p_start + fs_oof) / 2)[unseen], ((p_end + fe_oof) / 2)[unseen]), U)
    th3 = {b: float(pick(g3, b)[0]) for b in (5, 10)}
    print(f"v3 thresholds from its CV: B=5 {th3[5]:.4f}, B=10 {th3[10]:.4f}", flush=True)
    b = torch.load(OUT, weights_only=False)
    b["v3"] = {"context": stage_state(ctx3), "context_inputs": FU.shape[1], "thresholds": th3}
    torch.save(b, OUT)
    print(f"saved v3 into {OUT}")
    return 0


def verify_v3(which: str) -> int:
    """Score a fresh set through the bundle's v3 and print it next to the recorded grade."""
    import json

    from candidate import chapter_regions
    from sbml_eval import load_preds
    from serve_candidate import Candidate
    from stack_sbml import sbml_stream
    fresh = f"features_{which}.npz"
    T = load(DATA / fresh)
    potT = np.load(DATA / POTION[fresh])["X"]
    potT = potT[:, :potT.shape[1] - 2]
    bge_T = np.load(DATA / f"finetune_full_bge_seed0__{which}.npy")
    fs_T = np.load(DATA / f"finetune_full_edge_start_seed0__{which}.npy")
    fe_T = np.load(DATA / f"finetune_full_edge_resume_seed0__{which}.npy")
    c = Candidate(OUT)
    st = {**c.streams(T, potT, bge_T), "sbml": sbml_stream(T, load_preds())}
    ctx3, ps, pe = c.score_v3(T, st)
    meta = json.loads((DATA / "watch_meta.json").read_text(encoding="utf-8")) if (DATA / "watch_meta.json").exists() else {}
    print(f"{fresh}: {T.videos} videos, {T.reads} reads -- v3 scored through {OUT.name}\n")
    for b in (5, 10):
        kept = place(regions_by_video(ctx3, T, c.v3["thresholds"][b], 1), T, (ps + fs_T) / 2, (pe + fe_T) / 2)
        report(f"B={b:>2} bundle v3", kept, T)
        if meta:
            report(f"B={b:>2}   + creator chapters", chapter_regions(kept, T, meta), T)
    print("\nthe graded v3 at B=10 + chapters on holdout 4: 67.2% of ad time, 5.5 s lost, none over 60 s")
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


def from_checkpoints(which: str) -> int:
    """The same grade, but the three fine-tuned streams recomputed by the SAVED CHECKPOINTS.

    --verify uses the .npy scores the graded runs produced. Those came from a different training run
    of the same recipe, and GPU training is not bit-identical between runs, so the checkpoint in the
    bundle is a sibling of the graded model rather than the same weights. This grades the sibling: if
    it lands on the recorded numbers, the served system is the measured one in every way that counts.
    """
    import json

    from serve_candidate import Candidate
    fresh = f"features_{which}.npz"
    T = load(DATA / fresh)
    potT = np.load(DATA / POTION[fresh])["X"]
    potT = potT[:, :potT.shape[1] - 2]
    text = {}
    with open(DATA / f"examples_{which}.jsonl", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                text[(r["videoID"], r["i"])] = r["text"]
    d = np.load(DATA / fresh)
    texts = [text[(str(v), int(i))] for v, i in zip(d["video"], d["line"])]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = Candidate(OUT, device=dev)
    print(f"{fresh}: {T.videos} videos, {T.reads} reads -- fine-tuned streams from the checkpoints on {dev}", flush=True)
    bge = c.ft_stream("bge", texts, T.video)
    fs = c.ft_stream("edge_start", texts, T.video)
    fe = c.ft_stream("edge_resume", texts, T.video)
    for name, new, old in (("bge", bge, DATA / f"finetune_full_bge_seed0__{which}.npy"),
                           ("start", fs, DATA / f"finetune_full_edge_start_seed0__{which}.npy"),
                           ("resume", fe, DATA / f"finetune_full_edge_resume_seed0__{which}.npy")):
        if old.exists():
            o = np.load(old)
            print(f"  {name:>6}: correlation with the graded run {np.corrcoef(new, o)[0, 1]:.4f}, "
                  f"mean |difference| {np.abs(new - o).mean():.4f}")
    ctx, ps, pe = c.score(T, potT, bge)
    print()
    for b in (5, 10):
        report(f"B={b:>2} checkpoints (stack + fine-tuned BGE)",
               place(regions_by_video(ctx, T, c.thresholds[b], 1), T, ps, pe), T)
        report(f"B={b:>2} checkpoints v2 (averaged edge heads)",
               place(regions_by_video(ctx, T, c.thresholds_v2[b], 1), T, (ps + fs) / 2, (pe + fe) / 2), T)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", metavar="SET", help="a fresh set name, e.g. holdout4")
    ap.add_argument("--from-checkpoints", metavar="SET",
                    help="the same grade with the fine-tuned streams recomputed by the saved checkpoints")
    ap.add_argument("--add-v3", action="store_true",
                    help="add candidate v3 (+ the community model) to the saved bundle")
    ap.add_argument("--verify-v3", metavar="SET", help="score a fresh set through the bundle's v3")
    args = ap.parse_args()
    if args.add_v3:
        return add_v3()
    if args.verify_v3:
        return verify_v3(args.verify_v3)
    if args.from_checkpoints:
        return from_checkpoints(args.from_checkpoints)
    return verify(args.verify) if args.verify else build()


if __name__ == "__main__":
    raise SystemExit(main())
