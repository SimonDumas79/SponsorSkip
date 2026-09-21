"""The whole system, tier by tier: settings chosen by rule on cross-validation, graded once on the holdout.

The pieces, each a separate model with one job:
  detect    the marker scores every caption line; regions at the checker threshold, or regions
            from the context model (context_stack.py), which reads the marker's scores in context
  confirm   tier 1: the region judge (no language model, free, offline)
            tier 2: qwen3 on the local GPU (its recorded yes/no), alone or inside the judge
  place     the region's own edges, the edge heads (start / resume models), or qwen's line numbers

For each tier the settings (judge cut-off, which edges) are chosen on
cross-validation by ONE written rule: skip the most ad time while losing at
most B seconds of real show per video, with no single video losing more than
60 s. B is 5 and 10. Then every chosen system is trained on all the
cross-validation data and graded on the holdout, once, and every number is
reported whether it helps or not. Nothing here is chosen by looking at the
holdout.

    python marker/system_eval.py            # choose on CV, print the CV table
    python marker/system_eval.py --holdout  # and grade the choices on the holdout
"""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

from context_stack import context_features, fit_stage2
from edge_heads import edge_oof, place, qwen_placed, soft
from experiments import BASE, cross_validate
from region_judge import WITH_QWEN, WITHOUT_QWEN, fit_judge, out_of_fold as judge_oof, region_table
from replay import HEADER, grade_regions, load_verdicts, per_read, regions_by_video, row
from train import DATA, load

CV_RUN, HOLDOUT_RUN = DATA / "confirm_verdicts.jsonl", DATA / "confirm_verdicts_holdout.jsonl"
CUTS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97]
WORST_CAP = 60.0


class World:
    """Everything one dataset contributes: rows, marker scores, regions, and per-line edge probabilities."""

    def __init__(self, rows, level1, verdicts, edges_path, judges, p_start, p_end, level2):
        self.rows, self.level1, self.edges_path, self.level2 = rows, level1, edges_path, level2
        self.F, self.info = region_table(rows, level1, 0.84, 3, verdicts)
        self.judges, self.p_start, self.p_end = judges, p_start, p_end

    def kept(self, select) -> dict:
        out = {}
        for k, (i, f) in enumerate(zip(self.info, self.F)):
            if select(k, f):
                out.setdefault(i["video"], []).append((i["lo"], i["hi"]))
        return out

    def skip(self, kept: dict, edges: str) -> dict:
        if edges == "heads":
            return place(kept, self.rows, self.p_start, self.p_end)
        if edges == "qwen":
            return qwen_placed(kept, self.edges_path, self.rows)[0]
        return kept


def systems():
    """(tier, name, params, selector(world) -> kept regions, edges) for every candidate setting."""
    out = []
    for t, edges in product(np.round(1 / (1 + np.exp(-np.arange(2.0, 6.0, 0.1))), 4), ("region", "heads")):
        out.append(("tier 1 (free)", f"context model th {t}, {edges} edges",
                    lambda w, t=t: regions_by_video(w.level2, w.rows, t, 1), edges))
    for edges in ("region", "heads", "qwen"):
        if edges != "qwen":
            for t in CUTS:
                out.append(("tier 1 (free)", f"judge p>={t}, {edges} edges",
                            lambda w, t=t: w.kept(lambda k, f: w.judges["nq"][k] >= t), edges))
        for b in (0.995, 1.01):
            out.append(("tier 2 (qwen)", f"qwen yes{' or peak>=' + str(b) if b < 1 else ''}, {edges} edges",
                        lambda w, b=b: w.kept(lambda k, f: f[10] == 1.0 or f[0] >= b), edges))
        for t in CUTS:
            out.append(("tier 2 (qwen)", f"judge+qwen p>={t}, {edges} edges",
                        lambda w, t=t: w.kept(lambda k, f: w.judges["q"][k] >= t), edges))
    return out


def choose(world: World, budget: float) -> dict:
    """Per tier, the best system under the rule; and, labelled as such, the best that breaks only the cap.

    The second is reported, never silently substituted: on cross-validation every qwen system breaks
    the 60 s cap on the same video (npGAxyalY4I, a Hindi video with nonsense English auto-captions).
    """
    best = {}
    for tier, name, select, edges in systems():
        g = grade_regions(world.skip(select(world), edges), world.rows)
        if g["show"] > budget:
            continue
        slot = tier if g["worst"] <= WORST_CAP else f"{tier}, breaks the {WORST_CAP:.0f} s cap"
        if slot not in best or (g["coverage"], g["recall"], -g["show"]) > (
                best[slot][2]["coverage"], best[slot][2]["recall"], -best[slot][2]["show"]):
            best[slot] = (name, (select, edges), g)
    return best


def tag(tier: str) -> str:
    return ("T1" if tier.startswith("tier 1") else "T2") + ("*" if "breaks" in tier else "")


def marker_alone(rows, scores, budget):
    best = None
    for th, sm in product(np.round(1 / (1 + np.exp(-np.arange(0, 9, 0.05))), 4), (1, 3)):
        g = grade_regions(regions_by_video(scores, rows, th, sm), rows)
        if g["show"] <= budget and g["worst"] <= WORST_CAP and (
                best is None or (g["coverage"], g["recall"]) > (best[2]["coverage"], best[2]["recall"])):
            best = (float(th), sm, g)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)

    # Cross-validation world: every score out-of-fold.
    rows = load(DATA / "features.npz")
    d = np.load(DATA / "features.npz")
    level1 = cross_validate(BASE, rows, seed=0)
    F = context_features(rows, level1)
    cache = DATA / "edge_heads_oof.npy"
    if cache.exists():
        p_start, p_end = np.load(cache)
    else:
        p_start = edge_oof(rows, F, soft(d["is_start"], rows.video))
        p_end = edge_oof(rows, F, soft(d["is_resume"], rows.video))
    verdicts = load_verdicts(CV_RUN)
    RF, info = region_table(rows, level1, 0.84, 3, verdicts)
    y = np.array([i["is_read"] for i in info], dtype=int)
    judges = {"nq": judge_oof(RF, y, info, WITHOUT_QWEN), "q": judge_oof(RF, y, info, WITH_QWEN)}
    level2 = np.load(DATA / "context_stack_oof.npy")   # context_stack.py: out-of-fold, same channel folds
    cv = World(rows, level1, verdicts, CV_RUN.with_suffix(".edges.jsonl"), judges, p_start, p_end, level2)

    print(f"CROSS-VALIDATION: {rows.videos} videos, {rows.reads} reads. Rule: most ad time skipped with "
          f"<= B s of show lost per video and no video over {WORST_CAP:.0f} s.")
    print(HEADER)
    picks = {}
    for budget in (5, 10):
        alone = marker_alone(rows, level1, budget)
        if alone:
            picks[("marker alone", budget)] = alone[:2]
            print(row(f"B={budget}: marker alone th {alone[0]} smooth {alone[1]}", alone[2]))
        for tier, (name, sel, g) in choose(cv, budget).items():
            picks[(tier, budget)] = (name, sel)
            print(row(f"B={budget} {tag(tier)}: {name}", g))
    print("  (T1 = free tier, T2 = local-GPU tier; * = breaks the 60 s worst-video cap, reported, not preferred)")

    if not args.holdout:
        print("\nholdout not graded (pass --holdout once, at the end)")
        return 0

    # Holdout world: every component trained on ALL the cross-validation data, applied once.
    manifest = json.loads(HOLDOUT_RUN.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    h_rows = load(Path(manifest["features"]))
    h_level1 = np.load(manifest["scores"])
    HF = context_features(h_rows, h_level1)
    h_start = fit_stage2(F, soft(d["is_start"], rows.video), hidden=32)(HF)
    h_end = fit_stage2(F, soft(d["is_resume"], rows.video), hidden=32)(HF)
    h_verdicts = load_verdicts(HOLDOUT_RUN)
    HRF, h_info = region_table(h_rows, h_level1, 0.84, 3, h_verdicts)
    h_judges = {"nq": fit_judge(RF, y, WITHOUT_QWEN)(HRF), "q": fit_judge(RF, y, WITH_QWEN)(HRF)}
    h_level2 = fit_stage2(F, rows.y.astype(np.float32), hidden=32)(HF)
    hw = World(h_rows, h_level1, h_verdicts, HOLDOUT_RUN.with_suffix(".edges.jsonl"), h_judges, h_start, h_end,
               h_level2)

    print(f"\nHOLDOUT (once): {h_rows.videos} few-review videos, {h_rows.reads} reads, "
          "settings exactly as chosen above")
    print(HEADER)
    cue_cols = [i for i, n in enumerate(h_rows.feature_names) if n.startswith("cue_")]
    cues = (h_rows.X[:, cue_cols].sum(1) > 0).astype(np.float32)
    print(row("cue patterns (no model)", grade_regions(regions_by_video(cues, h_rows, 0.5, 3), h_rows)))
    results, skipped = {}, {}
    for (tier, budget), pick in picks.items():
        if tier == "marker alone":
            th, sm = pick
            kept = regions_by_video(h_level1, h_rows, th, sm)
            label = f"B={budget}: marker alone th {th}"
        else:
            name, (select, edges) = pick
            kept = hw.skip(select(hw), edges)
            label = f"B={budget} {tag(tier)}: {name}"
        g = grade_regions(kept, h_rows)
        results[label], skipped[(tier, budget)] = g, kept
        print(row(label, g))

    # Read by read: is the gain spread across reads, or a few lucky ones? And by read length.
    from scipy.stats import binomtest
    print("\nread by read against the marker alone at the same budget (a read counts as better or worse")
    print("when its skipped share moves by more than 10 points; sign test on those):")
    for (tier, budget), kept in skipped.items():
        if tier == "marker alone" or ("marker alone", budget) not in skipped:
            continue
        mine, base = per_read(kept, h_rows), per_read(skipped[("marker alone", budget)], h_rows)
        diff = np.array([m[3] - b[3] for m, b in zip(mine, base)])
        up, down = int((diff > 0.1).sum()), int((diff < -0.1).sum())
        p = binomtest(up, up + down).pvalue if up + down else 1.0
        lengths = np.array([m[2] for m in mine])
        cov = np.array([m[3] for m in mine])
        buckets = " | ".join(f"{name} {cov[mask].mean():.0%} (n={mask.sum()})" for name, mask in
                             (("<30 s", lengths < 30), ("30-90 s", (lengths >= 30) & (lengths < 90)),
                              (">=90 s", lengths >= 90)) if mask.any())
        print(f"  B={budget} {tag(tier)}: {up} reads better, {down} worse, sign test p={p:.3f}; "
              f"mean share skipped by read length: {buckets}")
    (DATA / "system_eval_holdout.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
