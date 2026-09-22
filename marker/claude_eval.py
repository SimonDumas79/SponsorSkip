"""The Claude tier graded by our rule, alone and as a detector in the stack. (2026-09-22)

The extension's default reader is Claude Haiku (server/agents.mjs). It was only ever checked on 15
human-marked reads. Here `claude_label.py --model haiku` (the research window sweep: every 60-line
window, overlapping, line-index answers, structural guards) labels the 160 pooled videos whose
SponsorBlock labels postdate the community model's training, and its segments are graded like
everything else: alone, and as one more line score in the stack (trained and graded on the same 160
videos with channel-grouped CV). Claude never saw our labels, so there is no leakage.
The prompt is the research labeller's, not word for word the server's.

    python marker/claude_eval.py
"""

import ctypes
import json
import sys

import numpy as np
import torch

from build_production import over_cap, pooled
from context_stack import context_features
from detector_bakeoff import CACHE, graded, line, pick
from experiments import runs
from replay import grade_regions
from sbml_eval import load_preds
from stack_check import oof_seeded, sweep_fine
from stack_sbml import sbml_stream
from train import DATA, Rows

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(8)


def claude_stream(rows, labels: dict) -> np.ndarray:
    out = np.zeros(len(rows), dtype=np.float32)
    for v in np.unique(rows.video):
        rec = labels.get(str(v))
        if not rec:
            continue
        r = np.flatnonzero(rows.video == v)
        s = rows.start_seconds[r]
        for seg in rec.get("segments", []):
            out[r[(s >= seg["start"]) & (s < seg["end"])]] = 1.0
    return out


def main() -> int:
    rows, _, _ = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    bake = dict(np.load(CACHE))
    fs, fe = np.load(DATA / "finetune_oof_edge_start_seed0.npy"), np.load(DATA / "finetune_oof_edge_resume_seed0.npy")
    labels = {r["videoID"]: r for r in json.load(open(DATA / "claude_labels_unseen_haiku.json", encoding="utf-8"))}
    have = np.array([str(v) in labels for v in rows.video])
    S = Rows(rows.X[have], rows.y[have], rows.video[have], rows.channel[have], rows.start_seconds[have],
             rows.split[have], rows.feature_names)
    cl = claude_stream(rows, labels)
    sb = sbml_stream(rows, load_preds())
    print(f"Claude (Haiku) labelled {S.videos} videos, {S.reads} reads\n")

    kept = {}
    for v in np.unique(S.video):
        r = np.flatnonzero(S.video == v)
        spans = runs((cl[have][r] > 0).astype(np.int8))
        if spans:
            kept[str(v)] = spans
    g = grade_regions(kept, S)
    print(f"  Claude alone, as it answers: ad time {g['coverage']:.1%}  show lost {g['show']:.1f} s/video  "
          f"over 60 s {over_cap(kept, S):.1%}")
    sbk = {}
    for v in np.unique(S.video):
        r = np.flatnonzero(S.video == v)
        spans = runs((sb[have][r] >= 0.5).astype(np.int8))
        if spans:
            sbk[str(v)] = spans
    g2 = grade_regions(sbk, S)
    print(f"  community SponsorBlock model (same videos): ad time {g2['coverage']:.1%}  show lost {g2['show']:.1f} s/video  "
          f"over 60 s {over_cap(sbk, S):.1%}\n")

    names = [str(n) for n in rows.feature_names]
    extra = rows.X[:, [i for i, n in enumerate(names) if n.startswith("cue_") or n.startswith("desc_")]][have]
    heads = ((p_start + fs) / 2)[have], ((p_end + fe) / 2)[have]
    bge = np.load(DATA / "finetune_oof_bge_seed0.npy")
    base = [level1, bake["meaning"], bake["structure"], bake["potion"], np.load(DATA / "sequence_oof_h32_r7.npy"), bge]
    F = [context_features(S, x[have]) for x in base]
    variants = {"stack + BGE": F, "+ community model": F + [context_features(S, sb[have])],
                "+ Claude": F + [context_features(S, cl[have])],
                "+ community model + Claude": F + [context_features(S, sb[have]), context_features(S, cl[have])]}
    for name, parts in variants.items():
        sc = oof_seeded(S, np.column_stack(parts + [extra]), 0)
        gg = graded(sweep_fine(sc, S, *heads), S)
        for b in (5, 10):
            print(line(f"B={b:>2} {name}", pick(gg, b)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
