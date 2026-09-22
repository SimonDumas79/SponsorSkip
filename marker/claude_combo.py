"""Claude's segments as the base, our models around them: union, veto, both. (2026-09-22)

On the 160 unseen pooled videos Claude Haiku alone skips 79.1% of ad time at 9.3 s lost per video
(5.0% of videos over 60 s), better than the community model (73.4% at 17.0 s) and better than any stack
with Claude inside it at the same cost (69-70%): the context model blurs Claude's exact segments.
So Claude's segments stay as they are, and our stack (BGE, no Claude; out of fold on the 160) is used
around them:
  union   add the stack's own regions (threshold t, averaged edge heads) where Claude found nothing
  veto    drop a Claude segment whose peak stack score is below v (cuts Claude's false alarms)
  both    union and veto
t and v are chosen by the pooled rule on these 160 videos (two knobs, the same videos: optimistic by a
little; the rule's budgets and the 2% cap apply).

    python marker/claude_combo.py
"""

import json
import sys

import numpy as np

from build_production import OVER_CAP_SHARE, over_cap, pooled
from detector_bakeoff import line
from edge_heads import merged, place
from experiments import runs
from replay import grade_regions, regions_by_video
from train import DATA, Rows

sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    rows, _, _ = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    fs, fe = np.load(DATA / "finetune_oof_edge_start_seed0.npy"), np.load(DATA / "finetune_oof_edge_resume_seed0.npy")
    labels = {r["videoID"]: r for r in json.load(open(DATA / "claude_labels_unseen_haiku.json", encoding="utf-8"))}
    have = np.array([str(v) in labels for v in rows.video])
    S = Rows(rows.X[have], rows.y[have], rows.video[have], rows.channel[have], rows.start_seconds[have],
             rows.split[have], rows.feature_names)
    stack = np.load(DATA / "stack_sbml_oof_s0.npy")[0]   # stack + BGE, out of fold on the same 160 (unseen) videos
    ps, pe = ((p_start + fs) / 2)[have], ((p_end + fe) / 2)[have]
    idx = {str(v): np.flatnonzero(S.video == v) for v in np.unique(S.video)}
    claude = {}
    for v, r in idx.items():
        s = S.start_seconds[r]
        spans = []
        for seg in labels[v].get("segments", []):
            m = np.flatnonzero((s >= seg["start"]) & (s < seg["end"]))
            if len(m):
                spans.append((int(m[0]), int(m[-1]) + 1))
        if spans:
            claude[v] = merged(spans)

    def peak(v, lo, hi):
        return float(stack[idx[v]][lo:hi].max())

    ths = np.unique(np.quantile(stack, 1 - np.geomspace(0.004, 0.25, 30)))
    vetoes = [None] + list(np.quantile(stack, [0.5, 0.7, 0.8, 0.9, 0.95]))
    cands = {"Claude alone": [("-", claude)]}
    ours = {float(t): place(regions_by_video(stack, S, float(t), 1), S, ps, pe) for t in ths}
    cands["union"] = []
    cands["veto"] = []
    cands["both"] = []
    for vt in vetoes[1:]:
        kept = {v: [sp for sp in sp_ if peak(v, *sp) >= vt] for v, sp_ in claude.items()}
        cands["veto"].append((f"v {vt:.3f}", {v: s for v, s in kept.items() if s}))
    for t, reg in ours.items():
        for vt in vetoes:
            base = claude if vt is None else {v: [sp for sp in sp_ if peak(v, *sp) >= vt] for v, sp_ in claude.items()}
            out = {}
            for v in idx:
                c = base.get(v, [])
                extra = [sp for sp in reg.get(v, []) if not any(a < sp[1] and sp[0] < b for a, b in claude.get(v, []))]
                if c or extra:
                    out[v] = merged(c + extra)
            cands["union" if vt is None else "both"].append((f"t {t:.3f}" + ("" if vt is None else f", v {vt:.3f}"), out))

    print(f"{S.videos} unseen pooled videos, {S.reads} reads; Claude Haiku segments + stack + BGE (out of fold)\n")
    for name, cs in cands.items():
        graded = [(label, kept, grade_regions(kept, S), over_cap(kept, S)) for label, kept in cs]
        if name == "Claude alone":
            _, kept, g, oc = graded[0]
            print(f"  {name:<14} as it answers: ad time {g['coverage']:6.1%}  show lost {g['show']:4.1f} s  over 60 s {oc:4.1%}")
            continue
        for b in (5, 10):
            ok = [x for x in graded if x[2]["show"] <= b and x[3] <= OVER_CAP_SHARE]
            if ok:
                label, kept, g, oc = max(ok, key=lambda x: x[2]["coverage"])
                print(f"  {name:<14} B={b:>2} ({label}): ad time {g['coverage']:6.1%}  show lost {g['show']:4.1f} s  over 60 s {oc:4.1%}")
            else:
                print(f"  {name:<14} B={b:>2}: nothing meets the rule")
        best = max(graded, key=lambda x: (x[3] <= OVER_CAP_SHARE, x[2]["coverage"] - x[2]["show"] / 100))
        label, kept, g, oc = max([x for x in graded if x[2]["show"] <= 9.3] or graded, key=lambda x: x[2]["coverage"])
        print(f"  {name:<14} at <= 9.3 s (Claude's cost, no cap): ad time {g['coverage']:6.1%} at {g['show']:4.1f} s, over 60 s {oc:4.1%} ({label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
