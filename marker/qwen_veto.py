"""Level 2 as designed with Simon (2026-09-22): v3 at a LOOSER threshold, and the local model vetoes finds.
Priced at zero GPU with qwen3's RECORDED answers on every window (qwen_sweep.py, reasoning off; the live
design would turn reasoning on, so this is a lower bound on what qwen can do as the checker).

A region v3 finds at the loose threshold is kept if (a) its peak v3 score clears the strict threshold v3
uses alone (it did not need a checker), or (b) qwen said "ad" on a window covering any of its lines.
"veto all" drops (a): every region must be confirmed by qwen. Same 160 videos, same v3 out-of-fold
scores, same edge heads and threshold rule as stack_sbml.py.

    python marker/qwen_veto.py
    python marker/qwen_veto.py --windows data/veto_windows.json   # the windows a reasoning-on sweep must ask
    python marker/qwen_veto.py --think                             # grade with qwen_sweep_think.jsonl
"""

import argparse
import datetime
import json
import sys

import numpy as np

from build_production import pooled
from detector_bakeoff import graded, line, pick
from edge_heads import place
from overlap import qwen_flags
from replay import regions_by_video
from stack_check import sweep_fine
from train import DATA, Rows

sys.stdout.reconfigure(encoding="utf-8")


LOOSEST = 0.30   # the loosest share of lines any candidate threshold flags (np.geomspace below)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", help="write the sweep windows overlapping any loose find, then stop")
    ap.add_argument("--think", action="store_true", help="use the reasoning-on answers (qwen_sweep_think.jsonl)")
    a = ap.parse_args()
    rows, _, _ = pooled()
    _, _, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    fs, fe = np.load(DATA / "finetune_oof_edge_start_seed0.npy"), np.load(DATA / "finetune_oof_edge_resume_seed0.npy")
    d = json.load(open(DATA / "sb_dates.json"))
    cut = datetime.datetime(2022, 4, 1).timestamp() * 1000
    unseen = np.array([not (d.get(str(v), {}).get("first") and d[str(v)]["first"] < cut) for v in rows.video])
    S = Rows(rows.X[unseen], rows.y[unseen], rows.video[unseen], rows.channel[unseen], rows.start_seconds[unseen],
             rows.split[unseen], rows.feature_names)
    idx = {str(v): np.flatnonzero(rows.video == v) for v in np.unique(rows.video)}
    if a.windows:
        from qwen_sweep import windows
        near = set()
        for seed in (0, 1, 2):
            sc = np.load(DATA / f"stack_sbml_oof_s{seed}.npy")[1]
            for v, spans in regions_by_video(sc, S, float(np.quantile(sc, 1 - LOOSEST)), 1).items():
                for x, z in spans:
                    near.add((v, int(x), int(z)))
        by_video = {}
        for v, x, z in near:
            by_video.setdefault(v, []).append((x, z))
        want = [[w["video"], w["lo"]] for w in windows() if w["video"] in by_video
                and any(x < w["hi"] and z > w["lo"] for x, z in by_video[w["video"]])]
        json.dump(want, open(a.windows, "w"))
        print(f"{len(want)} windows overlap a loose find (of {len(windows())} in all)")
        return 0
    if a.think:
        from qwen_sweep import OUT
        q = qwen_flags(idx, len(rows), path=OUT.with_name("qwen_sweep_think.jsonl"), require_complete=False)[0]
    else:
        q = qwen_flags(idx, len(rows))[0]
    q = q[unseen].astype(bool)
    heads = ((p_start + fs) / 2)[unseen], ((p_end + fe) / 2)[unseen]
    local = {str(v): np.flatnonzero(S.video == v) for v in np.unique(S.video)}
    for seed in (0, 1, 2):
        sc = np.load(DATA / f"stack_sbml_oof_s{seed}.npy")[1]
        base = graded(sweep_fine(sc, S, *heads), S)
        for b in (5, 10):
            print(line(f"s{seed} B={b:>2} v3 alone", pick(base, b)), flush=True)
        strict = {b: pick(base, b)[0] for b in (5, 10)}
        for b in (5, 10):
            for mode in ("veto all", "veto the extras"):
                cands = []
                for lo_th in np.unique(np.quantile(sc, 1 - np.geomspace(0.005, 0.30, 40))):
                    found = regions_by_video(sc, S, float(lo_th), 1)
                    keep = {}
                    for v, spans in found.items():
                        r = local[v]
                        ok = [(a, z) for a, z in spans
                              if q[r][a:z].any() or (mode == "veto the extras" and sc[r][a:z].max() >= strict[b])]
                        if ok:
                            keep[v] = ok
                    cands.append((float(lo_th), place(keep, S, *heads)))
                print(line(f"s{seed} B={b:>2} loose v3 + qwen {mode}", pick(graded(cands, S), b)), flush=True)
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
