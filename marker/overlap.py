"""How much of the same ad do the regex, the downloaded models and our models catch? (2026-09-22)

Simon's question: compare the regex with the small models we downloaded and with our own models,
and measure the total overlap of ad they catch. Measured in ad SECONDS on the pooled 205 videos,
out of fold, each detector at the setting it would actually run at:

  regex             the six hand-written cue patterns, as they are (a line matches or not)
  qwen3:8b          downloaded, asked about every window (qwen_sweep.py), its line ranges as it gave them
  potion model      downloaded potion-base-8M embeddings + our linear head + context model + edge heads
  MiniLM model      downloaded MiniLM embeddings (meaning only) + our head + context model + edge heads
  shipped free tier our marker (MiniLM + seams, cues, position) + context model + edge heads
  sequence model    ours, end to end, + context model + edge heads
  stacked model     ours, one context model over five detectors (detector_bakeoff.py), + edge heads

The learned ones use the pooled rule's B = 10 setting (most ad time with at most 10 s of show lost
per video and at most 2% of videos over 60 s); the regex and qwen have no knob and run as they are.

    python marker/overlap.py        # needs data/bakeoff_streams.npz; adds qwen when qwen_sweep.jsonl is complete
"""

import json
import sys
from itertools import combinations

import numpy as np

from build_production import over_cap, pooled
from detector_bakeoff import CACHE, graded, pick, sweep
from experiments import runs
from train import DATA, LAST_LINE_SECONDS
from qwen_sweep import OUT as QWEN, windows

sys.stdout.reconfigure(encoding="utf-8")


def to_flags(kept: dict, idx: dict, n: int) -> np.ndarray:
    f = np.zeros(n, dtype=bool)
    for vid, spans in kept.items():
        r = idx[vid]
        for lo, hi in spans:
            f[r[lo:hi]] = True
    return f


def qwen_flags(idx: dict, n: int, path=None, require_complete: bool = True):
    """Lines qwen said are a promotional read, and the videos whose every window it has answered."""
    src = path or QWEN
    if not src.exists():
        return None, set()
    answers = {}
    with src.open(encoding="utf-8") as f:
        for rec in map(json.loads, f):
            if rec.get("promo") is not None:
                answers[(rec["video"], rec["lo"])] = rec
    need = {}
    for w in windows():
        need.setdefault(w["video"], []).append(w["lo"])
    complete = {v for v, los in need.items() if all((v, lo) in answers for lo in los)}
    flags = np.zeros(n, dtype=bool)
    for (vid, lo), rec in answers.items():
        if (require_complete and vid not in complete) or not rec["promo"] or vid not in idx:
            continue
        r, width = idx[vid], rec["hi"] - rec["lo"]
        s, e = rec.get("start_line"), rec.get("end_line")
        if not isinstance(s, int) or not isinstance(e, int):
            s, e = 0, width - 1
        s, e = max(0, min(s, width - 1)), max(0, min(e, width - 1))
        if e < s:
            s, e = e, s
        flags[r[lo + s:lo + e + 1]] = True
    return flags, complete


def main() -> int:
    rows, _, _ = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    cache = dict(np.load(CACHE))
    videos = [str(v) for v in np.unique(rows.video)]
    idx = {v: np.flatnonzero(rows.video == v) for v in videos}
    secs = np.zeros(len(rows))
    for v, r in idx.items():
        s = rows.start_seconds[r]
        secs[r] = np.diff(np.append(s, s[-1] + LAST_LINE_SECONDS))
    ad = rows.y == 1
    n = len(rows)

    names = [str(x) for x in rows.feature_names]
    cue = rows.X[:, [i for i, x in enumerate(names) if x.startswith("cue_")]].sum(1) > 0
    det = {"regex": cue}
    learned = {"potion model": cache["ctx_potion"], "MiniLM model": cache["ctx_meaning"],
               "shipped free tier": level2, "sequence model": cache["ctx_sequence"],
               "stacked model": cache["ctx_stack"], "regex + our context model": cache["ctx_cues"]}
    for name, scores in learned.items():
        best = pick(graded(sweep(scores, rows, p_start, p_end), rows), 10)
        det[name] = to_flags(best[1], idx, n)
    q, complete = qwen_flags(idx, n)

    def report(mask: np.ndarray, title: str, dets: dict) -> None:
        nv = len({str(v) for v in rows.video[mask]})
        A = secs[mask & ad].sum()
        print(f"\n=== {title}: {nv} videos, {A / 60:.0f} min of ad ===")
        print(f"  {'detector':<28}{'ad caught':>10}{'show lost/video':>17}{'also caught by regex':>22}")
        for name, f in dets.items():
            got = secs[mask & ad & f].sum()
            lost = secs[mask & ~ad & f].sum() / nv
            both = secs[mask & ad & f & dets["regex"]].sum()
            print(f"  {name:<28}{got / A:10.1%}{lost:15.1f} s{both / max(got, 1e-9):21.0%}")

        core = [k for k in dets if k not in ("stacked model", "regex + our context model")]
        any_f = np.any([dets[k] for k in core], axis=0)
        all_f = np.all([dets[k] for k in core], axis=0)
        print(f"\n  across {', '.join(core)}:")
        print(f"    caught by at least one   {secs[mask & ad & any_f].sum() / A:6.1%}   "
              f"(show lost if all were skipped: {secs[mask & ~ad & any_f].sum() / nv:.1f} s/video)")
        print(f"    caught by every one      {secs[mask & ad & all_f].sum() / A:6.1%}")
        print(f"    caught by none           {secs[mask & ad & ~any_f].sum() / A:6.1%}")
        print("    caught by ONLY this one:")
        for k in core:
            others = np.any([dets[j] for j in core if j != k], axis=0)
            print(f"      {k:<26}{secs[mask & ad & dets[k] & ~others].sum() / A:6.1%}")
        st = dets["stacked model"]
        print(f"    the stacked model alone gets {secs[mask & ad & st].sum() / max(secs[mask & ad & any_f].sum(), 1e-9):.0%}"
              f" of what all of them catch together")

        groups = {"regex": dets["regex"],
                  "downloaded": np.any([dets[k] for k in ("potion model", "MiniLM model", "qwen3:8b") if k in dets], axis=0),
                  "ours": np.any([dets[k] for k in ("shipped free tier", "sequence model", "stacked model")], axis=0)}
        print("\n  THREE GROUPS (downloaded = " + " + ".join(k for k in ("potion model", "MiniLM model", "qwen3:8b") if k in dets)
              + "; ours = free tier + sequence + stacked), share of all ad time:")
        g = list(groups)
        for size in (3, 2, 1):
            for combo in combinations(g, size):
                inside = np.all([groups[k] for k in combo], axis=0)
                outside = np.any([groups[k] for k in g if k not in combo], axis=0) if size < 3 else np.zeros(n, bool)
                label = " + ".join(combo) + (" only" if size < 3 else " (all three)")
                print(f"    {label:<34}{secs[mask & ad & inside & ~outside].sum() / A:6.1%}")
        print(f"    {'none of them':<34}{secs[mask & ad & ~np.any(list(groups.values()), axis=0)].sum() / A:6.1%}")

        print("\n  PAIRWISE: of the ad the ROW detector catches, the share the COLUMN detector also catches")
        keys = list(dets)
        short = [k.split()[0][:8] for k in keys]
        print("  " + " " * 28 + "".join(f"{s:>9}" for s in short))
        for a in keys:
            got = secs[mask & ad & dets[a]].sum()
            cells = "".join(f"{secs[mask & ad & dets[a] & dets[b]].sum() / max(got, 1e-9):9.0%}" for b in keys)
            print(f"  {a:<28}{cells}")

        print("\n  FALSE ALARMS: of the show each detector wrongly skips, the share the shipped free tier also skips")
        base = dets["shipped free tier"]
        for name, f in dets.items():
            lost = secs[mask & ~ad & f].sum()
            print(f"    {name:<28}{secs[mask & ~ad & f & base].sum() / max(lost, 1e-9):6.0%}  of {lost / nv:5.1f} s/video")

    everyone = np.ones(n, dtype=bool)
    if q is not None and len(complete) == len(videos):
        report(everyone, "ALL POOLED VIDEOS", {**det, "qwen3:8b": q})
    else:
        report(everyone, "ALL POOLED VIDEOS (qwen not swept yet)", det)
        if q is not None and complete:
            report(np.isin(rows.video, list(complete)), "ONLY THE VIDEOS QWEN HAS SWEPT SO FAR", {**det, "qwen3:8b": q})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
