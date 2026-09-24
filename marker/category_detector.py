"""Detectors trained on Claude's category labels, used as v3's veto. (Simon, 2026-09-24: "train on Claude's categories")

SponsorBlock gave 32 self-promo reads in the 160 and a self-promo model on them barely learned (average
precision 0.011 against 0.005 by chance). Claude's category labels on all 205 pooled videos give 48
own-product reads, 24 other-promo reads and 104 channel-plug reads, separated from sponsor reads.

Targets, one per line, from the Claude segment covering the line's start:
  own+other   own_product or other_promo, style "read"   (the gap: v3 skips ~20% of own-product reads)
  non-sponsor any category but sponsor, style "read"      (adds channel plugs, which Simon counts)
  any promo   every Claude segment, any style

Each is a linear model on the same 395 line features, trained out-of-fold with channel-grouped folds over
all 205 videos, then smoothed over 5 lines. Used as the checker for v3's loose extra finds exactly as
qwen_veto.py does (a find above v3's own threshold never needs the checker). Graded on the 160 videos
v3's out-of-fold scores cover, 3 seeds, against SponsorBlock labels AND against Claude's any-promo labels.
Nothing here touches holdout 5 (seen) -- the fresh grade is holdout 6.

    python marker/category_detector.py
"""

import datetime
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from build_production import pooled
from detector_bakeoff import graded, line, pick
from edge_heads import place
from replay import regions_by_video
from stack_check import sweep_fine
from train import DATA, Rows

sys.stdout.reconfigure(encoding="utf-8")


def claude_targets(rows, extra_label_files: list[str] = ()) -> dict[str, np.ndarray]:
    """extra_label_files: more claude_label.py --categories outputs to merge in (e.g.
    ["claude_labels_categories_holdout5.json"]), for videos beyond the original 205 pooled ones.
    Videos with no entry in any of these files get an all-zero (implicit-negative) target, so callers
    must restrict training to videos actually covered (2026-09-24, enlarging the stack's training pool)."""
    labels = {}
    for fname in ("claude_labels_categories.json", *extra_label_files):
        path = DATA / fname
        if path.exists():
            labels.update({r["videoID"]: r["segments"] for r in json.loads(path.read_text(encoding="utf-8"))})
    out = {k: np.zeros(len(rows.y), dtype=np.int8) for k in ("own+other", "non-sponsor", "any promo")}
    for v in np.unique(rows.video):
        r = np.flatnonzero(rows.video == v)
        t = rows.start_seconds[r]
        for s in labels.get(str(v), []):
            inside = r[(t >= s["start"]) & (t < max(s["end"], s["start"] + 0.1))]
            out["any promo"][inside] = 1
            if s.get("style") == "read" and s.get("category") != "sponsor":
                out["non-sponsor"][inside] = 1
                if s.get("category") in ("own_product", "other_promo"):
                    out["own+other"][inside] = 1
    return out


def smooth(x: np.ndarray, video: np.ndarray, w: int = 5) -> np.ndarray:
    out = np.empty_like(x)
    k = np.ones(w) / w
    for v in np.unique(video):
        r = np.flatnonzero(video == v)
        out[r] = np.convolve(x[r], k, mode="same")
    return out


def main() -> int:
    rows, _, _ = pooled()
    targets = claude_targets(rows)
    X = rows.X[:, :395]
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    oof = {}
    for name, y in targets.items():
        p = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, rows.channel):
            p[te] = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        oof[name] = smooth(p, rows.video)
        ft = DATA / "finetune_oof_bge_claude_nonsponsor_seed0.npy"
        if name == "non-sponsor" and "--ft" in sys.argv and ft.exists():
            # the same target, fine-tuned BGE-small instead of a linear model on frozen features
            oof[name] = np.load(ft)
        print(f"{name:<12} {y.sum():5d} lines ({len(np.unique(rows.video[y == 1]))} videos): out-of-fold average "
              f"precision {average_precision_score(y, oof[name]):.3f} (chance {y.mean():.3f})", flush=True)
    np.save(DATA / "category_detector_oof.npy", np.stack([oof[k] for k in targets]))
    if "--union" in sys.argv:
        return union(rows, targets, oof)

    _, _, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    fs, fe = np.load(DATA / "finetune_oof_edge_start_seed0.npy"), np.load(DATA / "finetune_oof_edge_resume_seed0.npy")
    d = json.load(open(DATA / "sb_dates.json"))
    cut = datetime.datetime(2022, 4, 1).timestamp() * 1000
    unseen = np.array([not (d.get(str(v), {}).get("first") and d[str(v)]["first"] < cut) for v in rows.video])
    heads = ((p_start + fs) / 2)[unseen], ((p_end + fe) / 2)[unseen]
    truths = {"SponsorBlock": rows.y[unseen], "Claude any promo": targets["any promo"][unseen]}
    local = None
    print()
    for tname, ytrue in truths.items():
        S = Rows(rows.X[unseen], ytrue, rows.video[unseen], rows.channel[unseen], rows.start_seconds[unseen],
                 rows.split[unseen], rows.feature_names)
        local = {str(v): np.flatnonzero(S.video == v) for v in np.unique(S.video)}
        print(f"graded against {tname} labels ({int(ytrue.sum())} lines)")
        for seed in (0, 1, 2):
            sc = np.load(DATA / f"stack_sbml_oof_s{seed}.npy")[1]
            base = graded(sweep_fine(sc, S, *heads), S)
            chosen = pick(base, 10)
            print(line(f"  s{seed} B=10 v3 alone", chosen), flush=True)
            strict = chosen[0]
            for name in targets:
                det = oof[name][unseen]
                flag = det >= np.quantile(det, 0.95)   # its top 5% of lines count as "promotion here"
                cands = []
                for lo_th in np.unique(np.quantile(sc, 1 - np.geomspace(0.005, 0.30, 40))):
                    keep = {}
                    for v, spans in regions_by_video(sc, S, float(lo_th), 1).items():
                        r = local[v]
                        ok = [(a, z) for a, z in spans if flag[r][a:z].any() or sc[r][a:z].max() >= strict]
                        if ok:
                            keep[v] = ok
                    cands.append((float(lo_th), place(keep, S, *heads)))
                print(line(f"  s{seed} B=10 loose v3 + {name} detector", pick(graded(cands, S), 10)), flush=True)
        print(flush=True)
    return 0


def union(rows, targets, oof) -> int:
    """The detector as its OWN region source, added to v3's regions: a veto can only keep what v3 already
    found loosely, and v3 never touches 73% of own-product reads. Its threshold is swept and chosen by the
    same rule (most ad time within B = 10), v3 kept at its own threshold. Seeds 0-2, both label sets."""
    _, _, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    fs, fe = np.load(DATA / "finetune_oof_edge_start_seed0.npy"), np.load(DATA / "finetune_oof_edge_resume_seed0.npy")
    d = json.load(open(DATA / "sb_dates.json"))
    cut = datetime.datetime(2022, 4, 1).timestamp() * 1000
    unseen = np.array([not (d.get(str(v), {}).get("first") and d[str(v)]["first"] < cut) for v in rows.video])
    heads = ((p_start + fs) / 2)[unseen], ((p_end + fe) / 2)[unseen]
    for tname, ytrue in (("SponsorBlock", rows.y[unseen]), ("Claude any promo", targets["any promo"][unseen])):
        S = Rows(rows.X[unseen], ytrue, rows.video[unseen], rows.channel[unseen], rows.start_seconds[unseen],
                 rows.split[unseen], rows.feature_names)
        print(f"graded against {tname} labels", flush=True)
        for seed in (0, 1, 2):
            sc = np.load(DATA / f"stack_sbml_oof_s{seed}.npy")[1]
            chosen = pick(graded(sweep_fine(sc, S, *heads), S), 10)
            print(line(f"  s{seed} B=10 v3 alone", chosen), flush=True)
            base = regions_by_video(sc, S, chosen[0], 1)
            for name in ("own+other", "non-sponsor"):
                det = oof[name][unseen]
                cands = []
                for th in np.unique(np.quantile(det, 1 - np.geomspace(0.002, 0.10, 25))):
                    extra = regions_by_video(det, S, float(th), 1)
                    both = {v: sorted(set(base.get(v, [])) | set(extra.get(v, []))) for v in set(base) | set(extra)}
                    cands.append((float(th), place(both, S, *heads)))
                print(line(f"  s{seed} B=10 v3 + {name} regions", pick(graded(cands, S), 10)), flush=True)
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
