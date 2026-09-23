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


LOOSEST = 0.30
WORD = __import__("re").compile(r"[a-z][a-z0-9]{3,}")


def line_texts(rows) -> list[str]:
    """Caption text for every pooled row, in row order (the same order qwen_sweep.windows() uses)."""
    from qwen_sweep import SETS
    text = {}
    for examples, _ in SETS:
        with (DATA / examples).open(encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    r = json.loads(ln)
                    text[(str(r["videoID"]), int(r["i"]))] = r["text"]
    out = [""] * len(rows.y)
    for v in np.unique(rows.video):
        r = np.flatnonzero(rows.video == v)
        lines = []
        for examples, features in SETS:
            d = np.load(DATA / features)
            m = d["video"] == v
            if m.any():
                lines = [text[(str(v), int(i))] for i in d["line"][m]]
                break
        for k, t in zip(r, lines):
            out[k] = t
    return out


def brand_vocabulary(rows, texts: list[str], min_channels: int = 2, min_inside: float = 0.6) -> set[str]:
    """The brand words from ALL pooled reads, for a set of channels none of them came from."""
    from collections import Counter, defaultdict
    inside, total, chans = Counter(), Counter(), defaultdict(set)
    for k, t in enumerate(texts):
        for w in set(WORD.findall(t.lower())):
            total[w] += 1
            if rows.y[k]:
                inside[w] += 1
                chans[w].add(str(rows.channel[k]))
    return {w for w in inside if len(chans[w]) >= min_channels and inside[w] / total[w] >= min_inside}


def brand_lines(rows, texts: list[str], min_channels: int = 2, min_inside: float = 0.6) -> np.ndarray:
    """1 on lines naming a word that marks sponsor reads in OTHER channels (leave-one-channel-out).

    A word is a brand for channel c if it appears inside labelled reads of at least `min_channels` channels
    other than c, and at least `min_inside` of its lines (outside c) are inside a read. Mined from the text
    of the reads, not from any list: NordVPN, Squarespace, Manscaped and so on should fall out of it."""
    from collections import Counter, defaultdict
    words = [set(WORD.findall(t.lower())) for t in texts]
    inside, total, chans = defaultdict(Counter), defaultdict(Counter), defaultdict(set)
    for k, ws in enumerate(words):
        c = str(rows.channel[k])
        for w in ws:
            total[w][c] += 1
            if rows.y[k]:
                inside[w][c] += 1
                chans[w].add(c)
    tot_all = {w: sum(v.values()) for w, v in total.items()}
    in_all = {w: sum(v.values()) for w, v in inside.items()}
    flag = np.zeros(len(texts), dtype=bool)
    cache = {}
    for k, ws in enumerate(words):
        c = str(rows.channel[k])
        for w in ws:
            key = (w, c)
            if key not in cache:
                n_ch = len(chans[w] - {c})
                t = tot_all.get(w, 0) - total[w][c]
                i = in_all.get(w, 0) - inside[w][c]
                cache[key] = n_ch >= min_channels and t > 0 and i / t >= min_inside
            if cache[key]:
                flag[k] = True
                break
    return flag   # the loosest share of lines any candidate threshold flags (np.geomspace below)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", help="write the sweep windows overlapping any loose find, then stop")
    ap.add_argument("--selfpromo", action="store_true", help="also try a self-promo model as the checker (Simon, 09-24)")
    ap.add_argument("--save", action="store_true", help="write the seed-0 thresholds (and brand list) for a holdout")
    ap.add_argument("--brand", action="store_true", help="also try the brand-list veto")
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
    checkers = {"qwen": q}
    if a.brand:
        br = brand_lines(rows, line_texts(rows))[unseen]
        print(f"brand words flag {br.mean():.1%} of lines; {br[S.y == 1].mean():.1%} of read lines", flush=True)
        checkers = {"qwen": q, "brand": br, "qwen or brand": q | br}
    if a.selfpromo:
        # A second small model like the marker, trained on self-promo lines only, out-of-fold with channel
        # groups inside the 160. Its top 2% of lines count as "self-promo here". Labels: `category`.
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        cat = np.load(DATA / "features.npz", allow_pickle=True)
        pooled_cat = {}
        for _, feats in __import__("qwen_sweep").SETS:
            f = np.load(DATA / feats, allow_pickle=True)
            if "category" in f.files:
                for vv, ll, cc in zip(f["video"], f["line"], f["category"]):
                    pooled_cat[(str(vv), int(ll))] = str(cc)
        X = S.X[:, :395]
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        ysp = np.zeros(len(X), dtype=int)
        for v, r in {str(v): np.flatnonzero(S.video == v) for v in np.unique(S.video)}.items():
            d_line = None
            for _, feats in __import__("qwen_sweep").SETS:
                f = np.load(DATA / feats, allow_pickle=True)
                m = f["video"] == v
                if m.any():
                    d_line = f["line"][m]
                    break
            ysp[r] = [pooled_cat.get((v, int(l)), "") == "selfpromo" for l in d_line]
        sp = np.zeros(len(X))
        for tr, te in GroupKFold(5).split(X, ysp, S.channel):
            m = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(X[tr], ysp[tr])
            sp[te] = m.predict_proba(X[te])[:, 1]
        from sklearn.metrics import average_precision_score
        print(f"self-promo model: {ysp.sum()} self-promo lines, out-of-fold average precision "
              f"{average_precision_score(ysp, sp):.3f} (chance {ysp.mean():.3f})", flush=True)
        spf = sp >= np.quantile(sp, 0.98)
        checkers = {"qwen": q, "self-promo model": spf, "qwen or self-promo model": q | spf}
    heads = ((p_start + fs) / 2)[unseen], ((p_end + fe) / 2)[unseen]
    local = {str(v): np.flatnonzero(S.video == v) for v in np.unique(S.video)}
    fixed = {}
    for seed in (0, 1, 2):
        sc = np.load(DATA / f"stack_sbml_oof_s{seed}.npy")[1]
        base = graded(sweep_fine(sc, S, *heads), S)
        for b in (5, 10):
            print(line(f"s{seed} B={b:>2} v3 alone", pick(base, b)), flush=True)
        strict = {b: pick(base, b)[0] for b in (5, 10)}
        for b in (5, 10):
            for (cname, chk), mode in [((n, c), m) for n, c in checkers.items() for m in ("veto all", "veto the extras")]:
                cands = []
                for lo_th in np.unique(np.quantile(sc, 1 - np.geomspace(0.005, 0.30, 40))):
                    found = regions_by_video(sc, S, float(lo_th), 1)
                    keep = {}
                    for v, spans in found.items():
                        r = local[v]
                        ok = [(a, z) for a, z in spans
                              if chk[r][a:z].any() or (mode == "veto the extras" and sc[r][a:z].max() >= strict[b])]
                        if ok:
                            keep[v] = ok
                    cands.append((float(lo_th), place(keep, S, *heads)))
                chosen = pick(graded(cands, S), b)
                print(line(f"s{seed} B={b:>2} loose v3 + {cname} {mode}", chosen), flush=True)
                if seed == 0 and mode == "veto the extras" and chosen:
                    fixed.setdefault(cname, {})[b] = {"loose": chosen[0], "strict": strict[b]}
        print(flush=True)
    if a.save:
        # Pre-registration 4: the thresholds a holdout is graded with, chosen here on seed-0 out-of-fold
        # scores by the written rule, exactly as export_candidate.py chose v3's own. Written before the set exists.
        json.dump(fixed, open(DATA / "veto_thresholds.json", "w"), indent=1)
        if a.brand:
            json.dump(sorted(brand_vocabulary(rows, line_texts(rows))), open(DATA / "brand_words.json", "w"), indent=0)
        print(f"saved veto_thresholds.json{' and brand_words.json' if a.brand else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
