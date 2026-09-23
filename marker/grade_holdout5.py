"""Pre-registration 4, systems C-G, on holdout 5, graded once. (2026-09-24)

Everything runs through the SAVED bundle (`candidate.pt`, its checkpoints and its gate), as the
extension serves it; the community model runs live on each video's captions. Thresholds come from
the bundle and from `veto_thresholds.json`, all fixed before holdout 5 was built. Regions per system
are written to data/holdout5_regions.json for the category table.

    C  v2 + chapters                      D  v3 + chapters (level 1)
    E  loose v3, extras need a brand word  F  loose v3, extras need qwen's "ad" (level 2, reasoning off)
    G  v3 + claude-haiku edges (level 3)

    python marker/grade_holdout5.py [--skip-claude] [--skip-qwen]
"""

import argparse
import json
import re
import sys
import time

import numpy as np
import torch

from candidate import chapter_regions, report
from edge_heads import place
from export_candidate import OUT, POTION
from replay import regions_by_video
from serve_candidate import CandidateTier
from stack_sbml import sbml_stream
from train import DATA, load

sys.stdout.reconfigure(encoding="utf-8")
WHICH = "holdout5"
WORD = re.compile(r"[a-z][a-z0-9]{3,}")


def lost_by_video(kept: dict, T) -> list[float]:
    out = []
    for v in np.unique(T.video):
        r = np.flatnonzero(T.video == v)
        s = T.start_seconds[r]
        sec = np.diff(np.append(s, s[-1] + 3.0))
        y = T.y[r]
        out.append(sum(float(sec[lo:hi][y[lo:hi] == 0].sum()) for lo, hi in kept.get(str(v), [])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-claude", action="store_true")
    ap.add_argument("--skip-qwen", action="store_true")
    a = ap.parse_args()
    fresh = f"features_{WHICH}.npz"
    T = load(DATA / fresh)
    pot = np.load(DATA / POTION.get(fresh, f"features_potion_{WHICH}.npz"))["X"]
    pot = pot[:, :pot.shape[1] - 2]
    text = {}
    with open(DATA / f"examples_{WHICH}.jsonl", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                text[(r["videoID"], r["i"])] = r["text"]
    d = np.load(DATA / fresh)
    texts = [text[(str(v), int(i))] for v, i in zip(d["video"], d["line"])]
    meta = json.loads((DATA / "watch_meta.json").read_text(encoding="utf-8"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = CandidateTier(OUT, device=dev, v3=True, edges_with="haiku")
    print(f"{fresh}: {T.videos} videos, {T.reads} reads, {len(set(T.channel))} channels; bundle {OUT.name}, {dev}\n", flush=True)

    t0 = time.time()
    cheap = c.cheap_streams(T, pot)
    mask = c.gate_mask(cheap, T)
    g = c.gate
    bge = c.ft_stream("bge", texts, T.video, mask, float(g["neutral"]))
    fs = c.ft_stream("edge_start", texts, T.video, mask, float(g.get("neutral_start", 0.0)))
    fe = c.ft_stream("edge_resume", texts, T.video, mask, float(g.get("neutral_end", 0.0)))
    from sponsorblock_ml import LivePredictor
    community = LivePredictor(dev)
    preds = {}
    for v in dict.fromkeys(map(str, T.video)):
        caps = json.loads((DATA / "captions" / f"{v}.json").read_text(encoding="utf-8"))
        preds[v] = community(caps)
    ctx2, ps, pe = c.score_from(T, {**cheap, "bge": bge})
    ctx3, _, _ = c.score_v3(T, {**cheap, "bge": bge, "sbml": sbml_stream(T, preds)})
    ps, pe = (ps + fs) / 2, (pe + fe) / 2
    print(f"streams computed in {time.time() - t0:.0f} s (fine-tuned models read {mask.mean():.0%} of lines)\n", flush=True)

    local = {str(v): np.flatnonzero(T.video == v) for v in np.unique(T.video)}
    veto = json.loads((DATA / "veto_thresholds.json").read_text())
    brand_words = set(json.loads((DATA / "brand_words.json").read_text()))
    brand = np.array([bool(set(WORD.findall(t.lower())) & brand_words) for t in texts])
    systems = {}

    def show(name, kept):
        kept = chapter_regions(kept, T, meta)
        systems[name] = {v: [[int(x), int(z)] for x, z in s] for v, s in kept.items()}
        report(name, kept, T)
        lost = lost_by_video(kept, T)
        print(f"  {'':<44} worst single video {max(lost):.0f} s of show lost", flush=True)

    for b in (10, 5):
        show(f"C  B={b:>2} v2 + chapters", place(regions_by_video(ctx2, T, c.thresholds_v2[b], 1), T, ps, pe))
        show(f"D  B={b:>2} v3 + chapters (level 1)", place(regions_by_video(ctx3, T, c.v3["thresholds"][b], 1), T, ps, pe))

    def loose_regions(b: int, checker: np.ndarray, which: str) -> dict:
        th = veto[which][str(b)]
        found = regions_by_video(ctx3, T, float(th["loose"]), 1)
        keep = {}
        for v, spans in found.items():
            r = local[v]
            ok = [(x, z) for x, z in spans if checker[r][x:z].any() or ctx3[r][x:z].max() >= float(th["strict"])]
            if ok:
                keep[v] = ok
        return place(keep, T, ps, pe)

    for b in (10, 5):
        show(f"E  B={b:>2} loose v3 + brand words", loose_regions(b, brand, "brand"))

    if not a.skip_qwen:
        # Level 2: qwen answers only the 45-line windows (stride 40, as in qwen_sweep.py) that overlap a
        # loose extra find, reasoning off. Flags follow overlap.qwen_flags exactly.
        from confirm_check import unload
        from qwen_sweep import ASK, STRIDE, WINDOW, ask
        q = np.zeros(len(T.y), dtype=bool)
        asked, seen = 0, set()
        t1 = time.time()
        for b in (10, 5):
            th = veto["qwen"][str(b)]
            for v, spans in regions_by_video(ctx3, T, float(th["loose"]), 1).items():
                r = local[v]
                lines = [texts[k] for k in r]
                extras = [(x, z) for x, z in spans if ctx3[r][x:z].max() < float(th["strict"])]
                for lo in range(0, max(1, len(lines) - WINDOW + STRIDE), STRIDE):
                    hi = min(len(lines), lo + WINDOW)
                    if any(x < hi and z > lo for x, z in extras) and (v, lo) not in seen:
                        seen.add((v, lo))
                        ans = ask(ASK + "\n\n" + "\n".join(f"{i:3d}: {t}" for i, t in enumerate(lines[lo:hi])))
                        asked += 1
                        if ans and ans.get("promo"):
                            width = hi - lo
                            s, e = ans.get("start_line"), ans.get("end_line")
                            if not isinstance(s, int) or not isinstance(e, int):
                                s, e = 0, width - 1
                            s, e = max(0, min(s, width - 1)), max(0, min(e, width - 1))
                            if e < s:
                                s, e = e, s
                            q[r[lo + s:lo + e + 1]] = True
                    if hi == len(lines):
                        break
        unload()
        print(f"\nqwen answered {asked} windows in {time.time() - t1:.0f} s; unloaded\n", flush=True)
        for b in (10, 5):
            show(f"F  B={b:>2} loose v3 + qwen (level 2)", loose_regions(b, q, "qwen"))

    if not a.skip_claude:
        t2 = time.time()
        for b in (10, 5):
            found = regions_by_video(ctx3, T, c.v3["thresholds"][b], 1)
            show(f"G  B={b:>2} v3 + claude-haiku edges (level 3)", c._model_edges(found, T, texts, ps, pe))
            print(f"  {'':<44} Claude calls: {c.last_edge_stats}", flush=True)
        print(f"level 3 Claude time {time.time() - t2:.0f} s", flush=True)

    (DATA / "holdout5_regions.json").write_text(json.dumps(systems), encoding="utf-8")
    print("\nregions saved to data/holdout5_regions.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
