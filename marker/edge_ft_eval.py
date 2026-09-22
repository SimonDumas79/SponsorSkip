"""Fine-tuned MiniLM as the start/resume models: does it place edges better? (2026-09-22)

edge_clean.py showed the start model, not label noise, is what leaves starts a median ~4 s off on the
reads where Claude and SponsorBlock agree; perfect starts alone would lift ad time from ~43% to ~63%.
Fine-tuning MiniLM was the biggest gain for detection, so finetune_minilm.py --target start/resume
trains it on the same soft start/resume labels the MLP edge heads use (out of fold, same folds).

Here those scores replace the edge heads' inside edge_heads.place (same search windows), on the shipped
free tier's regions and on the stack + fine-tuned regions. Graded by the pooled rule, and by start/end
error on the reads where both label sources agree (data/edge_verify.jsonl, within 3 s at both edges).
Also tried: the average of the MLP heads and the fine-tuned ones.

    python marker/edge_ft_eval.py
"""

import json
import sys

import numpy as np

from build_production import pooled
from detector_bakeoff import graded, line, pick
from experiments import runs
from stack_check import sweep_fine
from train import DATA

sys.stdout.reconfigure(encoding="utf-8")


def agreed_reads(rows, idx) -> list[tuple[str, int, int]]:
    ver = {(r["video"], r["read"]): r for r in map(json.loads, open(DATA / "edge_verify.jsonl", encoding="utf-8"))}
    out = []
    for v, r in idx.items():
        for n, (a, b) in enumerate(runs(rows.y[r])):
            rec = ver.get((v, n))
            if rec and rec["claude_start"] is not None and abs(rec["d_start_s"]) <= 3 and abs(rec["d_end_s"]) <= 3:
                out.append((v, a, b))
    return out


def edge_errors(kept, rows, idx, agreed):
    es, ee = [], []
    for v, a, b in agreed:
        s = rows.start_seconds[idx[v]]
        spans = [sp for sp in kept.get(v, []) if sp[0] < b and a < sp[1]]
        if spans:
            sp = min(spans, key=lambda x: abs(x[0] - a))
            es.append(abs(s[min(sp[0], len(s) - 1)] - s[a]))
            ee.append(abs(s[min(sp[1], len(s) - 1)] - s[min(b, len(s) - 1)]))
    es, ee = np.array(es), np.array(ee)
    return f"{len(es)} agreed reads: start median {np.median(es):.1f} s ({np.mean(es <= 2):.0%} within 2 s), " \
           f"end median {np.median(ee):.1f} s ({np.mean(ee <= 2):.0%} within 2 s)"


def main() -> int:
    rows, _, _ = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    ft = dict(np.load(DATA / "finetune_streams.npz"))
    fs, fe = np.load(DATA / "finetune_oof_edge_start_seed0.npy"), np.load(DATA / "finetune_oof_edge_resume_seed0.npy")
    idx = {str(v): np.flatnonzero(rows.video == v) for v in np.unique(rows.video)}
    agreed = agreed_reads(rows, idx)
    heads = {"MLP edge heads (shipped)": (p_start, p_end), "fine-tuned start/resume": (fs, fe),
             "average of both": ((p_start + fs) / 2, (p_end + fe) / 2)}
    for system, scores in (("shipped free tier", level2), ("stack + fine-tuned s0", ft["stack_ft0"])):
        print(f"{system}:")
        for name, (ps, pe) in heads.items():
            g = graded(sweep_fine(scores, rows, ps, pe), rows)
            b5, b10 = pick(g, 5), pick(g, 10)
            print(line(f"  B= 5 {name}", b5))
            print(line(f"  B=10 {name}", b10))
            print(f"        at B=10, on {edge_errors(b10[1], rows, idx, agreed)}", flush=True)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
