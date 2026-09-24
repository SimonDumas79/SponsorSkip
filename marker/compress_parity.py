"""Can the three fine-tuned models ship smaller without changing v3's answers? (2026-09-24)

The bundle's bge / edge_start / edge_resume checkpoints are 133 MB each in fp32. Two compressions,
each checked against the fp32 original on the SAME rows, through the same serving code (CPU, gated):
  fp16   weights stored in half precision, upcast to fp32 at load: half the size, same compute
  int8   dynamic int8 quantization of every Linear layer (CPU only): about a quarter of the size
Parity is judged on what the viewer gets: the regions v3 skips, video by video, and the grade.
Run on holdout 4 (already graded, so nothing here can leak into a decision on a fresh test set).

    python marker/compress_parity.py [holdout4] [--limit N]
"""

import copy
import ctypes
import io
import json
import sys

import numpy as np
import torch

from candidate import report
from edge_heads import place
from export_candidate import OUT, POTION
from replay import regions_by_video
from sbml_eval import load_preds
from serve_candidate import CandidateTier
from stack_sbml import sbml_stream
from train import DATA, Rows, load

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(6)

KINDS = ("bge", "edge_start", "edge_resume")


def size_mb(state: dict) -> float:
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.tell() / 1e6


def variant(model: torch.nn.Module, how: str) -> torch.nn.Module:
    if how == "fp32":
        return model
    if how == "fp16":
        m = copy.deepcopy(model)
        m.load_state_dict({k: v.half().float() if v.is_floating_point() else v for k, v in m.state_dict().items()})
        return m.eval()
    if how == "int8":
        return torch.ao.quantization.quantize_dynamic(copy.deepcopy(model), {torch.nn.Linear}, dtype=torch.qint8).eval()
    raise ValueError(how)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    which = args[0] if args else "holdout4"
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    rows = load(DATA / f"features_{which}.npz")
    pot = np.load(DATA / POTION[f"features_{which}.npz"])["X"]
    pot = pot[:, :pot.shape[1] - 2]
    text = {}
    with open(DATA / f"examples_{which}.jsonl", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                text[(r["videoID"], r["i"])] = r["text"]
    d = np.load(DATA / f"features_{which}.npz")
    texts = [text[(str(v), int(i))] for v, i in zip(d["video"], d["line"])]
    if limit:
        wanted = np.array([str(v) for v in dict.fromkeys(rows.video)][:limit])
        keep = np.isin(rows.video.astype(str), wanted)
        rows = Rows(rows.X[keep], rows.y[keep], rows.video[keep], rows.channel[keep],
                    rows.start_seconds[keep], rows.split[keep], rows.feature_names)
        pot, texts = pot[keep], [t for t, k in zip(texts, keep) if k]

    tier = CandidateTier(OUT, device="cpu", v3=True)
    cheap = tier.cheap_streams(rows, pot)
    mask = tier.gate_mask(cheap, rows) if tier.gated else None
    g = tier.gate or {}
    neutral = {k: float(g.get(n, 0.0)) if tier.gated else 0.0
               for k, n in zip(KINDS, ("neutral", "neutral_start", "neutral_end"))}
    for k in KINDS:   # load the fp32 originals once
        tier.ft_stream(k, texts[:1], rows.video[:1], None, 0.0)
    originals = {k: tier._bge[k] for k in KINDS}
    sbml = sbml_stream(rows, load_preds())
    th = tier.v3["thresholds"][tier.budget]
    share = 1.0 if mask is None else float(mask.mean())
    print(f"{which}: {rows.videos} videos, {rows.reads} reads; the fine-tuned models read {share:.0%} of lines\n", flush=True)

    ref_scores, ref_regions = None, None
    for how in ("fp32", "fp16", "int8"):
        mb = 0.0
        for k in KINDS:
            model, tok, opts = originals[k]
            m = variant(model, how)
            tier._bge[k] = (m, tok, opts)
            st = m.state_dict()
            if how == "fp16":
                st = {n: v.half() if v.is_floating_point() else v for n, v in st.items()}
            mb += size_mb(st)
        s = {k: tier.ft_stream(k, texts, rows.video, mask, neutral[k]) for k in KINDS}
        ctx, ps, pe = tier.score_v3(rows, {**cheap, "bge": s["bge"], "sbml": sbml})
        regions = place(regions_by_video(ctx, rows, th, 1), rows,
                        (ps + s["edge_start"]) / 2, (pe + s["edge_resume"]) / 2)
        print(f"== {how}: three models {mb:.0f} MB", flush=True)
        if ref_scores is None:
            ref_scores, ref_regions = s, regions
        else:
            for k in KINDS:
                diff = np.abs(s[k] - ref_scores[k])
                print(f"   {k:<12} score diff vs fp32: mean {diff.mean():.5f}  max {diff.max():.4f}", flush=True)
            vids = sorted(set(ref_regions) | set(regions))
            same = sum(ref_regions.get(v, []) == regions.get(v, []) for v in vids)
            print(f"   regions identical on {same} of {len(vids)} videos with any region", flush=True)
        report(f"v3 with {how} models", regions, rows)
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
