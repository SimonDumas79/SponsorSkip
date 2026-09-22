"""Simon's layered design: cheap models find the reads, an expensive one places their edges. (2026-09-22)

His words, 2026-09-22: "It should be layered, cheap quick model does the initial early sweep, then a
stronger cheap model to get the other markers down, then if selected, either local or Claude will
begin parsing out the sponsored sections."

The candidate is not layered. All six detectors read every line of every video and a context model
stacks them, so the most expensive part is paid everywhere, including the 90% of a video that is
obviously the show. Two separate things follow from his design, and they are worth measuring apart:

LAYER 1 -> 2, a COST question. Let the cheap detectors (the linear marker, meaning, structure,
potion, the sequence model) sweep first, and run the fine-tuned BGE only where they see something.
If the grade does not move, the gating is free speed. `--gate` measures this: BGE's real score inside
the gate, its median elsewhere, then the same stack and the same fixed thresholds.

LAYER 3, an ACCURACY question, and the one with headroom. Our edge heads place a start 2.2-2.4 s out;
Claude placed 6 of 6 within 2.2 s when handed a window and asked for a line number, and reading whole
videos it takes 81.9% of ad time at 5.3 s lost, against our 67%. So: keep our regions, and let Claude
say where each one starts and ends. `--edges` measures this. It is not Claude as a seventh detector,
which was measured on 2026-09-22 and did not help (64.4% against 64.6%); it is Claude placing the
edges of regions the cheap layers already found, which is the qwen tier's shape with a better model.

An answer is trusted only near the region it was asked about (`edge_heads.place_from_answers`), so a
wild line number falls back to the region rather than cutting somewhere random.

    python marker/cascade.py --edges --set holdout4 --model haiku
    python marker/cascade.py --gate  --set holdout4
"""

import argparse
import ctypes
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from candidate import POTION, report
from claude_label import ask_claude
from context_stack import context_features
from edge_heads import place, place_from_answers
from export_candidate import OUT
from qwen_edges import edge_window
from replay import regions_by_video
from serve_candidate import Candidate
from train import DATA, load

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(8)

ASK = """You are given numbered caption lines from a YouTube video. A sponsor read (an advertisement the host reads, interrupting the video's actual subject) is somewhere in these lines, and it has already been found by another system. Your job is ONLY to say exactly where it starts and where it ends.

Answer ONLY with JSON: {"start_line": <number>, "end_line": <number>}
  start_line = the FIRST line of the sponsor read: the line where the host stops talking about the video's subject and begins the ad. Include the turn into it ("but first", "this video is sponsored by"), not the line before it.
  end_line   = the first line where the actual show RESUMES: the line AFTER the ad's last line.
If these lines contain no sponsor read at all, answer {"start_line": null, "end_line": null}."""


def texts_for(which: str, video: np.ndarray, line: np.ndarray) -> list[str]:
    text = {}
    with open(DATA / f"examples_{which}.jsonl", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                text[(r["videoID"], r["i"])] = r["text"]
    return [text[(str(v), int(i))] for v, i in zip(video, line)]


def fine_tuned(c: Candidate, which: str, texts: list[str], video: np.ndarray):
    """The three fine-tuned streams from the bundle's checkpoints, cached: two minutes on the card
    each time otherwise, and every experiment here needs the same three."""
    out = []
    for name in ("bge", "edge_start", "edge_resume"):
        cache = DATA / f"cascade_stream_{name}__{which}.npy"
        if cache.exists():
            out.append(np.load(cache))
            continue
        s = c.ft_stream(name, texts, video)
        np.save(cache, s)
        out.append(s)
    return out


def parts(which: str, dev: str):
    fresh = f"features_{which}.npz"
    T = load(DATA / fresh)
    pot = np.load(DATA / POTION[fresh])["X"]
    pot = pot[:, :pot.shape[1] - 2]
    d = np.load(DATA / fresh)
    texts = texts_for(which, d["video"], d["line"])
    c = Candidate(OUT, device=dev)
    bge, fs, fe = fine_tuned(c, which, texts, T.video)
    return c, T, pot, texts, bge, fs, fe


def streams(which: str, dev: str):
    """The candidate's context and edge scores for a fresh set, from the shipped bundle."""
    c, T, pot, texts, bge, fs, fe = parts(which, dev)
    ctx, ps, pe = c.score(T, pot, bge)
    return c, T, texts, ctx, (ps + fs) / 2, (pe + fe) / 2


def run_edges(which: str, model: str, workers: int, budget: int, dev: str, local: bool = False) -> int:
    c, T, texts, ctx, ps, pe = streams(which, dev)
    lines_by_video = {}
    for v in np.unique(T.video):
        r = np.flatnonzero(T.video == v)
        lines_by_video[str(v)] = [{"text": texts[k], "start": float(T.start_seconds[k])} for k in r]

    th = c.thresholds_v2[budget]
    found = regions_by_video(ctx, T, th, 1)
    jobs = [(vid, int(lo), int(hi)) for vid, spans in found.items() for lo, hi in spans]
    print(f"{which}: {T.videos} videos, {T.reads} reads; the cheap layers found {len(jobs)} regions "
          f"at B={budget} (threshold {th:.4f}).", flush=True)

    who = "qwen" if local else f"claude-{model}"
    cache = DATA / f"cascade_edges_{which}_{who}_B{budget}.json"
    answers = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}

    def one(job):
        vid, lo, hi = job
        key = f"{vid}:{lo}-{hi}"
        if key in answers:
            return
        # int() is not decoration: lo and hi come out of numpy, so wlo is an int64, and json.dumps
        # refuses it. On 2026-09-22 that crash threw away 93 answered Claude calls at the cache write.
        wlo, whi, text = edge_window(lines_by_video[vid], int(lo), int(hi))
        wlo = int(wlo)
        if local:
            from qwen_edges import ASK as LOCAL_ASK, ask_edges
            a = ask_edges(f"{LOCAL_ASK}\n\n{text}")
            a = a if a is not None else {"_error": "ollama"}
        else:
            a = ask_claude(f"{ASK}\n\n{text}", model=model)
        if isinstance(a, dict) and "_error" in a:
            # A failed call must never read as "no read here": it is recorded as a failure and the
            # region keeps the heads' edges, rather than silently looking like a cautious answer.
            answers[key] = {"wlo": wlo, "start_line": None, "end_line": None, "failed": True}
        else:
            a = a or {}
            answers[key] = {"wlo": wlo, "start_line": a.get("start_line"), "end_line": a.get("end_line")}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, _ in enumerate(pool.map(one, jobs), 1):
            if n % 20 == 0:
                cache.write_text(json.dumps(answers, indent=1), encoding="utf-8")
                print(f"  {n}/{len(jobs)} regions", flush=True)
    cache.write_text(json.dumps(answers, indent=1), encoding="utf-8")

    failed = sum(1 for a in answers.values() if a.get("failed"))
    blank = sum(1 for a in answers.values() if not a.get("failed") and a.get("start_line") is None)
    placed, used = place_from_answers(found, answers, T)
    print(f"  {len(answers)} answers: {used} edges taken, {blank} said no read here, {failed} FAILED\n", flush=True)

    report(f"B={budget:>2} regions with our averaged heads", place(found, T, ps, pe), T)
    report(f"B={budget:>2} regions with {who} edges", placed, T)
    report(f"B={budget:>2} regions unplaced (raw)", found, T)
    return 0


def smooth(x: np.ndarray, video: np.ndarray, w: int) -> np.ndarray:
    """Rolling maximum over w lines either side, within a video: the cheap layers' reach."""
    out = np.zeros(len(x), dtype=np.float32)
    for v in np.unique(video):
        r = np.flatnonzero(video == v)
        a = x[r]
        out[r] = [a[max(0, i - w):i + w + 1].max() for i in range(len(a))]
    return out


def run_gate(which: str, budget: int, dev: str) -> int:
    """Layers 1 -> 2: how much of the video does the expensive detector actually need to read?"""
    c, T, pot, texts, bge, fs, fe = parts(which, dev)
    st = c.streams(T, pot, bge)

    # The cheap sweep: the five detectors that are not the fine-tuned one, at their strongest point
    # within 15 lines. A line is handed to the expensive model only if this clears the gate.
    cheap = np.max(np.column_stack([st[k] for k in ("marker", "meaning", "structure", "potion", "sequence")]), axis=1)
    reach = smooth(cheap, T.video, 15)
    neutral = float(np.median(bge))
    print(f"{which}: {T.videos} videos, {T.reads} reads. The expensive detector's score is kept inside the "
          f"gate and replaced by its median ({neutral:.3f}) outside it.\n", flush=True)

    full = c.score(T, pot, bge)
    report(f"B={budget:>2} no gate (all six read every line)",
           place(regions_by_video(full[0], T, c.thresholds_v2[budget], 1), T, (full[1] + fs) / 2, (full[2] + fe) / 2), T)
    for share in (0.50, 0.30, 0.20, 0.10, 0.05):
        cut = float(np.quantile(reach, 1 - share))
        gated = np.where(reach >= cut, bge, neutral)
        g = c.score(T, pot, gated)
        kept = place(regions_by_video(g[0], T, c.thresholds_v2[budget], 1), T, (g[1] + fs) / 2, (g[2] + fe) / 2)
        report(f"B={budget:>2} gate {share:.0%} of lines to the fine-tuned model", kept, T)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="holdout4")
    ap.add_argument("--edges", action="store_true", help="layer 3: Claude places the edges of the regions we found")
    ap.add_argument("--gate", action="store_true", help="layers 1->2: run the fine-tuned model only where the cheap ones see something")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", type=int, choices=[5, 10], default=10)
    ap.add_argument("--local", action="store_true", help="layer 3 on the local GPU (qwen) instead of Claude")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.edges:
        return run_edges(args.set, args.model, 1 if args.local else args.workers, args.budget, dev, args.local)
    if args.gate:
        return run_gate(args.set, args.budget, dev)
    raise SystemExit("choose --edges or --gate")


if __name__ == "__main__":
    raise SystemExit(main())
