"""Two fixes for the local model's edge placement, both from Simon's questions. (2026-09-22)

The first measurement of qwen3:8b placing edges (cascade.py --edges --local) put it at 62.0% of ad
time against Claude's 71.2%. That comparison was NOT clean, and Simon caught it: the two models got
different prompts. Claude was asked with cascade.ASK, which spells out that the start is where the
host stops talking about the subject and that the turn into the ad ("but first", "this video is
sponsored by") belongs inside. qwen was asked with qwen_edges.ASK, which just says "the FIRST line of
the promotional read". So part of the gap may be the instruction, not the model.

  --same-prompt   qwen, asked exactly what Claude was asked. Separates model from prompt.

The second is Simon's walk idea, put where it might actually fit. It was rejected for our own edge
heads because the heads already read both sides of a candidate edge at once, so a sequential walk
only threw information away. A small language model has the opposite problem: asking it for a LINE
NUMBER makes it count lines, which is exactly what small models are bad at, and qwen's behaviour fits
that -- it accepted 70 of 93 offered edges (more than Claude's 59) and placed them short, covering
only 18% of reads fully against Claude's 46%. A walk asks no arithmetic at all. It shows a small
group of lines and asks one yes/no question: is this still the advertisement? Then it steps outward.

  --walk          from inside the region, step out in groups of GROUP lines until PATIENCE groups in
                  a row come back "no". No indices, no counting, one binary judgement at a time.

What the record says qwen is good at, for context: as a DETECTOR it finds 22.7% of ad time no other
detector finds, and alone takes 70.4% of ad time at 50 s of show lost per video. It is a wide net
with poor precision -- good at finding, bad at bounding. Edges are the opposite job, which is why
this is worth checking rather than assuming.

    python marker/local_edges.py --same-prompt
    python marker/local_edges.py --walk --group 3 --patience 2
"""

import argparse
import ctypes
import json
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from candidate import POTION, report
from cascade import ASK as CLAUDE_ASK, parts
from confirm_check import MODEL as OLLAMA_MODEL, OLLAMA, SCHEMA
from edge_heads import AFTER, BEFORE, INSIDE, TAIL, place
from qwen_edges import edge_window
from replay import regions_by_video
from train import DATA

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(8)

WALK_ASK = """These numbered caption lines come from the middle of a YouTube video.

An advertisement that the host reads for a sponsor is running nearby. Look ONLY at the lines marked >>> and answer one question about them: are those lines still part of that advertisement, or are they the video's actual subject (the show)?

The advertisement includes the host's turn into it ("but first", "this video is sponsored by") and everything until the show resumes.

Answer ONLY with JSON: {"promo": true} if the marked lines are still the advertisement, {"promo": false} if they are the show."""

# Simon's diagnosis, 2026-09-22: the walk asks "is this still the show" without ever telling the
# model what the show IS. A generic transitional line ("so yeah, anyway") gives it nothing to check
# resumption against. This grounds the question in the video's actual subject: the title, and (in the
# --anchor-lines variant) a sample of the show's own voice from well before the ad.
WALK_ASK_ANCHORED = """These numbered caption lines come from the middle of a YouTube video titled "{title}".

An advertisement that the host reads for a sponsor is running nearby. Look ONLY at the lines marked >>> and answer one question about them: do they belong to the video's actual subject -- what the title describes -- or are they still the advertisement?

The advertisement includes the host's turn into it ("but first", "this video is sponsored by") and everything until the host returns to that subject.
{extra}
Answer ONLY with JSON: {{"promo": true}} if the marked lines are still the advertisement, {{"promo": false}} if they have returned to the video's subject."""

MAX_WALK = 30   # default; --max-walk overrides. How far past the region a walk may reach.


def ask_bool_thinking(prompt: str, timeout: float = 180.0):
    """Like confirm_check.ask_bool, but with reasoning ON. The walk asks a genuinely nuanced call
    ("has the show resumed") and confirm_check.ask_bool forces an answer with think=False -- a
    setting built for a cheaper yes/no (region confirmation), not this one. Returns (answer, the
    reasoning text) so a run can be inspected instead of only graded.
    """
    body = json.dumps({
        "model": OLLAMA_MODEL, "stream": False, "think": True, "format": SCHEMA, "keep_alive": "300s",
        "options": {"num_ctx": 4096, "temperature": 0},
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            msg = json.load(r)["message"]
            return bool(json.loads(msg["content"])["promo"]), (msg.get("thinking") or "")[:600]
    except (urllib.error.URLError, KeyError, ValueError, TimeoutError):
        return None, None


def window(lines: list[dict], lo: int, hi: int, a: int, b: int) -> str:
    """The neighbourhood, with lines [a, b) marked as the ones being asked about."""
    wlo, whi = max(0, min(a, lo) - 12), min(len(lines), max(b, hi) + 12)
    out = []
    for i in range(wlo, whi):
        mark = ">>>" if a <= i < b else "   "
        out.append(f"{mark} {' '.join(lines[i]['text'].split())}")
    return "\n".join(out)


def walk_region(lines: list[dict], lo: int, hi: int, group: int, patience: int, ask,
                max_walk: int = MAX_WALK) -> tuple[int, int]:
    """Step outward from the region in groups of `group` lines while the model still says 'advert'."""
    a, misses = lo, 0
    while a > 0 and lo - a < max_walk:
        nxt = max(0, a - group)
        if ask(window(lines, lo, hi, nxt, a)):
            a, misses = nxt, 0
        else:
            misses += 1
            if misses >= patience:
                break
            a = nxt
    if misses:
        a = min(lo, a + misses * group)   # give back the groups the walk ate on its way to stopping
    b, misses = hi, 0
    while b < len(lines) and b - hi < max_walk:
        nxt = min(len(lines), b + group)
        if ask(window(lines, lo, hi, b, nxt)):
            b, misses = nxt, 0
        else:
            misses += 1
            if misses >= patience:
                break
            b = nxt
    if misses:
        b = max(hi, b - misses * group)
    return max(0, a), min(len(lines), max(b, a + 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="holdout4")
    ap.add_argument("--budget", type=int, choices=[5, 10], default=10)
    ap.add_argument("--same-prompt", action="store_true", help="ask qwen exactly what Claude was asked")
    ap.add_argument("--walk", action="store_true", help="walk outward with yes/no questions, no line numbers")
    ap.add_argument("--group", type=int, default=3)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4,
                    help="regions in parallel. A walk is sequential within a region, so this is the "
                         "only way to keep the card busy; ollama queues anything it cannot fit.")
    ap.add_argument("--max-walk", type=int, default=MAX_WALK)
    ap.add_argument("--tag", default="", help="names the cache, so settings do not overwrite each other")
    ap.add_argument("--anchor", action="store_true",
                    help="ground the walk's question in the video's title, instead of asking it to "
                         "judge 'is this the show' from local text alone")
    ap.add_argument("--anchor-lines", type=int, default=0,
                    help="also show N lines from the video's own opening as a sample of its subject "
                         "and tone (0 = title only)")
    ap.add_argument("--anchor-pre", type=int, default=0,
                    help="Simon's refinement: instead of the opening, anchor on N lines from just "
                         "before the ad started -- the show's actual topic at the moment it was "
                         "interrupted, which a long or multi-part video's title may not describe")
    ap.add_argument("--think", action="store_true",
                    help="let the walk's yes/no calls reason before answering (think=True), instead "
                         "of the setting confirm_check.ask_bool uses for cheaper region confirmation. "
                         "Writes a trace of every question and answer, so a run can be READ, not only "
                         "graded.")
    args = ap.parse_args()
    if not (args.same_prompt or args.walk):
        raise SystemExit("choose --same-prompt or --walk")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c, T, pot, texts, bge, fs, fe = parts(args.set, dev)
    ctx, ps, pe = c.score(T, pot, bge)
    ps, pe = (ps + fs) / 2, (pe + fe) / 2
    th = c.thresholds_v2[args.budget]
    found = regions_by_video(ctx, T, th, 1)
    by_heads = place(found, T, ps, pe)
    starts = {str(v): int(np.flatnonzero(T.video == v)[0]) for v in np.unique(T.video)}
    all_lines = [{"text": t, "start": float(s)} for t, s in zip(texts, T.start_seconds)]

    from confirm_check import ask_bool, unload
    from qwen_edges import ask_edges
    tag = "walk" if args.walk else "same prompt"
    stem = ('walk' if args.walk else 'sameprompt') + (f"_{args.tag}" if args.tag else "")
    cache = DATA / f"local_edges_{args.set}_{stem}_B{args.budget}.json"
    saved = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    n_regions = sum(len(s) for s in found.values())
    print(f"{args.set}: {T.videos} videos, {T.reads} reads; {n_regions} regions at B={args.budget}. "
          f"qwen3:8b, {tag}.", flush=True)

    # Regions are independent, so they run in parallel. A WALK is sequential inside one region (each
    # step depends on the last answer), which left the card idle at 0-1% between calls; the way to
    # use it is several regions at once. Ollama queues what it cannot fit, so this degrades safely.
    titles = {}
    if args.anchor:
        for vid in found:
            p = DATA / "captions" / f"{vid}.json"
            if p.exists():
                titles[vid] = json.loads(p.read_text(encoding="utf-8")).get("title") or "(untitled)"

    jobs = []
    for vid, spans in found.items():
        base = starts[vid]
        n = int((T.video == vid).sum())
        heads = list(by_heads.get(vid, []))
        for k, (lo, hi) in enumerate(spans):
            lo, hi = int(lo), int(hi)
            back = tuple(int(x) for x in heads[k]) if k < len(heads) else (lo, hi)
            jobs.append((vid, base, n, lo, hi, back))

    out, done = {}, 0
    calls = [0]
    lock = threading.Lock()
    trace_f = None
    if args.think:
        trace_path = DATA / f"local_edges_trace_{args.set}_{stem}_B{args.budget}.jsonl"
        trace_f = trace_path.open("a", encoding="utf-8")
        print(f"  tracing every question/answer to {trace_path}", flush=True)

    def run(job):
        vid, base, n, lo, hi, back = job
        key = f"{vid}:{lo}-{hi}"
        if key in saved:
            return vid, key, tuple(saved[key])
        lines = all_lines[base:base + n]
        if args.walk:
            if args.anchor:
                extra = ""
                if args.anchor_pre:
                    # 5-line buffer before lo, so the sample does not clip the ad's own lead-in.
                    pre_hi = max(0, lo - 5)
                    pre_lo = max(0, pre_hi - args.anchor_pre)
                    sample = " ".join(l["text"] for l in lines[pre_lo:pre_hi])
                    if sample:
                        # Simon's second point, 2026-09-22: a host often places the ad exactly at a
                        # topic break ("before we get into X, this is sponsored by Y" ... "now, X").
                        # If the marked lines require matching THIS sample's subject, a real return
                        # to a NEW topic would wrongly look like it is still the ad. The sample is
                        # therefore framed as tone/register only -- how the host talks when hosting --
                        # and the instruction says outright that the topic may have moved on.
                        extra = (
                            f'\nHere is a sample of the host\'s own voice from just before the ad, so '
                            f'you know what genuine hosting sounds like in this video (NOT necessarily '
                            f'the same subject the marked lines are about -- the show may move on to a '
                            f'new topic once the ad ends, and that is still the show): "{sample}"\n')
                elif args.anchor_lines:
                    opening = " ".join(l["text"] for l in lines[:args.anchor_lines])
                    extra = f'\nHere is a sample of the show, from its opening: "{opening}"\n'
                base_ask = WALK_ASK_ANCHORED.format(title=titles.get(vid, "(untitled)"), extra=extra)
            else:
                base_ask = WALK_ASK

            def ask(text: str, base_ask=base_ask, vid=vid, lo=lo, hi=hi) -> bool:
                with lock:
                    calls[0] += 1
                full = base_ask + "\n\n" + text
                if args.think:
                    result, reasoning = ask_bool_thinking(full)
                    if trace_f:
                        with lock:
                            trace_f.write(json.dumps({"video": vid, "region": [lo, hi], "window": text,
                                                      "reasoning": reasoning, "answer": result}) + "\n")
                            trace_f.flush()
                    return bool(result)
                return bool(ask_bool(full))
            span = walk_region(lines, lo, hi, args.group, args.patience, ask, args.max_walk)
        else:
            wlo, _, text = edge_window(lines, lo, hi)
            with lock:
                calls[0] += 1
            a = ask_edges(CLAUDE_ASK + "\n\n" + text)
            span = back
            if isinstance(a, dict) and a.get("start_line") is not None and a.get("end_line") is not None:
                s0 = min(max(int(wlo) + int(a["start_line"]), 0), n - 1)
                e0 = min(max(int(wlo) + int(a["end_line"]), s0 + 1), n)
                if lo - BEFORE <= s0 <= lo + INSIDE and hi - TAIL <= e0 <= hi + AFTER:
                    span = (s0, e0)
        return vid, key, (int(span[0]), int(span[1]))

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for vid, key, span in pool.map(run, jobs):
                saved[key] = list(span)
                out.setdefault(vid, []).append(span)
                done += 1
                if done % 10 == 0:
                    cache.write_text(json.dumps(saved, indent=1), encoding="utf-8")
                    print(f"  {done}/{n_regions} regions, {calls[0]} model calls", flush=True)
    finally:
        cache.write_text(json.dumps(saved, indent=1), encoding="utf-8")
        if trace_f:
            trace_f.close()
        unload()   # never leave qwen in VRAM

    print(f"  {done} regions, {calls} model calls\n", flush=True)
    report(f"B={args.budget:>2} our averaged heads", by_heads, T)
    report(f"B={args.budget:>2} qwen, {tag}", out, T)
    report(f"B={args.budget:>2} regions unplaced (raw)", found, T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
