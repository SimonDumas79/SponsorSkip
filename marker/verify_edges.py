"""Check every labelled read's edges with Claude, without showing it SponsorBlock's answer. (2026-09-22)

Simon's point: a SponsorBlock segment is set by hand against the video, and our line labels come from
mapping its seconds onto caption lines, so a read's labelled start and end can land off the true line
(caption timing drifts, and some marks are loose). The start/end models then learn the wrong line, and
the grading rewards matching the noise. Proposed: keep only edges that are verified, and use
Claude-verified ones where they disagree.

For each labelled read this cuts the caption lines from PAD lines before it to PAD lines after it,
numbers them, and asks Claude (claude -p, the subscription, no API credits) for the first line of the
promotional read and the first line where the show resumes. SponsorBlock's range is NOT shown. Claude's
answer is recorded beside SponsorBlock's, in lines and in seconds, to data/edge_verify.jsonl.

    python marker/verify_edges.py --workers 6              # every pooled read
    python marker/verify_edges.py --set channels           # the held-out channel set's reads
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from claude_label import ask_claude
from experiments import runs

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = DATA / "edge_verify.jsonl"
PAD = 20
MAX_LINES = 160
SETS = {"pooled": [("features.npz", "examples.jsonl"), ("features_tail.npz", "examples_tail.jsonl"),
                   ("features_holdout2.npz", "examples_holdout2.jsonl")],
        "channels": [("features_channels.npz", "examples_channels.jsonl")]}

ASK = """You are given numbered caption lines from part of a YouTube video. Somewhere in them there is probably a PROMOTIONAL READ, one of:
  - a sponsor read: an advertisement the host reads out for a company that paid them, or
  - self-promotion: the host pitching their own product, app, course, merch, membership, Patreon or event.
It interrupts the video's actual subject, and the show continues afterwards (possibly on a different topic).

Find the promotional read that is in these lines and give:
  start_line = the number of its FIRST line (the line where the host starts the promotion, including a lead-in such as "this video is sponsored by" or "but first").
  end_line   = the number of the first line AFTER it, where the show continues.
If it starts before line 0, answer 0. If it runs past the last line shown, answer the last line number plus one.
If there are several, give the longest one. If there is none, answer nulls.

Answer ONLY with JSON: {"start_line": <number or null>, "end_line": <number or null>}"""


def reads_of(set_name: str) -> list[dict]:
    out = []
    for feats, examples in SETS[set_name]:
        text = {}
        with (DATA / examples).open(encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    r = json.loads(ln)
                    text[(r["videoID"], r["i"])] = r["text"]
        d = np.load(DATA / feats)
        for vid in np.unique(d["video"]):
            k = np.flatnonzero(d["video"] == vid)
            y, starts, lines_i = d["y"][k], d["start_seconds"][k], d["line"][k]
            cat = d["category"][k]
            for n, (a, b) in enumerate(runs(y)):
                lo, hi = max(0, a - PAD), min(len(k), b + PAD)
                if hi - lo > MAX_LINES:   # a very long read: keep the edges in view, drop the middle
                    continue
                out.append({"video": str(vid), "read": n, "lo": int(lo), "hi": int(hi), "sb_start": int(a), "sb_end": int(b),
                            "category": "+".join(sorted(set(cat[a:b].tolist()) - {""})),
                            "seconds": [float(x) for x in starts[lo:hi + 1]] if hi < len(k) else
                                       [float(x) for x in starts[lo:hi]] + [float(starts[hi - 1] + 2.2)],
                            "text": [text[(str(vid), int(i))] for i in lines_i[lo:hi]]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(SETS), default="pooled")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    done = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as f:
            done = {(r["video"], r["read"]) for r in map(json.loads, f) if r.get("claude_start") is not None or r.get("none")}
    todo = [r for r in reads_of(args.set) if (r["video"], r["read"]) not in done]
    print(f"{args.set}: {len(todo)} reads to check with {args.model}", flush=True)
    lock = threading.Lock()
    counter = {"n": 0}

    def one(r: dict) -> None:
        rendered = "\n".join(f"{i:3d}: {t}" for i, t in enumerate(r["text"]))
        ans = ask_claude(f"{ASK}\n\n{rendered}", model=args.model, timeout=180)
        width = r["hi"] - r["lo"]
        s = e = None
        if ans and isinstance(ans.get("start_line"), int) and isinstance(ans.get("end_line"), int):
            s, e = max(0, min(ans["start_line"], width)), max(0, min(ans["end_line"], width))
            if e <= s:
                s = e = None
        sec = r["seconds"]
        rec = {"video": r["video"], "read": r["read"], "set": args.set, "category": r["category"],
               "sb_start": r["sb_start"], "sb_end": r["sb_end"], "lo": r["lo"], "hi": r["hi"],
               "claude_start": None if s is None else r["lo"] + s, "claude_end": None if e is None else r["lo"] + e,
               "none": ans is not None and ans.get("start_line") is None,
               "d_start_s": None if s is None else sec[s] - sec[r["sb_start"] - r["lo"]],
               "d_end_s": None if e is None else sec[e] - sec[r["sb_end"] - r["lo"]]}
        with lock:
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            counter["n"] += 1
            if counter["n"] % 20 == 0 or counter["n"] == len(todo):
                print(f"  [{counter['n']}/{len(todo)}]", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, todo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
