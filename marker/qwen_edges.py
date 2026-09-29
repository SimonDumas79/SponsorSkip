"""Ask qwen3 WHERE each confirmed read starts and ends, as caption line numbers, and record it.

Detection is solved well enough; the edges are not. On cross-validation the
marker's own region edges leave most of each read playing, or else skip real
show around it. Claude, handed a window of numbered caption lines and asked for
the first line of the read and the line where the show resumes, lands within
1-2 s. qwen3 was only ever tested on yes/no edge questions ("does an ad begin
here?"), which it answered with long tails. This script asks it the line-number
question instead, the same one the level-3 cascade asks Claude (serve_candidate.py's _model_edges).

For every region a confirm_check.py run recorded, where qwen said yes or the
marker was near-certain, it cuts a wider window (the region plus REACH lines
either side, capped at MAX_LINES) and asks for {"start_line", "end_line"}.
Every answer is written to <run>.edges.jsonl with the window's position, so the
edge rules can be replayed and graded without asking again. An answer outside
the window is recorded as it is; the replay decides what to trust.

    python marker/qwen_edges.py --verdicts marker/data/confirm_verdicts.jsonl
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from confirm_check import MODEL, OLLAMA, unload
from replay import load_verdicts
from train import DATA, load

REACH = 25       # caption lines of context either side of the flagged region
MAX_LINES = 120  # about five minutes of captions

ASK = """You are given numbered caption lines from part of a video. Somewhere inside them there is a promotional read: an advertisement the host reads out for a sponsor, or the host promoting their own product, app, course, merch or membership.

Answer ONLY with JSON: {"start_line": <number>, "end_line": <number>}
  start_line = the number of the FIRST line of the promotional read.
  end_line   = the number of the first line where the actual show RESUMES after it.
If there is no promotional read at all, answer {"start_line": null, "end_line": null}."""

SCHEMA = {"type": "object", "properties": {"start_line": {"type": ["integer", "null"]},
                                           "end_line": {"type": ["integer", "null"]}},
          "required": ["start_line", "end_line"]}


def ask_edges(prompt: str, timeout: float = 180.0) -> dict | None:
    body = json.dumps({
        "model": MODEL, "stream": False, "think": False, "format": SCHEMA, "keep_alive": "300s",
        "options": {"num_ctx": 8192, "temperature": 0},
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(json.load(r)["message"]["content"])
    except (urllib.error.URLError, KeyError, ValueError, TimeoutError):
        return None


def edge_window(lines: list[dict], lo: int, hi: int) -> tuple[int, int, str]:
    """(first line, end line, numbered text) of the window qwen places edges in for region [lo, hi)."""
    wlo, whi = max(0, lo - REACH), min(len(lines), hi + REACH)
    if whi - wlo > MAX_LINES:
        mid = (lo + hi) // 2
        wlo, whi = max(0, mid - MAX_LINES // 2), min(len(lines), mid + MAX_LINES // 2)
    return wlo, whi, "\n".join(f"{i - wlo:3d}: {' '.join(lines[i]['text'].split())}" for i in range(wlo, whi))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdicts", type=Path, default=DATA / "confirm_verdicts.jsonl")
    ap.add_argument("--peak", type=float, default=0.995, help="also ask about regions the marker is this sure of")
    args = ap.parse_args()

    manifest_path = args.verdicts.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    rows = load(Path(manifest.get("features", DATA / "features.npz")))
    captions = Path(manifest.get("captions", DATA / "captions"))
    out_path = args.verdicts.with_suffix(".edges.jsonl")
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}

    todo = [v for v in load_verdicts(args.verdicts).values()
            if (v["said"] is True or v["marker_peak"] >= args.peak) and v["id"] not in done]
    print(f"{len(todo)} confirmed regions to ask {MODEL} for edges ({len(done)} already recorded)", flush=True)

    caps_cache: dict[str, list] = {}
    t0 = time.time()
    try:
        with out_path.open("a", encoding="utf-8") as f:
            for n, v in enumerate(todo, 1):
                lines = caps_cache.setdefault(v["video"], json.loads(
                    (captions / f"{v['video']}.json").read_text(encoding="utf-8"))["lines"])
                wlo, whi, text = edge_window(lines, v["lo"], v["hi"])
                answer = ask_edges(f"{ASK}\n\n{text}")
                rec = {"id": v["id"], "video": v["video"], "lo": v["lo"], "hi": v["hi"], "wlo": wlo, "whi": whi,
                       "start_line": answer.get("start_line") if answer else None,
                       "end_line": answer.get("end_line") if answer else None, "answered": answer is not None}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if n % 10 == 0 or n == len(todo):
                    rate = (time.time() - t0) / n
                    print(f"  [{n}/{len(todo)}] {rate:.1f} s per window, ~{rate * (len(todo) - n) / 60:.0f} min left",
                          flush=True)
    finally:
        unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
