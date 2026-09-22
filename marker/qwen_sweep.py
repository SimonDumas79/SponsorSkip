"""qwen3:8b as a DETECTOR on its own: every window of every pooled video, recorded once. (2026-09-22)

Until now qwen was only ever asked about windows the marker had already flagged (confirm_check.py)
or windows centred on known reads (union_check.py), so its detections could not be compared with
the regex and the trained models on total ad time. This sweeps every video in windows of WINDOW
caption lines (STRIDE apart, so neighbours overlap a little), asks the confirmer's question plus
where the read starts and ends, and records each answer to data/qwen_sweep.jsonl. overlap.py turns
the answers into skipped lines and compares them with everything else.

Resumable: windows already answered are skipped. Gentle: the script pauses and unloads qwen whenever
a game is running or something else is using the GPU, and unloads it when it finishes.

    python -u marker/qwen_sweep.py                 # all 205 pooled videos
    python -u marker/qwen_sweep.py --limit 20      # a timing run
"""

import argparse
import ctypes
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

from confirm_check import MODEL, OLLAMA, unload
from train import DATA

WINDOW, STRIDE = 45, 40
OUT = DATA / "qwen_sweep.jsonl"
SETS = [("examples.jsonl", "features.npz"), ("examples_tail.jsonl", "features_tail.npz"),
        ("examples_holdout2.jsonl", "features_holdout2.npz")]

ASK = """You are given numbered caption lines from part of a video.

Is there a PROMOTIONAL READ in these lines? That means either:
  - a sponsor read: an advertisement the host reads out for a company that paid them, or
  - self-promotion: the host pitching their own app, course, merch, membership, Patreon or event.
Either one interrupts the video's actual subject, and the subject resumes afterwards.

NOT a promotional read: a review, unboxing or demonstration of a product that IS the video's
subject; a plain "like and subscribe"; mentioning a previous video.

If there is one, also give the number of its first line and of its last line. If it runs past the
last line shown, give the last line shown; if it began before the first line shown, give 0.

Answer ONLY with JSON: {"promo": true, "start_line": N, "end_line": M} or
{"promo": false, "start_line": null, "end_line": null}"""

SCHEMA = {"type": "object", "properties": {
    "promo": {"type": "boolean"},
    "start_line": {"type": ["integer", "null"]},
    "end_line": {"type": ["integer", "null"]}}, "required": ["promo", "start_line", "end_line"]}


def ask(prompt: str, timeout: float = 120.0) -> dict | None:
    body = json.dumps({
        "model": MODEL, "stream": False, "think": False, "format": SCHEMA, "keep_alive": "120s",
        "options": {"num_ctx": 4096, "temperature": 0},
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(json.load(r)["message"]["content"])
    except (urllib.error.URLError, KeyError, ValueError, TimeoutError):
        return None


def someone_else_needs_the_gpu() -> str | None:
    """A game, or another program keeping the GPU busy while qwen is idle between calls."""
    try:
        tasks = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True,
                               creationflags=0x08000000).stdout.lower()
        names = [ln.split('","')[0].strip('"') for ln in tasks.splitlines() if ln]
        launchers = ("eosoverlayrenderer", "epicwebhelper", "unrealcefsubprocess", "crashreportclient")
        for proc in names:
            if proc.startswith(launchers):
                continue
            if any(g in proc for g in ("valorant", "-win64-shipping", "cs2.exe", "fortnite", "overwatch", "eldenring")):
                return f"a game is running ({proc})"
        util = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, creationflags=0x08000000).stdout.strip()
        if util and int(util.splitlines()[0]) > 50:
            return f"the GPU is {util.splitlines()[0]}% busy with something else"
    except Exception:
        return None
    return None


def windows() -> list[dict]:
    """Every window of every pooled video, in the pooled row order (row-local line indices)."""
    out = []
    for examples, features in SETS:
        text = {}
        with (DATA / examples).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    text[(r["videoID"], r["i"])] = r["text"]
        d = np.load(DATA / features)
        video, line_i = d["video"], d["line"]
        for vid in np.unique(video):
            idx = np.flatnonzero(video == vid)
            lines = [text[(str(vid), int(line_i[k]))] for k in idx]
            starts = list(range(0, max(1, len(lines) - WINDOW + STRIDE), STRIDE))
            for lo in starts:
                hi = min(len(lines), lo + WINDOW)
                out.append({"video": str(vid), "lo": lo, "hi": hi, "lines": lines[lo:hi]})
                if hi == len(lines):
                    break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many NEW windows (0 = all)")
    args = ap.parse_args()
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)

    todo = windows()
    done = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as f:
            done = {(r["video"], r["lo"]) for r in map(json.loads, f) if r.get("promo") is not None}
    todo = [w for w in todo if (w["video"], w["lo"]) not in done]
    print(f"{len(done)} windows already answered, {len(todo)} to ask", flush=True)
    started, asked, yes = time.time(), 0, 0
    try:
        with OUT.open("a", encoding="utf-8") as out:
            for n, w in enumerate(todo, 1):
                if n % 20 == 1:
                    why = someone_else_needs_the_gpu()
                    while why:
                        unload()
                        print(f"  paused: {why}; checking again in 5 min", flush=True)
                        time.sleep(300)
                        why = someone_else_needs_the_gpu()
                rendered = "\n".join(f"{i:3d}: {t}" for i, t in enumerate(w["lines"]))
                a = ask(f"{ASK}\n\n{rendered}")
                rec = {"video": w["video"], "lo": w["lo"], "hi": w["hi"],
                       "promo": None if a is None else bool(a.get("promo")),
                       "start_line": None if a is None else a.get("start_line"),
                       "end_line": None if a is None else a.get("end_line")}
                out.write(json.dumps(rec) + "\n")
                out.flush()
                asked += 1
                yes += bool(rec["promo"])
                if n % 100 == 0 or n == len(todo):
                    rate = (time.time() - started) / asked
                    print(f"  [{n}/{len(todo)}] {yes} said yes; {rate:.2f} s per window, "
                          f"~{rate * (len(todo) - n) / 60:.0f} min left", flush=True)
                if args.limit and asked >= args.limit:
                    break
    except KeyboardInterrupt:
        print("stopped by hand", flush=True)
    finally:
        unload()
    print(f"asked {asked} windows in {(time.time() - started) / 60:.1f} min; qwen unloaded", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
