"""Grade level 4 (level4_eval.mjs output: the served Claude reader) on a holdout, like every other system.

A segment covers the caption lines whose start falls inside it. A failed read counts as nothing skipped;
in the extension it would fall back to SponsorBlock's answer, which is reported separately.

    python marker/grade_level4.py holdout5
"""

import json
import sys

import numpy as np

from candidate import report
from grade_holdout5 import lost_by_video
from train import DATA, load

sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "holdout5"
    T = load(DATA / f"features_{which}.npz")
    recs = {}
    for ln in (DATA / f"level4_{which}.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            recs[r["videoID"]] = r
    kept = {}
    for v in np.unique(T.video):
        v = str(v)
        r = np.flatnonzero(T.video == v)
        t = T.start_seconds[r]
        spans = []
        for s in (recs.get(v, {}).get("segments") or []):
            inside = np.flatnonzero((t >= s["start"]) & (t < s["end"]))
            if len(inside):
                spans.append((int(inside[0]), int(inside[-1]) + 1))
        if spans:
            kept[v] = spans
    failed = [v for v, r in recs.items() if r.get("segments") is None]
    print(f"{which}: {T.videos} videos, {T.reads} reads; {len(recs)} read by Claude, {len(failed)} failed")
    report("I  level 4: Claude reads everything (served)", kept, T)
    print(f"  {'':<44} worst single video {max(lost_by_video(kept, T)):.0f} s of show lost")
    secs = [r["seconds"] for r in recs.values() if r.get("segments") is not None]
    cost = sum(r.get("costUsd") or 0 for r in recs.values())
    print(f"  {'':<44} median {np.median(secs):.0f} s per video (max {max(secs)} s); plan usage ${cost:.2f} in all")
    json.dump({v: [list(x) for x in s] for v, s in kept.items()}, open(DATA / f"level4_{which}_regions.json", "w"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
