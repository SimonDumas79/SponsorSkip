"""End caps for level 4's runaway ends, graded on holdout 4 (already used) before holdout 6. (2026-09-24)

On holdout 5 the served whole-transcript reader skipped the most ad time (73.2%) but lost 28.7 s of show
per video, mostly from a few segments whose END ran on for minutes (worst 575 s). A flat 180 s cap would
also clip real reads: 23 of 558 labelled reads run longer (about 5% of ad time). The smarter rule caps a
segment only when Claude's resume quote -- the first words after the read -- is NOT found in the captions
near the end it gave, which is what a runaway end should look like.

Variants, on the same served answers (level4_eval.mjs output, with raw quotes):
  none           the answer as served
  flat N         every segment cut to at most N s
  unconfirmed N  cut to N s only when no overlapping raw segment's resume quote was found near its end

    python marker/level4_cap.py holdout4
"""

import json
import re
import sys

import numpy as np

from candidate import report
from grade_holdout5 import lost_by_video
from train import DATA, load

sys.stdout.reconfigure(encoding="utf-8")

norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", str(s).lower())).strip()


def resume_found(raw: dict, lines: list[dict], window: float = 15.0) -> bool:
    """Does the resume quote start in a caption line within `window` s of the raw end? (segments.mjs lineOf)"""
    words = norm(raw.get("resumeQuote") or "").split()[:4]
    if len(words) < 2:
        return False
    needle = " ".join(words)
    for i, ln in enumerate(lines):
        if abs(ln["start"] - raw["end"]) > window:
            continue
        here = norm(ln["text"])
        joined = f"{here} {norm(lines[i + 1]['text']) if i + 1 < len(lines) else ''}"
        at = joined.find(needle)
        if at != -1 and at <= len(here):
            return True
    return False


def capped(rec: dict, lines: list[dict], mode: str, cap: float) -> list[dict]:
    out = []
    for s in rec["segments"]:
        seg = dict(s)
        if mode != "none" and seg["end"] - seg["start"] > cap:
            if mode == "flat":
                seg["end"] = seg["start"] + cap
            else:
                raws = [r for r in rec.get("raw") or [] if r["start"] < seg["end"] and r["end"] > seg["start"]]
                if not any(resume_found(r, lines) for r in raws):
                    seg["end"] = seg["start"] + cap
        out.append(seg)
    return out


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "holdout4"
    T = load(DATA / f"features_{which}.npz")
    recs = {}
    for ln in (DATA / f"level4_{which}.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            recs[r["videoID"]] = r
    failed = [v for v, r in recs.items() if r.get("segments") is None]
    cost = sum(r.get("costUsd") or 0 for r in recs.values())
    print(f"{which}: {T.videos} videos, {T.reads} reads; {len(recs)} read, {len(failed)} failed, plan usage ${cost:.2f}\n")
    caps = {}
    for v in recs:
        caps[v] = json.loads((DATA / "captions" / f"{v}.json").read_text(encoding="utf-8"))["lines"]
    long_ = [(v, s) for v, r in recs.items() for s in (r.get("segments") or []) if s["end"] - s["start"] > 120]
    print(f"segments over 120 s: {len(long_)}")
    for mode, cap in [("none", 0), ("flat", 120), ("flat", 180), ("flat", 240), ("unconfirmed", 120),
                      ("unconfirmed", 180), ("unconfirmed", 240)]:
        kept = {}
        for v in np.unique(T.video):
            v = str(v)
            rec = recs.get(v)
            if not rec or rec.get("segments") is None:
                continue
            r = np.flatnonzero(T.video == v)
            t = T.start_seconds[r]
            spans = []
            for s in capped(rec, caps[v], mode, cap):
                inside = np.flatnonzero((t >= s["start"]) & (t < s["end"]))
                if len(inside):
                    spans.append((int(inside[0]), int(inside[-1]) + 1))
            if spans:
                kept[v] = spans
        name = "as served" if mode == "none" else f"{mode} {cap:.0f} s"
        report(name, kept, T)
        print(f"  {'':<44} worst single video {max(lost_by_video(kept, T)):.0f} s of show lost")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
