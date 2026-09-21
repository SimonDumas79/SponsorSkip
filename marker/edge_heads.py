"""Place each read's edges with models trained for exactly that: where a read STARTS, where the show RESUMES.

The third separate job in the system. Detection (the marker) says roughly where a
read is; confirmation (the region judge, or qwen) says whether it is real; this
says where to cut. Two small second-stage models read the same context as
context_stack.py (the marker's logits for 15 lines either side, the seam, the
cues) and learn two different labels from build_dataset.py: is_start, the first
line of a read, and is_resume, the first line after it. Both are trained
out-of-fold with the marker's own channel folds.

Then, for each region the system decides to skip, the start goes to the line
with the highest start probability near the region's beginning, and the end to
the line with the highest resume probability near its end. Graded by the share
of ad time skipped and the real show lost, against the region's own edges and
(once qwen_edges.py has run) against qwen's line numbers.

    python marker/edge_heads.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from context_stack import context_features, fit_stage2
from replay import grade_regions, load_verdicts, regions_by_video, row, HEADER
from region_judge import WITHOUT_QWEN, fit_judge, out_of_fold as judge_oof, region_table
from train import DATA, load

BEFORE, INSIDE = 20, 10   # the start is looked for from 20 lines before the region to 10 lines into it
TAIL, AFTER = 10, 20      # the end from 10 lines before the region's end to 20 lines after it


def soft(labels: np.ndarray, video: np.ndarray) -> np.ndarray:
    """An edge label on its line and the lines either side: one line off is nearly as good as exact."""
    out = labels.astype(np.float32).copy()
    for shift in (-1, 1):
        moved = np.roll(labels, shift).astype(np.float32)
        same_video = np.roll(video, shift) == video
        out = np.maximum(out, moved * same_video)
    return out


def edge_oof(rows, F, target: np.ndarray, folds: int = 5, seed: int = 0) -> np.ndarray:
    channels = np.array(sorted(set(rows.channel)))
    np.random.default_rng(seed).shuffle(channels)
    p = np.zeros(len(rows), dtype=np.float32)
    for fold in np.array_split(channels, folds):
        test = np.isin(rows.channel, fold)
        p[test] = fit_stage2(F[~test], target[~test], hidden=32)(F[test])
    return p


def place(kept: dict[str, list[tuple[int, int]]], rows, p_start: np.ndarray, p_end: np.ndarray):
    """Move each kept region's edges to the most likely start and resume lines nearby."""
    placed = {}
    for vid, spans in kept.items():
        r = np.flatnonzero(rows.video == vid)
        ps, pe = p_start[r], p_end[r]
        for lo, hi in spans:
            a0, a1 = max(0, lo - BEFORE), min(len(r), lo + INSIDE)
            start = a0 + int(np.argmax(ps[a0:a1]))
            b0, b1 = max(start + 1, hi - TAIL), min(len(r), hi + AFTER + 1)
            end = b0 + int(np.argmax(pe[b0:b1])) if b1 > b0 else hi
            placed.setdefault(vid, []).append((start, max(end, start + 1)))
    return placed


def qwen_placed(kept, edges_path: Path, rows):
    """qwen's line numbers, translated from window positions to video lines; region edges where it gave none."""
    answers = {}
    if edges_path.exists():
        for line in edges_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                answers[rec["id"]] = rec
    placed, used = {}, 0
    for vid, spans in kept.items():
        n = int((rows.video == vid).sum())
        for lo, hi in spans:
            a = answers.get(f"{vid}:{lo}-{hi}")
            if a and a["start_line"] is not None and a["end_line"] is not None:
                start = min(max(a["wlo"] + int(a["start_line"]), 0), n - 1)
                end = min(max(a["wlo"] + int(a["end_line"]), start + 1), n)
                # Trust it only near the region the marker found: a wild answer falls back to the region.
                if lo - BEFORE <= start <= lo + INSIDE and hi - TAIL <= end <= hi + AFTER:
                    placed.setdefault(vid, []).append((start, end))
                    used += 1
                    continue
            placed.setdefault(vid, []).append((lo, hi))
    return placed, used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    torch.set_num_threads(4)

    verdicts_path = DATA / "confirm_verdicts.jsonl"
    rows = load(DATA / "features.npz")
    from experiments import BASE, cross_validate
    level1 = cross_validate(BASE, rows, seed=0)   # the marker the recorded regions came from
    d = np.load(DATA / "features.npz")
    F = context_features(rows, level1)
    p_start = edge_oof(rows, F, soft(d["is_start"], rows.video))
    p_end = edge_oof(rows, F, soft(d["is_resume"], rows.video))
    np.save(DATA / "edge_heads_oof.npy", np.stack([p_start, p_end]))

    verdicts = load_verdicts(verdicts_path)
    RF, info = region_table(rows, level1, 0.84, 3, verdicts)
    y = np.array([i["is_read"] for i in info], dtype=int)
    judge = judge_oof(RF, y, info, WITHOUT_QWEN)

    def kept_by(select):
        out = {}
        for i, f, pj in zip(info, RF, judge):
            if select(f, pj):
                out.setdefault(i["video"], []).append((i["lo"], i["hi"]))
        return out

    print(HEADER)
    qwen_rule = kept_by(lambda f, pj: f[10] == 1.0 or f[0] >= 0.995)
    for name, kept in [("qwen rule", qwen_rule)] + [(f"judge p>={t}", kept_by(lambda f, pj, t=t: pj >= t))
                                                     for t in (0.5, 0.7, 0.8, 0.9, 0.95)]:
        print(row(f"{name}: region edges", grade_regions(kept, rows)))
        print(row(f"{name}: edge heads", grade_regions(place(kept, rows, p_start, p_end), rows)))
        if name == "qwen rule":
            placed, used = qwen_placed(kept, verdicts_path.with_suffix(".edges.jsonl"), rows)
            if used:
                print(row(f"{name}: qwen edges ({used} placed)", grade_regions(placed, rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
