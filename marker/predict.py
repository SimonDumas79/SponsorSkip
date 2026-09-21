"""Run the free tier end to end on one video's captions, and print the segments to skip.

This is the path a live video takes, with no language model anywhere:
  1. build the caption rows exactly as build_dataset.py does, and the 395 numbers
     per line exactly as features.py does (the same functions, imported), so the
     live features cannot drift from the training ones
  2. the marker scores each line
  3. the context model reads those scores 15 lines either side and says which
     lines are inside a read
  4. runs above the threshold become regions, and the edge heads move each
     region's start and end to the most likely start and resume lines
  5. the regions become (start, end) seconds

The saved system is exactly the one graded on both holdouts (free tier: 49.6% and
49.3% of ad time skipped, at 9.9 s and 7.5 s of real show lost per video). The
threshold is the one the written rule chose on cross-validation.

    python marker/predict.py export                          # train and save data/production/free_tier.pt
    python marker/predict.py run marker/data/captions/ID.json   # the segments to skip, as JSON
    python marker/predict.py check                           # the live path must reproduce the graded system
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from build_dataset import label_video
from context_stack import context_features, fit_stage2
from edge_heads import place, soft
from experiments import BASE, cross_validate, fit_full, predict_full
from features import MODEL_NAME, embed_lines, video_features
from replay import grade_regions, regions_by_video, row, HEADER
from train import DATA, FEATURES, Rows, load

BUNDLE = DATA / "production" / "free_tier.pt"
THRESHOLD = 0.9781   # system_eval.py's cross-validated choice for the free tier (B = 5 and 10 s)


# ----------------------------------------------------------------------------- saving and loading


def stage_state(stage) -> dict:
    return {"state": stage.model.state_dict(), "mean": torch.from_numpy(stage.mean.astype(np.float32)),
            "std": torch.from_numpy(stage.std.astype(np.float32)), "hidden": stage.hidden}


def stage_from(state: dict, n_in: int):
    hidden = state["hidden"]
    model = nn.Linear(n_in, 1) if hidden == 0 else nn.Sequential(
        nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1))
    model.load_state_dict(state["state"])
    model.eval()
    mean, std = state["mean"].numpy(), state["std"].numpy()

    def predict(G: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(model(torch.from_numpy((G - mean) / std).float()).squeeze(1)).numpy()
    return predict


def export() -> None:
    """Train every component exactly as system_eval.py did for its holdout grades, and save them."""
    rows = load(DATA / "features.npz")
    d = np.load(DATA / "features.npz")
    F = context_features(rows, cross_validate(BASE, rows, seed=0))   # out-of-fold marker scores
    marker = fit_full(BASE, rows, seed=0)
    stages = {"context": fit_stage2(F, rows.y.astype(np.float32), hidden=32),
              "start": fit_stage2(F, soft(d["is_start"], rows.video), hidden=32),
              "end": fit_stage2(F, soft(d["is_resume"], rows.video), hidden=32)}
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "marker": [{"columns": torch.from_numpy(p["columns"]), "mean": torch.from_numpy(p["mean"]),
                    "std": torch.from_numpy(p["std"]), "state": p["model"].state_dict()} for p in marker],
        "stages": {name: stage_state(s) for name, s in stages.items()},
        "context_inputs": F.shape[1], "threshold": THRESHOLD, "features": FEATURES, "encoder": MODEL_NAME,
        "trained_on": f"features.npz: {rows.videos} videos, {rows.reads} reads",
    }, BUNDLE)
    print(f"saved {BUNDLE} (marker + context model + start/end heads, threshold {THRESHOLD})")


class FreeTier:
    def __init__(self, path: Path = BUNDLE, device: str | None = None):
        b = torch.load(path)
        self.threshold = b["threshold"]
        self.marker = []
        for p in b["marker"]:
            model = nn.Linear(len(p["columns"]), 1)
            model.load_state_dict(p["state"])
            self.marker.append({"columns": p["columns"].numpy(), "mean": p["mean"].numpy(),
                                "std": p["std"].numpy(), "model": model.eval()})
        n_in = b["context_inputs"]
        self.context, self.start, self.end = (stage_from(b["stages"][k], n_in) for k in ("context", "start", "end"))
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(b["encoder"], device=device)

    def rows_for(self, caps: dict) -> Rows:
        lines = label_video(caps, [])   # the same rows build_dataset.py makes, labels all 0
        vectors = embed_lines([r["text"] for r in lines], self.encoder, 128)
        X = video_features(vectors, lines, caps.get("duration") or 0.0).astype(np.float32)
        n = len(lines)
        return Rows(X, np.zeros(n, dtype=np.int8), np.array([caps["videoID"]] * n),
                    np.array([lines[0]["channel_id"]] * n), np.array([r["start"] for r in lines], dtype=np.float32),
                    np.array(["live"] * n))

    def regions(self, rows: Rows) -> list[tuple[int, int]]:
        """Line ranges [start, end) to skip."""
        level1 = predict_full(self.marker, rows.X)
        G = context_features(rows, level1)
        kept = regions_by_video(self.context(G), rows, self.threshold, 1)
        placed = place(kept, rows, self.start(G), self.end(G))
        return next(iter(placed.values()), [])

    def segments(self, caps: dict) -> list[dict]:
        rows = self.rows_for(caps)
        starts = rows.start_seconds
        end_of = lambda j: float(starts[j]) if j < len(starts) else float(caps.get("duration") or starts[-1] + 3.0)
        return [{"start": round(float(starts[a]), 2), "end": round(end_of(b), 2), "category": "sponsor"}
                for a, b in self.regions(rows)]


# ----------------------------------------------------------------------------- commands


def check() -> int:
    """Holdout 2 through the live path: same regions as the evaluator, and the same grade."""
    tier = FreeTier()
    h = load(DATA / "features_holdout2.npz")
    h_l1 = np.load(DATA / "confirm_verdicts_holdout2.scores.npy")
    offline = {}
    G = context_features(h, h_l1)
    offline = place(regions_by_video(tier.context(G), h, tier.threshold, 1), h, tier.start(G), tier.end(G))
    live, same = {}, 0
    for vid in np.unique(h.video):
        caps = json.loads((DATA / "captions" / f"{vid}.json").read_text(encoding="utf-8"))
        live[str(vid)] = tier.regions(tier.rows_for(caps))
        same += live[str(vid)] == offline.get(str(vid), [])
    print(f"{same} of {h.videos} videos: the live path cut exactly the evaluator's regions")
    print(HEADER)
    print(row("evaluator (features_holdout2.npz)", grade_regions(offline, h)))
    print(row("live path (captions -> segments)", grade_regions(live, h)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export")
    run = sub.add_parser("run")
    run.add_argument("captions", type=Path, help="a caption file as fetch_captions.py / server/youtube.mjs write them")
    sub.add_parser("check")
    sub.add_parser("serve", help="read {videoID, channel, duration, lines:[{start,text}]} on stdin, "
                                 "write {segments} on stdout: how server/agents.mjs calls it")
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.cmd == "export":
        export()
    elif args.cmd == "serve":
        caps = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        caps.setdefault("channel_id", caps.get("channel") or "unknown")
        segments = FreeTier(device="cpu").segments(caps) if caps.get("lines") else []
        sys.stdout.write(json.dumps({"segments": segments}))
    elif args.cmd == "run":
        caps = json.loads(args.captions.read_text(encoding="utf-8"))
        json.dump({"videoID": caps["videoID"], "segments": FreeTier().segments(caps)}, sys.stdout, indent=1)
        print()
    else:
        return check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
