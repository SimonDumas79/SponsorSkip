"""Load the candidate bundle and score a video with it. The serving half of export_candidate.py.

The shipped FreeTier in predict.py runs one linear marker, a context model and two edge heads. The
candidate runs six detectors into one context model, so it needs its own loader. What it costs per
video, beyond the old free tier: the potion embedding (8M parameters) and the fine-tuned BGE-small
(33M, one window of 17 lines per line). Both are CPU-only; BGE dominates.

    from serve_candidate import Candidate
    c = Candidate()
    ctx, p_start, p_end = c.score(rows, potion_X, bge_scores)   # streams for a set of rows

`bge_scores` may be handed in (the graded runs precomputed them) or left out, in which case the
checkpoint in the bundle computes them from the caption lines.
"""

import numpy as np
import torch
from torch import nn

from context_stack import context_features
from experiments import predict_full
from predict import BUNDLE, stage_from
from sequence_model import SequenceMarker

DEFAULT = BUNDLE.parent / "candidate.pt"


class Candidate:
    def __init__(self, path=DEFAULT, device: str | None = None):
        b = torch.load(path, weights_only=False)
        self.bundle_dir = path.parent
        self.order = b["order"]
        self.extra_cols = b["extra_cols"]
        self.thresholds = {int(k): float(v) for k, v in b["thresholds"].items()}
        self.thresholds_v2 = {int(k): float(v) for k, v in b["thresholds_v2"].items()}
        self.detectors = {}
        for name, parts in b["detectors"].items():
            out = []
            for p in parts:
                model = nn.Linear(len(p["columns"]), 1)
                model.load_state_dict(p["state"])
                out.append({"columns": p["columns"].numpy(), "mean": p["mean"].numpy(),
                            "std": p["std"].numpy(), "model": model.eval()})
            self.detectors[name] = out
        pot = b["potion"]
        self.potion_mean, self.potion_std = pot["mean"].numpy(), pot["std"].numpy()
        n_pot = self.potion_mean.shape[0]
        self.potion = nn.Linear(n_pot, 1)
        self.potion.load_state_dict(pot["state"])
        self.potion.eval()
        s = b["sequence"]
        self.seq_mean, self.seq_std = s["mean"].numpy(), s["std"].numpy()
        self.sequence = SequenceMarker(len(self.seq_mean), s["hidden"], s["reach"])
        self.sequence.load_state_dict(s["state"])
        self.sequence.eval()
        self.context = stage_from(b["stages"]["context"], b["context_inputs"])
        self.start = stage_from(b["stages"]["start"], b["edge_inputs"])
        self.end = stage_from(b["stages"]["end"], b["edge_inputs"])
        self.fine_tuned = b.get("fine_tuned", {})
        self.device = device
        self._bge = None

    def _linear(self, name: str, X: np.ndarray) -> np.ndarray:
        return predict_full(self.detectors[name], X)

    def _potion_stream(self, potX: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = self.potion(torch.from_numpy((potX - self.potion_mean) / self.potion_std).float()).squeeze(1).numpy()
        return 1.0 / (1.0 + np.exp(-z))

    def _sequence_stream(self, rows) -> np.ndarray:
        X = torch.from_numpy((rows.X[:, :395] - self.seq_mean) / self.seq_std).float()
        out = np.zeros(len(rows.y), dtype=np.float32)
        with torch.no_grad():
            for v in np.unique(rows.video):
                r = np.flatnonzero(rows.video == v)
                out[r] = torch.sigmoid(self.sequence(X[torch.from_numpy(r)])).numpy()
        return out

    def bge_stream(self, texts: list[str], video: np.ndarray) -> np.ndarray:
        """The fine-tuned detector's score per line, from the checkpoint in the bundle."""
        from finetune_minilm import OPT, Scorer, encode, pairs
        from transformers import AutoTokenizer
        if self._bge is None:
            ck = torch.load(self.bundle_dir / self.fine_tuned["bge"], weights_only=False)
            OPT.update(ck["opt"])
            model = Scorer()
            model.load_state_dict(ck["state"])
            self._bge = (model.eval(), AutoTokenizer.from_pretrained(OPT["model"]))
        model, tok = self._bge

        class _R:
            def __init__(self, v):
                self.video = v

            def __len__(self):
                return len(self.video)

        data = pairs(_R(video), texts)
        out = np.zeros(len(texts), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(data), 64):
                ids, mask, types = encode(tok, data[i:i + 64], "cpu")
                out[i:i + ids.shape[0]] = torch.sigmoid(model(ids, mask, types).float()).numpy()
        return out

    def streams(self, rows, potX: np.ndarray, bge: np.ndarray) -> dict:
        return {"marker": self._linear("marker", rows.X), "meaning": self._linear("meaning", rows.X),
                "structure": self._linear("structure", rows.X), "potion": self._potion_stream(potX),
                "sequence": self._sequence_stream(rows), "bge": bge}

    def score(self, rows, potX: np.ndarray, bge: np.ndarray):
        """Returns (context score, start score, end score), one per row."""
        st = self.streams(rows, potX, bge)
        F = np.column_stack([context_features(rows, st[k]) for k in self.order] + [rows.X[:, self.extra_cols]])
        E = context_features(rows, st["marker"])
        return self.context(F), self.start(E), self.end(E)
