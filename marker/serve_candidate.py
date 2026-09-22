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

    def ft_stream(self, which: str, texts: list[str], video: np.ndarray) -> np.ndarray:
        """A fine-tuned model's score per line, from its checkpoint in the bundle.

        `which` is "bge" (inside a read), "edge_start" or "edge_resume". Each is the same
        architecture over the same 17-line window; only the target it was trained on differs.
        """
        from finetune_minilm import OPT, Scorer, encode, pairs
        from transformers import AutoTokenizer
        if self._bge is None:
            self._bge = {}
        if which not in self._bge:
            ck = torch.load(self.bundle_dir / self.fine_tuned[which], weights_only=False)
            OPT.update(ck["opt"])
            model = Scorer()
            model.load_state_dict(ck["state"])
            self._bge[which] = (model.eval(), AutoTokenizer.from_pretrained(OPT["model"]))
        model, tok = self._bge[which]

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


class CandidateTier(Candidate):
    """Captions in, segments out: the candidate as the extension would call it.

    Three encoders run per video, all on the CPU: MiniLM for the 395 features the linear detectors
    and the sequence model read, potion-base-8M for the potion detector, and the fine-tuned
    BGE-small for the sixth detector (and, in v2, the two edge models). BGE dominates the cost.
    """

    def __init__(self, path=DEFAULT, device: str | None = "cpu", budget: int = 10, v2: bool = True):
        super().__init__(path, device)
        self.budget, self.v2 = budget, v2
        self._encoders = {}

    def _encoder(self, kind: str):
        if kind not in self._encoders:
            if kind == "potion":
                from model2vec import StaticModel
                from features import ENCODERS
                self._encoders[kind] = StaticModel.from_pretrained(ENCODERS["potion"][0])
            else:
                from sentence_transformers import SentenceTransformer
                from features import MODEL_NAME
                self._encoders[kind] = SentenceTransformer(MODEL_NAME, device=self.device)
        return self._encoders[kind]

    def rows_for(self, caps: dict):
        from build_dataset import label_video
        from features import embed_lines, video_features
        from train import Rows
        lines = label_video(caps, [])
        texts = [r["text"] for r in lines]
        duration = caps.get("duration") or 0.0
        X = video_features(embed_lines(texts, self._encoder("minilm"), 128), lines, duration).astype(np.float32)
        pot = video_features(embed_lines(texts, self._encoder("potion"), 128), lines, duration).astype(np.float32)
        n = len(lines)
        rows = Rows(X, np.zeros(n, dtype=np.int8), np.array([str(caps["videoID"])] * n),
                    np.array([lines[0]["channel_id"]] * n),
                    np.array([r["start"] for r in lines], dtype=np.float32), np.array(["live"] * n))
        return rows, pot[:, :pot.shape[1] - 2], texts

    def regions(self, caps: dict) -> list[tuple[int, int]]:
        from edge_heads import place
        from replay import regions_by_video
        rows, pot, texts = self.rows_for(caps)
        bge = self.ft_stream("bge", texts, rows.video)
        ctx, ps, pe = self.score(rows, pot, bge)
        if self.v2:
            ps = (ps + self.ft_stream("edge_start", texts, rows.video)) / 2
            pe = (pe + self.ft_stream("edge_resume", texts, rows.video)) / 2
            th = self.thresholds_v2[self.budget]
        else:
            th = self.thresholds[self.budget]
        placed = place(regions_by_video(ctx, rows, th, 1), rows, ps, pe)
        return next(iter(placed.values()), []), rows

    def segments(self, caps: dict) -> list[dict]:
        spans, rows = self.regions(caps)
        starts = rows.start_seconds
        end_of = lambda j: float(starts[j]) if j < len(starts) else float(caps.get("duration") or starts[-1] + 3.0)
        return [{"start": round(float(starts[a]), 2), "end": round(end_of(b), 2), "category": "sponsor"}
                for a, b in spans]
