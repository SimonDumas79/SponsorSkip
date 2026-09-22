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
        self.gate = b.get("gate")
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

    def ft_stream(self, which: str, texts: list[str], video: np.ndarray,
                  mask: np.ndarray | None = None, neutral: float = 0.0) -> np.ndarray:
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
        # The gate: score only the lines asked for and give the rest the model's neutral value. The
        # windows still read their real neighbours, so a scored line sees the same text either way.
        want = np.arange(len(data)) if mask is None else np.flatnonzero(mask)
        out = np.full(len(texts), float(neutral), dtype=np.float32)
        data = [data[i] for i in want]
        # Serving is CPU by design (the extension must not fight a game for the GPU), but the batch
        # checks grade tens of thousands of lines, which is an hour on the CPU and two minutes on the
        # card. `device` decides: "cpu" unless a caller asks for cuda.
        dev = self.device if self.device in ("cpu", None) else self.device
        dev = "cpu" if dev is None else dev
        model = model.to(dev)
        with torch.no_grad():
            for i in range(0, len(data), 64):
                ids, att, types = encode(tok, data[i:i + 64], dev)
                got = torch.sigmoid(model(ids, att, types).float()).cpu().numpy()
                out[want[i:i + len(got)]] = got
        return out

    def cheap_streams(self, rows, potX: np.ndarray) -> dict:
        """The five detectors that are not the fine-tuned one: Simon's first layer."""
        return {"marker": self._linear("marker", rows.X), "meaning": self._linear("meaning", rows.X),
                "structure": self._linear("structure", rows.X), "potion": self._potion_stream(potX),
                "sequence": self._sequence_stream(rows)}

    def gate_mask(self, cheap: dict, rows) -> np.ndarray:
        """Which lines the expensive detector has to read: those near anything the cheap five saw.

        The cut is fixed on the pooled data and stored in the bundle, so a video is not judged
        against itself; a video with no ad in it simply sends fewer lines through.
        """
        from export_candidate import smooth
        m = np.max(np.column_stack([cheap[k] for k in ("marker", "meaning", "structure", "potion", "sequence")]), axis=1)
        return smooth(m, rows.video, int(self.gate["window"])) >= float(self.gate["cut"])

    def streams(self, rows, potX: np.ndarray, bge: np.ndarray) -> dict:
        return {**self.cheap_streams(rows, potX), "bge": bge}

    def score(self, rows, potX: np.ndarray, bge: np.ndarray):
        """Returns (context score, start score, end score), one per row."""
        return self.score_from(rows, self.streams(rows, potX, bge))

    def score_from(self, rows, st: dict):
        """The same, from streams already computed (so the cheap five are not run twice)."""
        F = np.column_stack([context_features(rows, st[k]) for k in self.order] + [rows.X[:, self.extra_cols]])
        E = context_features(rows, st["marker"])
        return self.context(F), self.start(E), self.end(E)


class CandidateTier(Candidate):
    """Captions in, segments out: the candidate as the extension would call it.

    Three encoders run per video, all on the CPU: MiniLM for the 395 features the linear detectors
    and the sequence model read, potion-base-8M for the potion detector, and the fine-tuned
    BGE-small for the sixth detector (and, in v2, the two edge models). BGE dominates the cost.
    """

    def __init__(self, path=DEFAULT, device: str | None = "cpu", budget: int = 10, v2: bool = True,
                 gated: bool = True):
        super().__init__(path, device)
        self.budget, self.v2 = budget, v2
        self.gated = gated and bool(self.gate)
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
        # The description matters: the stack's context model reads two features derived from it
        # (desc_hit_line, desc_hits_context). Serving without it would hand the model zeros where
        # the graded run had real values, so it would not be the system that was measured.
        X = video_features(embed_lines(texts, self._encoder("minilm"), 128), lines, duration,
                           caps.get("description")).astype(np.float32)
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
        # Layer 1 sweeps, layer 2 reads only where it saw something. Measured on holdout 4: a fifth
        # of the lines costs 0.1 of a point of ad time and does a fifth of the expensive work.
        cheap = self.cheap_streams(rows, pot)
        mask = self.gate_mask(cheap, rows) if self.gated else None
        neutral = float(self.gate["neutral"]) if self.gated else 0.0
        bge = self.ft_stream("bge", texts, rows.video, mask, neutral)
        ctx, ps, pe = self.score_from(rows, {**cheap, "bge": bge})
        if self.v2:
            ns = float(self.gate.get("neutral_start", 0.0)) if self.gated else 0.0
            ne = float(self.gate.get("neutral_end", 0.0)) if self.gated else 0.0
            ps = (ps + self.ft_stream("edge_start", texts, rows.video, mask, ns)) / 2
            pe = (pe + self.ft_stream("edge_resume", texts, rows.video, mask, ne)) / 2
            th = self.thresholds_v2[self.budget]
        else:
            th = self.thresholds[self.budget]
        placed = place(regions_by_video(ctx, rows, th, 1), rows, ps, pe)
        # A chapter the creator titled "Sponsor" is a read they marked themselves. Adopted on the
        # pooled set and on holdout 4: about a point of ad time, less show lost, and not one false
        # alarm in any measured set. Free, so it is applied whenever yt-dlp returned chapters.
        if caps.get("chapters"):
            from candidate import chapter_regions
            # yt-dlp names them start_time/end_time/title; the rule was written against the watch-page
            # shape, which is title/start. Normalise here rather than in the rule.
            ch = [{"title": c.get("title") or "", "start": float(c.get("start") or c.get("start_time") or 0.0)}
                  for c in caps["chapters"]]
            placed = chapter_regions(placed, rows, {str(caps["videoID"]): {"chapters": ch}})
        return next(iter(placed.values()), []), rows

    def segments(self, caps: dict) -> list[dict]:
        spans, rows = self.regions(caps)
        starts = rows.start_seconds
        end_of = lambda j: float(starts[j]) if j < len(starts) else float(caps.get("duration") or starts[-1] + 3.0)
        return [{"start": round(float(starts[a]), 2), "end": round(end_of(b), 2), "category": "sponsor"}
                for a, b in spans]


def check(which: str, limit: int) -> int:
    """The live path against the evaluator: captions in, and the same regions out.

    export_candidate.py --verify proves the bundle reproduces the grade from FEATURE FILES. This
    proves the other half: that going from a raw caption file all the way to segments -- three
    encoders, six detectors, the stack, the edge heads -- lands on the same regions. A mismatch here
    means the serving path builds features differently from the way the training data was built,
    which is the failure that would quietly cost accuracy in the extension and nowhere else.
    """
    import json
    import time

    from edge_heads import place
    from replay import regions_by_video
    from train import DATA, Rows, load

    from export_candidate import OUT, POTION
    rows = load(DATA / f"features_{which}.npz")
    pot = np.load(DATA / POTION[f"features_{which}.npz"])["X"]
    pot = pot[:, :pot.shape[1] - 2]
    # The offline side uses the CHECKPOINTS too, not the .npy scores the graded runs wrote. Those
    # came from a different training run, so comparing against them would mix two questions. Here the
    # only difference between the two sides is where the features came from: built live from a
    # caption file, or read from the prebuilt .npz. That is the thing this check is for.
    text = {}
    with open(DATA / f"examples_{which}.jsonl", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                text[(r["videoID"], r["i"])] = r["text"]
    d = np.load(DATA / f"features_{which}.npz")
    texts = [text[(str(v), int(i))] for v, i in zip(d["video"], d["line"])]
    # Only the videos actually compared: the offline side runs the fine-tuned models on the CPU, and
    # scoring the whole set to check five videos turned a two-minute check into twenty.
    wanted = [str(v) for v in dict.fromkeys(rows.video)][:limit]
    keep = np.isin(rows.video.astype(str), np.array(wanted))
    rows = Rows(rows.X[keep], rows.y[keep], rows.video[keep], rows.channel[keep],
                rows.start_seconds[keep], rows.split[keep], rows.feature_names)
    pot = pot[keep]
    texts = [t for t, k in zip(texts, keep) if k]
    tier = CandidateTier(OUT, device="cpu")
    # The offline side is gated exactly as serving is. Comparing a gated live path against an
    # ungated offline one would report a difference that is the gate, not a bug in the path.
    cheap = tier.cheap_streams(rows, pot)
    mask = tier.gate_mask(cheap, rows) if tier.gated else None
    g = tier.gate or {}
    neutral = float(g.get("neutral", 0.0)) if tier.gated else 0.0
    ns = float(g.get("neutral_start", 0.0)) if tier.gated else 0.0
    ne = float(g.get("neutral_end", 0.0)) if tier.gated else 0.0
    bge = tier.ft_stream("bge", texts, rows.video, mask, neutral)
    fs = tier.ft_stream("edge_start", texts, rows.video, mask, ns)
    fe = tier.ft_stream("edge_resume", texts, rows.video, mask, ne)
    ctx, ps, pe = tier.score_from(rows, {**cheap, "bge": bge})
    share = 1.0 if mask is None else float(mask.mean())
    print(f"  the fine-tuned models read {share:.0%} of lines", flush=True)
    print(flush=True)
    offline = place(regions_by_video(ctx, rows, tier.thresholds_v2[tier.budget], 1), rows,
                    (ps + fs) / 2, (pe + fe) / 2)

    vids = wanted
    same = 0
    for vid in vids:
        caps = json.loads((DATA / "captions" / f"{vid}.json").read_text(encoding="utf-8"))
        t0 = time.time()
        live, _ = tier.regions(caps)
        want = offline.get(vid, [])
        ok = live == want
        same += ok
        print(f"  {vid}  {'same' if ok else 'DIFFERENT'}  live {live}  evaluator {want}  "
              f"({time.time() - t0:.1f} s)", flush=True)
    print(f"\n{same} of {len(vids)} videos: the live path cut exactly the evaluator's regions")
    return 0 if same == len(vids) else 1


if __name__ == "__main__":
    import argparse
    import ctypes
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    torch.set_num_threads(8)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["check"])
    ap.add_argument("set", nargs="?", default="holdout4")
    ap.add_argument("--limit", type=int, default=6)
    a = ap.parse_args()
    raise SystemExit(check(a.set, a.limit))
