"""Step 5: turn labelled caption lines into numbers a model can learn from.

A model cannot read text. Every example has to become a fixed-length row of
numbers, the same length for every example. That row is what your training loop
will see. Here is what goes in each row and why.

1. MEANING OF THE SURROUNDING TEXT (384 numbers)
   Each caption line is run through MiniLM, a small sentence model that turns
   text into 384 numbers, arranged so that text about similar things lands in
   similar places. We average the lines around this one, so the row describes
   the neighbourhood, not the fragment. This is what lets the model notice
   "this passage sounds like a product pitch" without anyone writing a list of
   product words.

2. RELATEDNESS -- your idea (3 numbers)
   The five lines BEFORE this point are averaged into one vector, the five
   AFTER into another, and we measure the angle between them. Close together
   means the subject carried on. Far apart means the subject just changed --
   a seam. A sponsor read always begins at a seam, because the host stops
   talking about the show and starts talking about a mattress.

   It is given as three numbers, not one: the seam right here, the sharpest
   seam in the ten lines behind, and how far back that seam was. The model can
   then learn something like "ad language, and a seam twelve lines back" --
   which is a much better description of the START of a read than either part
   alone. This is exactly what qwen3 could not do on 2026-09-20: it recognised
   ad language fine and still placed starts a minute late.

   Note it compares BLOCKS of lines, not a line and the one before it. Caption
   lines are three-word fragments; consecutive fragments are always related, so
   line-to-line comparison would be noise.

3. AD CUES (a handful of numbers)
   Whether stock sponsor phrasing appears nearby -- "brought to you by",
   "use code", "link in the description". Free, and it catches the conventional
   reads instantly. These are also the whole of the word-pattern baseline, so
   the model has to earn its keep by beating them.

4. POSITION (2 numbers)
   How far into the video this is, in minutes and as a fraction. Reads cluster
   near the start and around a third of the way in.

Output: data/features.npz  -- X (rows of numbers), y (0/1), plus the video id,
line index and split for every row, so a prediction can always be traced back
to the exact caption line it came from.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

ENCODERS = {
    # name: (identifier, what it is)
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "22M parameters, ~90 MB, a real transformer"),
    # A static model: no transformer at inference at all, just a token lookup and
    # an average. Small enough to reimplement in plain JavaScript, which is what
    # would let the marker ship inside the extension itself.
    "potion": ("minishlab/potion-base-8M", "7.5M parameters, ~8 MB, a lookup table"),
}
MODEL_NAME = ENCODERS["minilm"][0]
BLOCK = 5      # lines each side for the relatedness comparison
CONTEXT = 4    # lines each side averaged into the meaning vector
LOOKBACK = 10  # how far back to hunt for the sharpest seam

# The conventional phrasings. Deliberately small: this is the baseline to beat,
# and every pattern added here is a pattern that won't generalise to the tail.
CUES = {
    "sponsor_word": r"\bsponsor(ed|s|ship)?\b",
    "brought_to_you": r"\bbrought to you by\b|\bthanks to\b.{0,30}\bfor sponsoring\b",
    "promo_code": r"\b(use|with) (the )?code\b|\bpromo code\b|\bdiscount code\b",
    "link_below": r"\blink (in|below)\b|\bdescription below\b|\bcheck out\b.{0,40}\b(link|below)\b",
    "offer": r"\b\d{1,2}% off\b|\bfree trial\b|\bsign up (at|for)\b|\bfirst \d+ (people|viewers)\b",
    "url": r"\b[a-z0-9-]+\.(com|co|io|net|org)/[a-z0-9-]+",
}
CUE_RE = {name: re.compile(p, re.I) for name, p in CUES.items()}


def cue_row(text: str) -> list[float]:
    return [1.0 if r.search(text) else 0.0 for r in CUE_RE.values()]


def read_examples(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def embed_lines(texts: list[str], model, batch_size: int) -> np.ndarray:
    """One vector per caption line, unit length so a dot product IS the cosine."""
    if hasattr(model, "encode_as_sequence") or model.__class__.__name__ == "StaticModel":
        vectors = np.asarray(model.encode(texts, show_progress_bar=True), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.where(norms > 0, norms, 1)).astype(np.float32)
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)


def block_mean(vectors: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Average of vectors[lo:hi], re-normalised. Empty range -> zeros."""
    lo, hi = max(lo, 0), min(hi, len(vectors))
    if hi <= lo:
        return np.zeros(vectors.shape[1], dtype=np.float32)
    v = vectors[lo:hi].mean(axis=0)
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def video_features(vectors: np.ndarray, rows: list[dict], duration: float) -> np.ndarray:
    """The full row of numbers for every caption line in one video."""
    n = len(rows)
    context = np.stack([block_mean(vectors, i - CONTEXT, i + CONTEXT + 1) for i in range(n)])

    # Relatedness: how much the meaning either side of each point disagrees.
    # 0 = the subject carried straight on, 1 = a complete change of subject.
    seam = np.zeros(n, dtype=np.float32)
    for i in range(n):
        before, after = block_mean(vectors, i - BLOCK, i), block_mean(vectors, i, i + BLOCK)
        seam[i] = 1.0 - float(np.dot(before, after)) if before.any() and after.any() else 0.0

    sharpest = np.zeros(n, dtype=np.float32)
    distance = np.zeros(n, dtype=np.float32)
    for i in range(n):
        window = seam[max(0, i - LOOKBACK) : i + 1]
        j = int(np.argmax(window))
        sharpest[i] = window[j]
        distance[i] = (len(window) - 1 - j) / LOOKBACK  # 0 = right here, 1 = ten lines back

    cues = np.array([cue_row(r["context"]) for r in rows], dtype=np.float32)
    minutes = np.array([[r["start"] / 60.0] for r in rows], dtype=np.float32)
    fraction = np.array([[r["start"] / duration if duration else 0.0] for r in rows], dtype=np.float32)

    return np.hstack([context, seam[:, None], sharpest[:, None], distance[:, None], cues, minutes, fraction])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", type=Path, default=HERE / "data" / "examples.jsonl")
    ap.add_argument("--captions", type=Path, default=HERE / "data" / "captions")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "features.npz")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default=None, help="cuda or cpu; default picks cuda when present")
    ap.add_argument("--encoder", choices=sorted(ENCODERS), default="minilm")
    args = ap.parse_args()

    name, what = ENCODERS[args.encoder]

    rows = read_examples(args.examples)
    by_video: dict[str, list[dict]] = {}
    for r in rows:
        by_video.setdefault(r["videoID"], []).append(r)
    print(f"{len(rows):,} caption lines across {len(by_video)} videos")

    if args.encoder == "potion":
        from model2vec import StaticModel

        model = StaticModel.from_pretrained(name)
        print(f"embedding with {name} ({what})")
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(name, device=args.device)
        print(f"embedding with {name} ({what}) on {model.device}")
    order = sorted(by_video)
    flat = [r["text"] for vid in order for r in sorted(by_video[vid], key=lambda r: r["i"])]
    vectors = embed_lines(flat, model, args.batch_size)

    X_parts, y_parts, meta = [], [], []
    at = 0
    for vid in order:
        vrows = sorted(by_video[vid], key=lambda r: r["i"])
        caps = json.loads((args.captions / f"{vid}.json").read_text(encoding="utf-8"))
        X_parts.append(video_features(vectors[at : at + len(vrows)], vrows, caps.get("duration") or 0.0))
        y_parts.append(np.array([r["label"] for r in vrows], dtype=np.int8))
        meta.extend((vid, r["i"], r["split"], r["is_start"], r["start"], r["channel_id"],
                     r.get("is_resume", 0), r.get("category", "sponsor" if r["label"] else "")) for r in vrows)
        at += len(vrows)

    X = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_parts)
    np.savez_compressed(
        args.out,
        X=X,
        y=y,
        video=np.array([m[0] for m in meta]),
        line=np.array([m[1] for m in meta], dtype=np.int32),
        split=np.array([m[2] for m in meta]),
        is_start=np.array([m[3] for m in meta], dtype=np.int8),
        start_seconds=np.array([m[4] for m in meta], dtype=np.float32),
        channel=np.array([m[5] for m in meta]),
        is_resume=np.array([m[6] for m in meta], dtype=np.int8),   # first line after a read: an END label
        category=np.array([m[7] for m in meta]),                    # "sponsor", "selfpromo" or ""
        feature_names=np.array(
            [f"meaning_{i}" for i in range(vectors.shape[1])]
            + ["seam_here", "seam_sharpest_behind", "seam_distance_behind"]
            + [f"cue_{name}" for name in CUES]
            + ["minutes_in", "fraction_in"]
        ),
    )
    print(f"X {X.shape}  y {y.shape}  positives {int(y.sum()):,} ({100 * y.mean():.1f}%)")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
