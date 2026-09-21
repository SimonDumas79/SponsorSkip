"""Simon's question: can a local model place an edge if you never ask it to SEARCH?

Measured 2026-09-20, qwen3:8b asked to find a sponsor read's start inside a
50-line window placed the start a median 55.7 s late. The failure was
localisation, not judgement -- scanning a block of text and reporting WHERE
something is.

So this asks it nothing but a local yes/no. MiniLM nominates the five sharpest
seams in the window (free, already computed), and for each one the model sees
only the five lines before and the five lines after, and answers one question:
did the subject change here? No searching, no line numbers, no timestamps.

Blocks of five lines, not single lines, because caption lines are three-word
fragments -- "and that's why we" -- and two consecutive fragments are always
related, so a literal sentence-to-sentence question would be noise.

The comparison is like for like: the same windows and the same five candidates
that marker/edge_check.py scores, so the only thing that changes is who picks.

Run: python marker/pairwise_probe.py     (uses the GPU; unloads the model after)
"""

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OLLAMA = "http://localhost:11434"
BLOCK = 5  # caption lines each side of a candidate seam

SCHEMAS = {
    "bool": {"type": "object", "properties": {"yes": {"type": "boolean"}}, "required": ["yes"]},
    "score": {
        "type": "object",
        "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 10}},
        "required": ["score"],
    },
}

QUESTIONS = {
    # v1, measured 2026-09-20: qwen answered "changed" on 3-5 of 5 candidates in
    # 20 of 27 windows. Not selective enough to rank candidates.
    "topic": """Here are two consecutive passages from the middle of a YouTube video's captions.

PASSAGE A:
{a}

PASSAGE B:
{b}

Is passage B still talking about the same thing as passage A, or has the subject changed?
A subject change means the speaker stopped what they were doing and started something else, such as beginning to read out an advertisement.

Answer ONLY with JSON: {{"yes": true}} if the subject changed, {{"yes": false}} if it did not.""",

    # v2: ask the thing qwen is demonstrably good at. It scored 6 of 6 on
    # DETECTING sponsor reads; this keeps the local, two-item framing but points
    # it at recognition instead of change-of-subject. It is also self-limiting:
    # it can only be true at the real boundary, because a candidate inside the ad
    # has an advertisement in passage A too.
    # v3: a 0-10 scale instead of yes/no. The binary answer saturated -- it said
    # yes to 3.6 of 5 candidates -- so a grade is the obvious next lever.
    "score": """Here are two consecutive passages from the middle of a YouTube video's captions.

PASSAGE A:
{a}

PASSAGE B:
{b}

How likely is it that a sponsor read -- an advertisement the host reads out -- BEGINS exactly here, at the start of passage B?
10 means passage A is clearly the normal show and passage B clearly starts the advertisement.
0 means this is definitely not where an advertisement begins, either because neither passage is an advertisement or because the advertisement was already running in passage A.

Answer ONLY with JSON: {{"score": <0-10>}}""",

    # v4: the same question, but the answer is read from the model's own token
    # probabilities rather than from the word it happened to sample. Small models
    # are measurably miscalibrated when ASKED how sure they are (their verbal
    # confidence saturates), while the logits carry a usable signal. Needs a plain
    # prompt, not a JSON schema, so the first token really is Yes or No.
    "ad_lp": """Here are two consecutive passages from the middle of a YouTube video's captions.

PASSAGE A:
{a}

PASSAGE B:
{b}

Does a sponsor read -- an advertisement the host reads out -- BEGIN in passage B?
Say Yes only if passage B starts the advertisement and passage A is still the normal show.
Say No if neither passage is an advertisement, or if the advertisement was already running in passage A.

Answer with one word, Yes or No.""",

    "ad": """Here are two consecutive passages from the middle of a YouTube video's captions.

PASSAGE A:
{a}

PASSAGE B:
{b}

Does a sponsor read -- an advertisement the host reads out -- BEGIN in passage B?
Answer true only if passage B starts the advertisement and passage A is still the normal show.
Answer false if neither passage is an advertisement, or if the advertisement was already running in passage A.

Answer ONLY with JSON: {{"yes": true}} or {{"yes": false}}""",
}


def ask_ollama(model: str, prompt: str, kind: str = "bool", timeout: float = 90.0) -> float | None:
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "think": False,  # a yes/no needs no reasoning trace, and thinking triples the latency
            "format": SCHEMAS[kind],
            "keep_alive": "120s",
            "options": {"num_ctx": 2048, "temperature": 0},
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            content = json.load(r)["message"]["content"]
        answer = json.loads(content)
        return float(answer["score"]) if kind == "score" else float(bool(answer["yes"]))
    except (urllib.error.URLError, KeyError, ValueError, TimeoutError):
        return None


def ask_logprob(model: str, prompt: str, timeout: float = 90.0) -> float | None:
    """P(Yes) against P(No) on the first generated token, as a number in 0..1."""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "logprobs": True,
            "top_logprobs": 12,
            "keep_alive": "120s",
            "options": {"temperature": 0, "num_predict": 1, "num_ctx": 2048},
        }
    ).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            doc = json.load(r)
        top = doc["logprobs"][0]["top_logprobs"]
    except (urllib.error.URLError, KeyError, IndexError, ValueError, TimeoutError):
        return None

    yes = no = 0.0
    for entry in top:
        word = entry["token"].strip().lower()
        if word.startswith("yes"):
            yes += math.exp(entry["logprob"])
        elif word.startswith("no"):
            no += math.exp(entry["logprob"])
    return yes / (yes + no) if (yes + no) > 0 else None


def unload(model: str) -> None:
    """Never leave a model sitting in VRAM."""
    body = json.dumps({"model": model, "keep_alive": 0}).encode()
    try:
        urllib.request.urlopen(
            urllib.request.Request(f"{OLLAMA}/api/generate", data=body, headers={"Content-Type": "application/json"}),
            timeout=20,
        ).read()
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=HERE / "data" / "features.npz")
    ap.add_argument("--captions", type=Path, default=HERE / "data" / "captions")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--window", type=int, default=10, help="marker fuzziness, in caption lines each way")
    ap.add_argument("--candidates", type=int, default=5, help="how many seams the model adjudicates")
    ap.add_argument("--question", choices=sorted(QUESTIONS), default="ad", help="which question to put to the model")
    args = ap.parse_args()

    data = np.load(args.features, allow_pickle=False)
    names = list(data["feature_names"])
    seam_col = names.index("seam_here")
    X, video, line = data["X"], data["video"], data["line"]
    is_start, secs = data["is_start"], data["start_seconds"]

    texts: dict[str, list[str]] = {}
    for path in args.captions.glob("*.json"):
        caps = json.loads(path.read_text(encoding="utf-8"))
        texts[caps["videoID"]] = [l["text"] for l in caps["lines"]]

    span = 2 * args.window + 1
    seam_err, model_err, disagreements, calls, failures = [], [], 0, 0, 0
    record: list[dict] = []
    started = time.time()

    for vid in np.unique(video):
        rows = np.flatnonzero(video == vid)
        order = rows[np.argsort(line[rows])]
        seam = X[order, seam_col]
        t = secs[order]
        lines = texts.get(str(vid))
        if not lines:
            continue

        for n, i in enumerate(np.flatnonzero(is_start[order])):
            offset = 3 + (n * 5) % (span - 6)  # vary where the start sits, so nothing is centred
            lo, hi = i - offset, i - offset + span
            if lo < 0 or hi > len(order):
                continue

            # MiniLM nominates; the model only ever adjudicates these.
            local = np.argsort(seam[lo:hi])[::-1]
            picks = [lo + int(c) for c in local if BLOCK <= lo + int(c) - 0 and lo + int(c) + BLOCK <= len(order)][
                : args.candidates
            ]
            if not picks:
                continue

            seam_pick = picks[0]
            seam_err.append(float(t[seam_pick] - t[i]))

            changed, scored = [], []
            verdicts = []
            for c in picks:
                a = " ".join(lines[int(line[order[j]])] for j in range(c - BLOCK, c))
                b = " ".join(lines[int(line[order[j]])] for j in range(c, c + BLOCK))
                prompt = QUESTIONS[args.question].format(a=a, b=b)
                if args.question.endswith("_lp"):
                    kind, verdict = "score", ask_logprob(args.model, prompt)
                else:
                    kind = "score" if args.question == "score" else "bool"
                    verdict = ask_ollama(args.model, prompt, kind)
                calls += 1
                if verdict is None:
                    failures += 1
                elif kind == "score":
                    scored.append((c, verdict))
                elif verdict:
                    changed.append(c)
                verdicts.append({"line": int(c), "seconds": float(t[c]), "value": verdict})

            if scored:  # graded: take the candidate the model rates highest
                best = max(s for _, s in scored)
                model_pick = min(c for c, s in scored if s == best)
            else:
                model_pick = min(changed) if changed else seam_pick  # earliest accepted boundary
            # Keep every raw verdict, so a different decision rule can be tried
            # later without spending another five minutes of GPU on the same calls.
            record.append({
                "video": str(vid), "true_start_line": int(i), "true_start_seconds": float(t[i]),
                "seam_pick_seconds": float(t[seam_pick]), "candidates": verdicts,
            })
            model_err.append(float(t[model_pick] - t[i]))
            if model_pick != seam_pick:
                disagreements += 1

            print(
                f"  {str(vid):12s} start@{t[i]:7.1f}s   seam {t[seam_pick] - t[i]:+6.1f}s   "
                f"qwen {t[model_pick] - t[i]:+6.1f}s   ({len(changed) or len(scored)}/{len(picks)} answered)"
            )

    unload(args.model)
    out = args.features.parent / f"pairwise_verdicts_{args.question}.json"
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"raw verdicts -> {out}")

    def show(name: str, errs: list[float]) -> None:
        e = np.abs(np.array(errs))
        print(
            f"  {name:16s} median {np.median(e):5.1f} s   worst {e.max():5.1f} s   "
            f"within 2 s {100 * (e <= 2).mean():4.0f}%   within 5 s {100 * (e <= 5).mean():4.0f}%"
        )

    print(f"\n=== {len(seam_err)} read starts, {calls} model calls, {failures} unusable, "
          f"{(time.time() - started) / 60:.1f} min ===")
    show("MiniLM seam", seam_err)
    show(f"seam + qwen [{args.question}]", model_err)
    print(f"\nqwen moved the pick on {disagreements} of {len(seam_err)} windows.")
    print("Model unloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
