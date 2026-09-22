"""Label a transcript by sweeping every window of it past Claude.

Simon's idea, 2026-09-20: rather than depending on SponsorBlock's coverage,
have Claude mark the training data itself. That frees the dataset from
SponsorBlock entirely -- any timestamped transcript becomes trainable.

WHY A FULL SWEEP, and not "let the marker propose, Claude confirm":
the cheap version would hand Claude only the regions the cue patterns already
found. Claude would then confirm them accurately and the training set would
contain exactly the reads a regex can find -- rebuilding, by our own hand, the
coverage bias we are trying to escape. The student can only learn what the
teacher was shown. So every window gets looked at, whether anything looks
interesting in it or not.

Windows are the format Claude is measurably good at: on 2026-09-20 it placed
6 of 6 read starts within 2.2 s when given a window and asked for a LINE INDEX,
against a 33-minute error when given a whole 82,000-character transcript. Line
indices are also checkable -- an answer outside the window it was handed is
rejected rather than trusted.

Run:
  python marker/claude_label.py --limit 5              label 5 videos
  python marker/claude_label.py --score                compare labels to SponsorBlock
"""

import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

WINDOW_LINES = 60   # ~2.5 minutes of captions, the size measured at 0.6 s median
STEP_LINES = 40     # overlap, so a read straddling a cut is seen whole at least once
MAX_READ_SECONDS = 180     # the longest plausible host-read ad; SponsorBlock's median is 49 s
MAX_READ_FRACTION = 0.35   # more of the video than this is its subject, not an ad in it

ASK = """You are given numbered caption lines from part of a YouTube video.

A SPONSOR READ is an advertisement that INTERRUPTS the video. The host is talking about the video's actual subject, stops, reads an ad for a company that paid them, and then returns to the subject. The show resumes afterwards.

These are NOT sponsor reads, and this is the distinction that matters most:
- A REVIEW, unboxing, demonstration or tutorial about a product. If the product IS the video's subject, nothing is interrupted, so there is no sponsor read -- however enthusiastic or price-quoting the host gets.
- A corporate, promotional or brand video, where the whole video exists to promote a company.
- The host's own merch, Patreon, memberships, or their other videos.
- A mention of a sponsor in passing ("thanks to our sponsors") without an actual ad being read.

Ask yourself: does the video go BACK to something else afterwards? If the surrounding lines are about the same product too, it is the subject, not an advertisement.

Answer ONLY with JSON: {"start_line": <number>, "end_line": <number>}
  start_line = the number of the FIRST line of the sponsor read.
  end_line   = the number of the first line where the actual show RESUMES after it.
If there is no sponsor read in these lines at all, answer {"start_line": null, "end_line": null}."""


def ask_claude(text: str, model: str = "haiku", timeout: float = 120.0) -> dict | None:
    try:
        return _ask(text, model, timeout)
    except subprocess.TimeoutExpired:   # one slow answer must not end the whole sweep (2026-09-22)
        return None


def _ask(text: str, model: str, timeout: float) -> dict | None:
    proc = subprocess.run(
        ["claude", "-p", "--model", model, "--tools", "", "--strict-mcp-config",
         "--no-session-persistence", "--output-format", "json"],
        # encoding is NOT optional here: on Windows, text mode defaults to cp1252,
        # and a prompt containing any non-ASCII character fails inside subprocess's
        # writer THREAD, which run() never raises. Claude then answers a truncated
        # prompt and the caller sees a confident, wrong result. Measured 2026-09-20:
        # it silently blanked a Polish-language video's whole transcript.
        input=text, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    try:
        result = json.loads(proc.stdout).get("result", "")
        match = re.search(r"\{[\s\S]*?\}", result.replace("```json", "").replace("```", ""))
        return json.loads(match.group(0)) if match else None
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None


def render(lines: list[dict], lo: int, hi: int) -> str:
    return "\n".join(f"{i - lo:3d}: {lines[i]['text']}" for i in range(lo, hi))


def label_video(caps: dict, model: str, verbose: bool = True) -> dict:
    """Every window of one transcript, swept past Claude. Returns merged segments."""
    lines = caps["lines"]
    found: list[tuple[int, int]] = []
    windows = rejected = 0

    for lo in range(0, len(lines), STEP_LINES):
        hi = min(lo + WINDOW_LINES, len(lines))
        if hi - lo < 12:
            break
        windows += 1
        answer = ask_claude(f"{ASK}\n\n{render(lines, lo, hi)}", model=model)   # before 2026-09-22 10:40 the model was never passed: every run was Haiku
        if not answer or answer.get("start_line") is None:
            continue
        start, end = answer.get("start_line"), answer.get("end_line")
        # An answer that does not resolve inside the window it was given is wrong
        # by construction, so it is thrown away rather than trusted.
        if not isinstance(start, int) or not (0 <= start < hi - lo):
            rejected += 1
            continue
        if not isinstance(end, int) or not (start < end <= hi - lo):
            end = min(start + 30, hi - lo)
        found.append((lo + start, lo + end))

    # A "sponsor read" covering a large share of the video is the video's SUBJECT,
    # not an advertisement in it. Measured 2026-09-20: Claude marked 152 s of a
    # 6-minute app-demo video as a sponsor read. Structure catches what the prompt
    # may not, so both are used.
    span = lines[-1]["start"] - lines[0]["start"] if len(lines) > 1 else 0
    found = [
        (a, b) for a, b in found
        if (lines[min(b, len(lines) - 1)]["start"] - lines[a]["start"]) <= MAX_READ_SECONDS
        and (not span or (lines[min(b, len(lines) - 1)]["start"] - lines[a]["start"]) / span <= MAX_READ_FRACTION)
    ]

    merged: list[list[int]] = []
    for start, end in sorted(found):
        if merged and start <= merged[-1][1] + 3:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    segments = [
        {
            "start_line": a,
            "end_line": b,
            "start": round(lines[a]["start"], 2),
            "end": round(lines[min(b, len(lines) - 1)]["start"], 2),
        }
        for a, b in merged
    ]
    if verbose:
        print(f"  {caps['videoID']}  {len(lines):5d} lines  {windows:3d} windows  "
              f"{len(segments)} segment(s)  {rejected} rejected")
    return {"videoID": caps["videoID"], "windows": windows, "rejected": rejected, "segments": segments}


def score(labels: list[dict], truth: dict) -> None:
    """How well do Claude's labels agree with SponsorBlock's human ones?"""
    matched = missed = extra = 0
    start_errors = []
    for row in labels:
        real = truth.get(row["videoID"], [])
        mine = row["segments"]
        used = set()
        for r_start, r_end in real:
            hit = next((i for i, s in enumerate(mine)
                        if i not in used and s["start"] < r_end and s["end"] > r_start), None)
            if hit is None:
                missed += 1
            else:
                used.add(hit)
                matched += 1
                start_errors.append(mine[hit]["start"] - r_start)
        extra += len(mine) - len(used)

    total = matched + missed
    print(f"\nagainst SponsorBlock on {len(labels)} videos, {total} human-marked reads:")
    print(f"  Claude found        {matched}/{total} ({100 * matched / max(total, 1):.0f}%)")
    print(f"  Claude also marked  {extra} segment(s) SponsorBlock does not have")
    if start_errors:
        errs = sorted(abs(e) for e in start_errors)
        print(f"  start error         median {errs[len(errs) // 2]:.1f} s   worst {errs[-1]:.1f} s   "
              f"within 5 s {sum(1 for e in errs if e <= 5)}/{len(errs)}")
    print("\nSegments Claude marks that SponsorBlock lacks are not necessarily wrong --")
    print("SponsorBlock's coverage is the thing we are trying to beat. They need reading.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captions", type=Path, default=HERE / "data" / "captions")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "claude_labels.json")
    ap.add_argument("--candidates", type=Path, default=HERE / "data" / "candidates.json")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--score", action="store_true", help="score existing labels instead of making more")
    ap.add_argument("--workers", type=int, default=1,
                    help="videos labelled in parallel. Each call is a subprocess waiting on the network, "
                         "so this costs little CPU and cuts wall time roughly linearly.")
    ap.add_argument("--shuffle", type=int, default=0,
                    help="seed for a random video order. A HOLDOUT must be sampled randomly, not "
                         "alphabetically, or it is not a fair test of anything.")
    ap.add_argument("--cue", choices=["only", "none"], default=None,
                    help="restrict to transcripts that do (only) or do not (none) contain a stock sponsor phrase. "
                         "'none' measures what the cue pre-filter would DROP: reads Claude finds in videos no "
                         "regex would have flagged. That number decides whether the filter is safe to use.")
    args = ap.parse_args()

    truth = {v["videoID"]: v["segments"] for v in json.loads(args.candidates.read_text(encoding="utf-8"))}

    if args.score:
        labels = json.loads(args.out.read_text(encoding="utf-8"))
        score(labels, truth)
        return 0

    done = {}
    if args.out.exists():
        done = {row["videoID"]: row for row in json.loads(args.out.read_text(encoding="utf-8"))}

    paths = [p for p in sorted(args.captions.glob("*.json")) if p.stem not in done]
    if args.cue:
        want = args.cue == "only"
        paths = [p for p in paths if json.loads(p.read_text(encoding="utf-8")).get("hasCue", False) is want]
    if args.shuffle:
        random.Random(args.shuffle).shuffle(paths)
    paths = paths[: args.limit]
    print(f"labelling {len(paths)} videos with claude-{args.model}, sweeping every window")
    started = time.time()
    lock = threading.Lock()

    def one(path: Path) -> None:
        caps = json.loads(path.read_text(encoding="utf-8"))
        row = label_video(caps, args.model)
        # Write after every video, so a night's work survives being stopped.
        with lock:
            done[caps["videoID"]] = row
            args.out.write_text(json.dumps(list(done.values()), indent=1), encoding="utf-8")

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(one, paths))
    else:
        for path in paths:
            one(path)

    print(f"\n{len(done)} videos labelled in total, this run took {(time.time() - started) / 60:.1f} min")
    score(list(done.values()), truth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
