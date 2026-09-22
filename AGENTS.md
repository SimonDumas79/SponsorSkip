# Connecting an AI model to SponsorSkip

SponsorSkip always runs the marker first. The marker is local, free, needs no network and no
language model: it finds where the sponsor reads are. A model is optional and does exactly one job
on top of that — saying where each read it found **starts and ends**.

That split is measured, not a guess. On 64 fresh videos from 64 channels:

| who places the edges of the same 93 regions | ad time skipped | real show lost per video |
|---|---|---|
| nobody (the raw region) | 53.1% | 3.3 s |
| the marker's own edge heads | 68.4% | 7.9 s |
| **claude-haiku** | **71.2%** | **4.6 s** |
| qwen3:8b on a local GPU | 62.0% | 5.4 s |

Two things follow, and they are the whole reason this file exists.

**A model earns its place on edges, not on detection.** The same Claude added as another *detector*
alongside the marker's six was measured on the same day and changed nothing (64.4% against 64.6%).
If you are wiring a model in, wire it to the edges.

**A weak model is worse than no model.** qwen3:8b loses 6 points of ad time against the marker's own
edge heads and breaks the 60-second cap. It is not offered in the extension for that reason. Do not
assume a local model helps; measure it against the table above before offering it.

## The contract

A reader is a process. It reads one JSON object on stdin and writes one on stdout. Nothing else.

```jsonc
// in
{ "videoID": "abc123", "channel": "UC...", "duration": 1011.0,
  "lines": [ { "start": 0.0, "text": "..." } ],   // the transcript, one caption line each
  "chapters": [ { "title": "Sponsor", "start_time": 412.0 } ],   // may be null
  "description": "..." }                                         // may be null

// out
{ "segments": [ { "start": 412.5, "end": 471.0, "category": "sponsor" } ] }
```

`category` is `"sponsor"` or `"selfpromo"`. Seconds, not line numbers. An empty list is a valid
answer and means "no sponsor read in this video" — it must never be how a failure reports itself.

## Adding yourself as a reader

1. **Answer the contract.** `marker/predict.py serve --tier <name>` is the reference implementation;
   `server/agents.mjs` shows how it is spawned.
2. **Register the tier** in three places: the `--tier` choices in `marker/predict.py`, `AGENTS` and
   `MARKER_TIMEOUTS` in `server/agents.mjs`, and the reader list in `server/server.mjs`.
3. **Offer it** in `extension/popup.html`, with its progress steps in `extension/popup.js` and its
   status line in `extension/content.js`.
4. **Grade it before offering it**, on a set of channels that are in no training file. `marker/
   cascade.py --edges` is the harness: it takes the regions the marker found, asks your model for
   each one's edges, and prints ad time and show lost beside the marker's own placement. Run it and
   put the numbers in `marker/README.md` whether they are good or bad.

## What the model is told, and what is done with its answer

It gets a window of numbered caption lines around one region and returns two line numbers. It is
never handed the whole transcript for this job — that is a different, more expensive reader.

Its answer is trusted **only near the region the marker found**: at most 20 lines before the region's
start, 10 lines into it, and symmetrically at the end (`edge_heads.BEFORE/INSIDE/TAIL/AFTER`). An
answer outside that falls back to the marker's own edge heads. So a confused model cannot cut
somewhere random; the worst it can do is be ignored.

## Rate limits, timeouts and outages

**The layer degrades to the free tier and no further.** Every region the model does not answer —
usage limit, timeout, malformed JSON, an answer outside the window — keeps the marker's edge heads.
It never falls back to the raw region: that is 53.1% against 68.4%, so a silent raw fallback would
quietly cost 15 points the first time the model was rate-limited.

**Serving fails fast.** `claude_label.ask_claude` retries for up to six minutes, which is correct for
an overnight batch and wrong for a video somebody is waiting on, so the serving path passes
`attempts=1, timeout=60`. Use the same rule for any model you add.

**A failed call must never read as "no read here."** Return an error the caller can see. This has
bitten the project twice: a usage-limited sweep once looked merely cautious and its numbers were
believed for hours.

## House rules for anything you add here

- Run commands from the repo root; scripts open `marker/data/...` by relative paths.
- Never choose between systems by their holdout numbers. Tune on cross-validation, grade a holdout
  once, and write the result down even when it is bad. `marker/README.md` records the rejections
  too, and they are the more useful half.
- Do not leave a model loaded on the GPU. The local-model scripts unload when they finish.
- The extension must not fight a game for the GPU. Serving is CPU by design.
