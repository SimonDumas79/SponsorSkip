# marker

The cheap detector that finds *candidate* sponsor regions, so Claude only has to
read a short window around each one instead of a whole transcript.

Why the split exists: on 2026-09-20 Claude, reading an 82,000-character
transcript, placed a sponsor read at 1616 s when it starts at 3616 s — it quoted
the read correctly and mistyped the number. Given a window and asked for a *line
index*, the same model placed all six test reads within 2.2 s. So: a marker
shortlists, Claude verifies in a window, and any answer that resolves outside
its own window is rejected.

**The marker is judged on recall, not accuracy.** A false alarm costs one 7-second
Claude window. A miss costs a whole sponsor read played at full volume. Those are
not the same size of mistake, so they are not scored the same way.

The full explanation, with diagrams:
<https://claude.ai/artifact/AxjiintXBAVa4rbeyfX2aN>

## Running it, in order

```bash
pip install --user sentence-transformers scikit-learn   # torch + yt-dlp already present

python marker/sample_segments.py --prefixes 40     # SponsorBlock -> data/raw_segments.json
python marker/pick_videos.py --take 450            # filter        -> data/candidates.json
python marker/fetch_captions.py --limit 300        # yt-dlp, slow  -> data/captions/*.json
python marker/fetch_selfpromo.py                   # SponsorBlock  -> data/selfpromo.json
python marker/build_dataset.py --not-in marker/data/examples_tail.jsonl   # label -> data/examples.jsonl
python marker/features.py                          # MiniLM        -> data/features.npz
python marker/score.py --baseline                  # the number to beat
python marker/train.py                             # CV by channel, thresholds, marker.pt
python marker/train.py --holdout                   # ONCE, at the end: the holdout grade
```

The holdout is built the same way from `candidates_tail.json` and `captions_tail/`
with `build_dataset.py --all-to holdout` and `features.py --out data/features_tail.npz`.
Build it FIRST: the training build's `--not-in` reads it to keep every holdout
channel out of training.

Every step is resumable and skips work already done, so a stopped fetch can just
be run again. Nothing in `data/` is committed.

**Labels are SponsorBlock's `sponsor` AND `selfpromo` segments**, because the
extension skips both by default. Measured 2026-09-21: with sponsor-only labels the
marker's worst "false alarm" was 215 s of a host's own app launch, and the metric
called it lost show.

`train.py` trains the linear marker and chooses its two thresholds by rule from
channel-grouped cross-validation (with a checker: highest threshold keeping 90%
recall; alone: lowest threshold flagging under the show budget). `experiments.py`
searches training and post-processing levers with a genetic algorithm, judged by
recall across budgets of 1-15 s of lost show per video, and never touches the
holdout; frontier candidates are saved to `data/zoo/` with their out-of-fold
scores so they can be cloned, mutated or stacked later.

## The system of models (2026-09-21)

The marker is one of several small models, each with one job. Every stage after
the marker is trained on OUT-OF-FOLD scores with the same channel folds, and
every language-model answer is recorded once and replayed after that.

| Job | Script | What it is |
|---|---|---|
| detect | `train.py` | the linear marker: one score per caption line |
| detect, in context | `context_stack.py` | a second stage reading the marker's logits for 15 lines either side: it learns the shape of a read |
| confirm, free | `region_judge.py` | a logistic model over each flagged region's facts (length, confidence, cues, seam, position) |
| confirm, local GPU | `confirm_check.py` | qwen3 says yes/no per flagged window; every answer recorded to `data/confirm_verdicts*.jsonl` |
| place edges | `edge_heads.py` | start and resume models trained on `is_start` / `is_resume` |
| place edges, local GPU | `qwen_edges.py` | qwen3 gives the start line and the resume line, as Claude does in `server/verify.mjs` |
| choose and grade | `replay.py`, `system_eval.py` | routing rules scored over the recorded answers for free (the Dream-RSI idea); settings chosen by one written rule on CV, graded once per holdout (`--set holdout`, `--set holdout2`) |
| ship | `predict.py` | the free tier end to end on one caption file (`run`), on stdin for the server (`serve`); `check` proves the live path cuts exactly the evaluator's regions |
| ship, all data | `build_production.py` | the free tier retrained on all 205 labelled videos: `data/production/free_tier_pooled.pt`, the candidate for the next fresh test |

`data/production/free_tier.pt` is what the server's opt-in `?reader=marker` loads (0.7.0).
It is the graded system (a copy is kept as `free_tier_graded.pt`); replace it with the
pooled bundle only after that passes a fresh test.

**Measured, 2026-09-21** (share of ad time skipped / real show lost per video):

| | holdout 1 (78 videos) | holdout 2 (50 videos) |
|---|---|---|
| cue patterns | 28.0% / 19.3 s | 30.5% / 14.4 s |
| marker alone | 31.5% / 8.3 s | 35.0% / 8.3 s |
| free tier (context model + edge heads) | 49.6% / 9.9 s | 49.3% / 7.5 s |
| qwen tier (judge + qwen confirm + qwen edges) | 63.4% / 19.9 s | 63.4% / 9.9 s |

Open problems, in order: reads under 30 s (the free tier skips 13-28% of them, the
marker alone 17%); one outlier video per set losing 130-170 s (garbled translated
captions; a video *about* ads); and the worst-video rule, which must be a share of
videos (as in `build_production.py`), not a maximum, once sets grow.

**Grade on ad TIME, not just reads.** Region recall counts a read as found if one
line of it is skipped, so a policy that skips a tight core looks excellent while
most of each ad plays: measured 2026-09-21, a rule touching 82% of reads skipped
only 26% of ad time. `replay.py` reports `ad cov` (share of all ad seconds skipped)
beside recall, and the system is chosen to maximise ad time skipped under a
budget of real show lost per video, with a cap on the worst single video.

## Notes worth keeping

- **SponsorBlock's CSV dumps are switched off** (bandwidth); the hash-prefix API
  is the way in, and it never learns which video is being asked about.
- **A 429 from YouTube is usually one awkward video, not a block on us** — measured
  here at roughly one video in ten. The fetcher backs off twice, gives up on that
  video, and only stops entirely after five refusals in a row.
- **Captions are fetched exactly as `server/youtube.mjs` fetches them** (same
  `SUB_LANGS`, same json3 parsing). If the training data were cleaner than what
  the extension sees live, the measured accuracy wouldn't survive the trip into
  production.
- **The split is by channel, never at random.** SponsorBlock covers big channels
  with conventional reads best; a model trained on it inherits that bias, and
  only a channel-held-out validation set will show it.
