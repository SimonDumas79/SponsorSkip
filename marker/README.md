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

Named-channel data (`channel_videos.py` -> `candidates_channels.json`, captions in
`captions_channels/`) is built with `--all-to channels --keep-selfpromo-only
--not-in` every other examples file. The flag matters only there: the channel
scan keeps videos whose only SponsorBlock marks are self-promo, and without it
`build_dataset.py` drops them (it judges "labels found nothing" on sponsor marks
alone so the graded builds never change; verified byte-identical 2026-09-21).

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

### Short reads: diagnosed, and the obvious fixes measured (2026-09-21, `short_reads.py`)

They are **missed, not cut short**: on CV, 30 of the 35 reads under 30 s are never
touched by the context model at its threshold. The marker itself does score them
(median peak 0.81 under 15 s, 0.98 at 15-30 s) but the context model, reading 15
lines either side, averages a 4-10 line bump away (median peak 0.64 and 0.91,
threshold 0.978). The edge heads are not the problem: they cut none of the found
short reads and grew 4. Half the reads under 15 s are `selfpromo` one-liners.

What stops the threshold from dropping is the **worst-video cap**, not the budget:
at every threshold below 0.978 one video crosses 60 s, and it is a different video
each step (a machine-translated Russian MMO video about the in-game store; a
Windows 11 feature roundup with "partnership with 1Password"; an ASMR video). Those
are genuine false positives, not label gaps, and none has a region over 64 s, so a
region-length cap (180 / 240 s) or a share-of-video cap (35 / 50%) changes nothing.

Measured on CV and NOT adopted, all inside the noise of 121 reads (one read = 0.8
points) or worse:

| lever | ad time / show lost (B = 10) | short reads (<15 s, 15-30 s) |
|---|---|---|
| shipped free tier | 44.5% / 4.9 s | 12%, 21% |
| edge heads' inward reach capped by region length | 45.0% / 4.9 s | 12%, 21% |
| context model with 3- and 7-line summaries added | 45.9% / 5.8 s | 12%, 25% |
| context model weighted so each read counts once | 38.4% / 3.9 s | 17%, 25% |
| both | 43.8% / 6.1 s | 23%, 30% |
| edge heads' OUTWARD reach cut (20 -> 10 / 5 / 0 lines) | 36.7-42% / 3.2-4.9 s | worse |
| translated-title videos left to SponsorBlock | 42.6% / 3.8 s | same |

On the pooled 205-video set (280 reads, 80 short; `--pooled`, build_production's
2%-share rule, B = 10) the same three context-model variants give: baseline 43.4%
of ad time at 6.2 s with short reads at 5% / 20%; multi-scale 44.8% at 7.2 s;
read-weighted 41.7% at 7.2 s with short reads at 21% / 32%; both 43.5% at 7.8 s
with short reads at 21% / 34%. Read-weighting quadruples what short reads get,
but pays for it in long reads and lost show: total ad time does not move.

**Would more of the same data help?** (`learning_curve.py`, pooled set subsampled by
channel, three draws each, B = 10): 25% of channels 29.7% of ad time, 50% 37.6%,
75% 44.3%, 100% 43.6%. Steep to 75%, flat after, with a spread of about 5 points
between draws, so "flat" is inside the noise but the gains are clearly slowing.
More SponsorBlock-tail crawl of the same kind is at diminishing returns; the data
worth having is a different kind (hard negatives, labels owing nothing to
SponsorBlock), or a different signal. **The video description is that signal**,
probed on 25 CV videos with sponsor reads (`descriptions_probe.json`): tokens taken
from the description alone (link domains, the word after "code", capitalised names on
link lines) land inside 33 of 44 reads, while only 20% of the caption lines they hit
are inside a read, so it is a feature for the context model, not a rule.

**Built and measured, not adopted** (`fetch_descriptions.py`, `add_descriptions.py`,
`description_eval.py`; all 215 labelled videos' descriptions in `data/descriptions.json`,
two columns `desc_hit_line` / `desc_hits_context` appended to every feature file, 71%
line precision on the CV set with the tightened tokens in `features.description_tokens`).
On the 77-video CV set the marker with the columns gave 45.9% of ad time at 3.6 s lost
(shipped: 44.5% at 4.9 s, worst video 54 s -> 32 s). On the pooled 205-video set it did
NOT replicate: 31.7% at 4.0 s against 43.6% at 6.2 s, because the 2%-share cap forced
the threshold up (worst video 117 s); holdout 1's descriptions hit at only 44% line
precision. The marker alone improved on both sets (34.5 -> 38.9%, 39.6 -> 42.2%), the
system did not. The columns stay in the feature files for later experiments but the
shipped marker ignores them: `experiments.kept_columns` uses the first 395 unless a
config sets `use_description`.

The heads' outward reach is what wins ad time on real reads; on a false region it
widens the damage (44 s -> 144 s on the Windows video), and that is the price, not a
bug. Cutting it loses far more than it saves.

**Machine-translated caption tracks are a class of their own**: non-Latin titles are
7 of 77 CV videos, 0 of 78 in holdout 1, 6 of 50 in holdout 2 and 4 of 10 in the
first holdout 3. They hold the worst video in holdout 2 and every qwen system's
cap-breaker in holdout 1. yt-dlp reports the spoken language (`%(language)s` = `ru`
for bqtppv75MJg, and the `en` track's URL carries `tlang=`), so both fetchers now
record `language`; the decision to fall back to SponsorBlock on non-English videos
is Simon's, and the older caption files carry no field (the title's script is the
proxy for them).

Also fixed there: `edge_heads.place` merges overlapping placed spans, because two
nearby regions often land on the same start line once moved (one video had the
same span three times) and `predict.py` was sending each copy to the extension.

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
