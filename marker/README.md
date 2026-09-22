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

### Not marking, or wrong length? (2026-09-21, `error_budget.py`)

Pooled out-of-fold scores, 205 videos, 280 reads, free tier at the pooled rule's
B = 10 threshold (0.982): 43.4% of ad time skipped, 6.2 s of show lost per video.

| Ad time NOT skipped, share of all ad time | |
|---|---|
| reads never marked at all (134 of 280, median 26 s) | 36.1% |
| found, but the skip starts late | 14.1% |
| found, but the skip ends early | 4.9% |
| holes inside found reads | 1.5% |

**Mostly not marking, but the marker did see them.** Of the 134 unmarked reads,
101 have a line the marker scores >= 0.84; only 13 (2.5% of ad time) never pass
0.5. The context model's threshold throws them away, and the threshold is that
high because of false alarms: 18 regions in 17 videos, all content ABOUT products
(a Pixel review, a watch roundup, a Monopoly history, a scam-exposure video, a
Game Pass debate). Lost show splits 53% overshoot around real reads, 47% false
alarms. Found reads get a median 61% of their time skipped; the missing part is
mostly the START: 31 regions begin more than 20 lines after the true start (out of
the start head's reach, 29 min of ad) and 66 have the true start in reach but the
head picks the wrong line (15 min).

**Ceilings** (threshold rechosen by the same rule with one stage made perfect):

| | ad time | show lost / video |
|---|---|---|
| as shipped | 43.4% | 6.2 s |
| perfect ends only | 49.8% | 5.2 s |
| perfect yes/no checker only (no false alarms; threshold drops to 0.943) | 58.7% | 6.8 s |
| perfect starts only | 62.8% | 4.9 s |
| perfect edges on found reads | 68.2% | 3.8 s |
| perfect checker AND edges (74% of reads touched at the loosest threshold) | 82.0% | 0.0 s |

**Wider start search, measured and rejected**: 20 -> 30 lines before the region
gives 45.1% (+1.7, noise); 40 or 60 lines give 37-41% because on a false region the
wider reach widens the damage and the rule pushes the threshold up. Every free
lever pays this tax. The headroom is in a checker that removes false alarms and
places starts, which is Claude's job (15 of 15 reads, 1.3 s median start error)
and partly qwen's (the qwen tier's 63.4% on both holdouts sits inside this ceiling).

### Every detector compared, then combined: stacking works (2026-09-21, `detector_bakeoff.py`, `stack_check.py`)

Simon's question: compare the marker with the other detectors we built, and combine
them all. Seven free detectors, each out of fold on the pooled 205 videos with the same
channel folds, each given its own context model and the edge heads, pooled rule:

| Detector, B = 10 | ad time | show lost / video |
|---|---|---|
| cue patterns + context model | 43.5% | 9.5 s |
| description signal + context | 23.5% | 5.5 s |
| structure only (seam, cues, position) + context | 41.6% | 6.7 s |
| meaning only (MiniLM) + context | 45.4% | 7.4 s |
| potion-base-8M marker + context | 44.9% | 6.7 s |
| **shipped marker + context** | 41.7% | 6.0 s |
| sequence model + context | 39.6% | 5.4 s |

Alone they are all within a few points. They are NOT alike: meaning and the marker
correlate 0.93, but cues, description and structure 0.16-0.55 with the rest, and the cue
patterns touch 37 reads the marker's free tier misses (potion 27, structure 28).

Combinations, B = 10: skipping every detector's regions (union) gets 59.2% but loses
17.4 s per video with 7.3% of videos over 60 s, so it fails the rule. Averaging the
logits gets 42.0% at 3.8 s. Letting a second detector veto the free tier's regions
("agreement") gets 43-48%, inside the noise. **One context model reading all five
learned detectors' scores in its window** gets about 50%.

`stack_check.py` then tested it three ways:

| | shipped, 3 seeds | stack, 3 seeds |
|---|---|---|
| B = 5 | 37.0 / 38.4 / 40.5% | 42.5 / 47.1 / 48.3% |
| B = 10 | 39.0 / 41.3 / 44.6% | 49.5 / 49.9 / 52.4% |

Lost show is the same, 4.7-6.7 s per video. The worst stacked seed beats the best
shipped seed at both budgets. Read by read at B = 10: the stack skips more of 41 reads,
less of 19 (sign test p = 0.006); it touches 23 reads the shipped tier misses entirely,
and misses 10 that it touches.

Ablation, one seed each, so read differences under ~3 points as noise: the raw cue and
description columns add nothing (all five detectors without them: 50.6% / 44.0%); no
single addition does it (marker + potion 45.6%, marker + sequence 46.5%, marker + meaning
+ structure 45.8%); dropping potion or the sequence model costs about 3 points each. The
gain comes from combining several different views, not from one of them.

**Status: a candidate, not shipped.** It was chosen on pooled CV after trying ~15
combinations, so it needs a fresh test before it replaces anything. At serving time it
needs potion's embeddings (a ~8 MB lookup table) and three more small models on the
same features; all CPU-cheap.

### Fine-tuning MiniLM helps; the window alone does not (2026-09-22, `finetune_minilm.py`, `finetune_eval.py`, `window_control.py`)

`finetune_minilm.py` trains every weight of all-MiniLM-L6-v2 on (line, +-8 lines) pairs, same
pooled channel folds, 2 epochs, ~13 min per seed on the 3080. Settings fixed before any result.

| Line level | average precision | ROC AUC |
|---|---|---|
| frozen marker (shipped) | 0.600 | 0.907 |
| frozen MiniLM given the same 17-line window (`window_control.py`) | 0.606 | 0.917 |
| fine-tuned MiniLM, seed 0 / seed 1 | 0.676 / 0.686 | 0.938 / 0.935 |

The window alone does nothing; training the encoder is what helps. Through the pipeline
(context model + edge heads, pooled rule):

| Ad time skipped | B = 5 | B = 10 |
|---|---|---|
| shipped free tier, 3 context seeds (`stack_check.py`) | 37.0-40.5% | 39.0-44.6% |
| fine-tuned free tier, seeds 0 / 1 (better of shipped or own heads) | 40.9 / 46.3% | 54.4 / 51.4% |
| stack of 5 detectors, 3 seeds | 42.5-48.3% | 49.5-52.4% |
| **stack + fine-tuned, seeds 0 / 1** | **48.8 / 48.5%** | **53.1 / 52.8%** |

Read by read at B = 10 against the shipped tier: seed 0 53 better / 7 worse, seed 1 70 / 39
(p = 0.004). Lost show stays 4.5-8.8 s per video. It does NOT cut false alarms (16-21 regions
against 18). Which edge heads suit it flips between seeds, so read that choice as noise; the
stack with the fine-tuned scores added is the steadiest result. Candidate only: CV, needs the
fresh test.

### "Does the show resume?" has no signal; "is the region an island?" does (2026-09-22, `resume_check.py`)

Mean MiniLM embeddings over 12 lines either side, 232 real reads against 27 free-tier false-alarm
regions. Before-vs-after similarity separates them 45% of the time (50% = none): real reads often
sit between two subjects (Simon's point), so the show does not "resume" to the same topic. The
region's similarity to its CLOSER side separates them 75% of the time (reads 0.65, false alarms
0.74): a read is unlike both neighbours, product talk resembles at least one. Not built yet.

### Watch-page hints: chapters yes, the paid-promotion flag no (2026-09-22, `fetch_watch_meta.py`, `watch_meta_eval.py`)

One GET of each watch page reads YouTube's "Includes paid promotion" flag (paidContentOverlay in the
player response; yt-dlp does not expose it) and the creator's chapters. Pooled 205 videos, all of
which have a read:

- **The flag is on only 44% of them**, so creators mostly do not tick it. As a gate (a looser
  threshold on flagged videos, both thresholds chosen by the rule) it does nothing: 39.0% vs 38.4%
  at B = 5, 43.4% vs 44.6% at B = 10. Not adopted. Its absence says little; it may still matter
  on sponsor-free videos (to be measured on the negatives crawl).
- **Chapters are on 35%, and 20 of them are titled for a read** ("Sponsor", "Ad"...). All 20 sit
  on a labelled read (20/20), covering 20 of 280 reads, with edges a median 1.9 s (start) and
  1.3 s (end) from SponsorBlock's. Used as edges (and as a skip even with no region): 40.7% vs
  38.4% at B = 5, 46.5% vs 44.6% at B = 10. **Adopted**: free, creator-stated, no false alarm seen.
  yt-dlp returns chapters in the same call that fetches captions.

### Joint start/end pairs: no gain (2026-09-22, `joint_edges.py`)

Scoring each (start, end) pair together, with the region's interior evidence weighted by ALPHA
and TAU chosen by the rule, picked ALPHA = 0 every time: identical to the separate start/end
models on the shipped tier (41.7%) and within 0.1 point on the stack (49.8 vs 49.7%).

### The island filter: real signal, no gain at the chosen thresholds (2026-09-22, `island_check.py`)

Skipping a region only when it is unlike its closer side (cutoff chosen with the threshold by the
rule): shipped tier 38.4 -> 38.4% (B = 5), no cutoff helps at B = 10; 5-detector stack +0.9 / +0.7
points (inside noise); stack + fine-tuned MiniLM (both seeds): no cutoff helps. It removes at most
one false-alarm region. The 75% separation was measured at a loose threshold; the false alarms left
at the rule's threshold are the ones it cannot tell apart. Not adopted as a filter.

Joint pairs on stack + fine-tuned: +1.7 at B = 5 (ALPHA 0.1, TAU 2, the best of 21 settings) and
nothing at B = 10; read as selection noise. Not adopted.

### Edge labels checked by Claude: the noise is real, cleaning it does not help (2026-09-22, `verify_edges.py`, `edge_clean.py`)

Simon's point: SponsorBlock marks are set against the video and then mapped onto caption lines, so
edges can land off the true line. Claude (Sonnet, `claude -p`, SponsorBlock's range not shown) placed
260 of the 280 pooled reads: median gap 0 s at both edges, but only 60% of starts and 69% of ends
within 2 s, 10% of each more than 10 s apart, and only 53% of reads agree within 3 s at BOTH edges.
For 18 labelled reads (6%) Claude found no promotional read in the window. Reading the disagreements,
neither side is always right (Claude keeps lead-ins like "a quick word from..."; SponsorBlock keeps
story-style lead-ins and adjacent self-promo).

Retraining the start/end models with disputed edges masked, or with Claude's edges, changes nothing
that matters: shipped tier 44.6% at B = 10 as shipped vs 44.1% masked vs 43.0-43.3% with Claude's
edges; stack + fine-tuned 53.1% vs 52.8% vs 51.1%. On the reads where both agree, the placed start is
still a median 3.6-4.3 s off in every variant. What limits starts is the start model, not label noise.
Kept: `data/edge_verify.jsonl` (the 139 fully agreed reads are the trustworthy edge benchmark).

### Fine-tune seeds, seed averaging, a 7-detector stack, English scope (2026-09-22, `finetune_compare.py`, `scope_english.py`)

Stack + fine-tuned MiniLM over three fine-tune seeds: 47.3-48.8% (B = 5) and 52.8-53.7% (B = 10) of
ad time. Averaging the three seeds' logits into one detector (47.0 / 55.9%), and a 7-detector stack
that also carries the single-line fine-tune (47.1 / 54.2%), land inside that band: not adopted. The
single-line fine-tune (window 0) ranks lines far worse alone (AP 0.372 vs 0.676) but its stack gave
46.1 / 56.5% on one seed: a second seed is queued.

**Language.** `video_language.json` (langdetect on each video's original-language title and
description; agrees with yt-dlp's field on all 169 videos that record one): 64 of 205 pooled videos
(30%) are not English, and 6-7 of the 17 false-alarm videos are among them (machine-translated
tracks). The best system graded on English videos only: **53.4% (B = 5) / 56.7% (B = 10)** against
48.8 / 53.1% on all. Training the context model on English only is worse (44.3-47.3 / 51.4-52.9%):
less data costs more than cleaner data gains, so training stays on everything. If non-English
videos fall back to SponsorBlock (Simon's decision), English is the free tier's scope.

False alarms of the best system at B = 10: 17 regions in 16 videos from 16 different channels.

### Hard-negative mining: fewer false alarms, less ad time (2026-09-22, `hardneg.py`)

The rooms' first build. The best system (stack + fine-tuned) at a loose threshold (10% of lines)
makes 459 false regions (3,676 show lines); those lines get weight W in the context model's loss:

| W | B = 5 | B = 10 | false regions at B = 10 | fixed / newly broken |
|---|---|---|---|---|
| 1 (reference) | 48.8% @ 4.9 s | 53.1% @ 7.0 s | 17 | |
| 3 | 50.0% @ 4.8 s | 51.7% @ 5.8 s | 14 | 7 / 4 |
| 10 | 46.4% @ 4.9 s | 47.6% @ 5.0 s | 10 | 9 / 2 |

It removes false alarms, but the model grows cautious on real reads too, so the rule cannot turn
the room into more ad time; 9 fixed is under the rooms' 13-of-18 gate. Not adopted on the pooled
rule. It is the first thing to try on the sponsor-free panel, where every false alarm is pure loss.

### Conditional widening and a GRU over the detectors' scores: not adopted (2026-09-22, `widen_gate.py`, `seq_stack.py`)

- **Conditional widening** (a region still needs the strict threshold, then its edges walk outwards
  while the score stays above a looser LOW threshold, before the edge heads place it; LOW and the
  threshold chosen together by the rule): the rule picked "no widening" every time, on the shipped
  tier and on stack + fine-tuned, back-only and both ends. The out-of-window starts stay unsolved.
- **GRU over the six detectors' logits** (2-layer bidirectional, inside / start / resume heads on one
  trunk, trained on 300-line chunks): 45.8-46.5% (B = 5) and 54.5-56.7% (B = 10) against the stack's
  48.8 / 53.1%: worse at one budget, better at the other, inside the band. Its own start/resume heads
  did not beat the shipped edge heads. Not adopted; worth one retry with the BGE stream as input.

### Sponsor-free videos: false alarms are rare (2026-09-22, `sample_negatives.py`, `negatives_eval.py`)

Every pooled video has a read, so the rule never saw a video that should be left alone.
`sample_negatives.py` asks SponsorBlock's hash-prefix API for EVERY category and keeps videos with
marks (intro, outro, filler...) but no sponsor or self-promotion; tier B has 2+ marks in 2+
categories or a locked/upvoted one (621 candidates, 189 tier B; captions crawling). The saved
bundles run end to end on them (first 57 tier-B videos, 44 English):

| | no skip at all | mean skipped | > 10 s | > 60 s |
|---|---|---|---|---|
| graded free tier (`free_tier.pt`) | 93% | 2.6 s | 5% | 2% (one video) |
| pooled free tier (`free_tier_pooled.pt`) | 98% | 0.3 s | 2% | 0% |

Both pass the research round's target (>= 90% untouched, <= 5% over 10 s). The one bad case is
104 s on a "How To Use VideoProc Converter" tutorial: a product-subject video, and possibly an
unmarked sponsored one (Claude check pending). False alarms concentrate in product-talk videos,
which is where hard negatives should come from. Updated as the crawl reaches 190.

Claude check (Sonnet whole-video sweep, the data room's random + flagged design): of 26 randomly
chosen tier-B videos Claude found a sponsor read in 1 (a 15 s read): about 4% label noise, so the
panel is clean enough to trust. Of the 4 videos the graded tier flagged, Claude found a read in
none: they are genuine false alarms (the VideoProc tutorial among them).

### Audits the rooms asked for (2026-09-22, `audits.py`)

- **Channels**: 203 channels for 205 videos (2 with a second video), so channel-grouped folds are
  already near leave-one-video-out; channel repetition cannot inflate the CV numbers.
- **Channel-clustered bootstrap** (2000 resamples of channels) of the gain over the shipped tier, each
  system at its own rule choice: stack + fine-tuned MiniLM +9.0 to +10.5 points (B = 5) and +8.2 to
  +9.1 (B = 10) over three seeds, every interval clear of zero; **stack + fine-tuned BGE-small (seed 0)
  +13.8 [+9.8, +18.1] and +14.1 [+10.6, +17.7]**. What the bootstrap cannot remove: every system was
  chosen on these same folds, so the size still needs a fresh many-channel test.
- **Stability**: of the reads each stack + fine-tuned seed skips 5+ points more of (53 / 55 / 57), 45
  win in every seed; the gain is not a reshuffle.
- **Misses** (B = 10, Wilson 95%): complete misses 47% [41, 53] shipped, 39-40% stack + fine-tuned,
  32% [27, 38] stack + BGE.
- **Start error vs caption gaps**: Spearman +0.04 on 72 agreed reads; the ~4 s start floor is not a
  caption-gap effect.

### BGE-small beats MiniLM as the fine-tuned detector; recipe tweaks do not (2026-09-22, `finetune_compare.py`)

Same recipe (2 epochs, line + 8 lines either side), only the encoder changed. Stack = the five
bake-off detectors + the fine-tuned run, one context model, pooled rule:

| fine-tuned run | line AP | stack B = 5 | stack B = 10 |
|---|---|---|---|
| MiniLM-L6, seeds 0 / 1 / 2 | 0.676 / 0.686 / 0.660 | 48.8 / 48.5 / 47.3% | 53.1 / 52.8 / 53.7% |
| **BAAI/bge-small-en-v1.5, seeds 0 / 1 / 2** | **0.730 / 0.747 / 0.732** | **52.1 / 48.1 / 49.3%** | **58.7 / 57.3 / 57.0%** |
| MiniLM + layer-wise LR decay 0.9 | 0.678 | 48.0% | 56.9% |
| MiniLM + R-Drop (alpha 5) | 0.697 | 50.1% | 53.3% |
| MiniLM + top 2 layers re-initialised, 4 epochs | 0.657 | 49.9% | 56.3% |
| MiniLM, single line (window 0) | 0.372 | 46.1% | 56.5% |

At B = 10 every BGE seed beats every MiniLM seed (by about 4 points); at B = 5 the gap is smaller
and mixed. The recipe tweaks stay in MiniLM's band. **Adopted: fine-tuned BGE-small as the sixth
detector.** Serving cost: 33M parameters, 12 layers, about twice MiniLM-L6 per window (ONNX int8 and
embedding each line once are the known ways back down). **BGE-base (110M, 60 min per run) does not
beat it**: line AP 0.752, stack 48.6% / 57.5% (seed 0), inside BGE-small's band at three times the size.

With chapters and in English scope (three BGE seeds): + creator chapters 47.3-50.2% (B = 5) and
58.2-60.0% (B = 10): +1.2 to +1.3 points at B = 10 in every seed, mixed at B = 5 (chapter spans
spend lost-show budget). Graded on English videos only: 51.4-53.9% / 59.3-63.2%.

Which detectors still earn their place with BGE in (`bge_ablation.py`, BGE seeds 0 / 1, B = 5 / B = 10):
BGE alone 41.4 / 56.0% and 42.9 / 48.0%; marker + BGE 46.4 / 50.0% and 48.3 / 53.2%; marker + meaning
+ structure + BGE (one frozen encoder + BGE) 47.5 / 56.1% and 49.9 / 58.3%; all six 52.1 / 58.7% and
48.1 / 57.3%. The one-encoder + BGE stack is within noise of all six (behind by 4.6 / 2.6 points on
seed 0, ahead by 1.8 / 1.0 on seed 1), so potion and the conv model are optional: the cheaper build
costs nothing measurable. BGE alone and marker + BGE are clearly worse.

### Fine-tuned BGE start/resume models: more precise starts (2026-09-22, `edge_ft_eval.py`)

`finetune_minilm.py --target start|resume --model BAAI/bge-small-en-v1.5` trains BGE on the same soft
start/resume labels the MLP edge heads use; its scores replace or join the heads (same windows).
Start error is measured on the reads where Claude and SponsorBlock agree:

| system, edges | B = 5 | B = 10 | start error at B = 10 |
|---|---|---|---|
| shipped tier, MLP heads | 38.4% | 44.6% | median 4.3 s, 29% within 2 s |
| shipped tier, fine-tuned heads | 41.1% | 46.4% | median 2.6 s, 40% within 2 s |
| stack + fine-tuned, MLP heads | 48.8% | 53.1% | median 3.6 s, 37% within 2 s |
| stack + fine-tuned, fine-tuned heads | 47.1% | 48.0% | median 2.4 s, 45% within 2 s |
| stack + fine-tuned, average of both | 48.5% | 56.4% | median 2.4 s, 44% within 2 s |

The first thing to move the ~4 s start floor: fine-tuning is what the start model needed, not
cleaner labels. Ad time on the stack moves inconsistently (one seed); the averaged heads are the
candidate to confirm with more seeds. Not in the pre-registered candidate.

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
