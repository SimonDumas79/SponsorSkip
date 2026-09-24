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

### Shipping the candidate (2026-09-22)

Measuring a system and serving it are two different jobs, and for a day and a half only the first one
was done. These are the steps that put a measured system in front of a user:

```bash
bash marker/data/export_chain.sh                          # GPU, ~15 min: the three fine-tuned
                                                          # checkpoints (inside, start, resume)
python marker/export_candidate.py                         # CPU: everything else -> data/production/candidate.pt
python marker/export_candidate.py --verify holdout4       # does the bundle reproduce the grade?
python marker/export_candidate.py --from-checkpoints holdout4   # does the SIBLING the checkpoints
                                                          # hold reproduce it too? (slow, CPU)
python marker/serve_candidate.py check holdout4 --limit 6 # captions in, the evaluator's regions out
python marker/predict.py serve --tier candidate           # what server/agents.mjs calls
```

Each check answers a different question and none of them is optional. `--verify` reads the prebuilt
feature files, so it tests the bundle's weights and thresholds. `--from-checkpoints` recomputes the
three fine-tuned streams from the saved weights, which matters because they come from a fresh
training run of the same recipe and GPU training is not bit-identical between runs. `check` starts
from a raw caption file and tests the whole serving path, including the three encoders: a mismatch
there means the serving path builds features differently from the way the training data was built,
which would cost accuracy in the extension and nowhere else.

The reader is `marker-candidate` in the extension popup and through the local server. Creator
chapters are fetched by the same `yt-dlp` call that fetches the captions and applied when serving.

## The system of models (2026-09-22)

Three layers, and the point of the shape is that each one only does what the layer below cannot.
Every stage after a detector is trained on OUT-OF-FOLD scores with the same channel folds, and every
language-model answer is recorded once and replayed after that.

**Layer 1, the cheap sweep** — five detectors read every line of the video: the linear marker over
395 MiniLM features (`train.py`), the same model with meaning or with structure removed, a linear
model over the potion-base-8M embedding, and a 1-D convolution over the whole video
(`sequence_model.py`). All small, all CPU.

**Layer 2, the fine-tuned reader** — BGE-small fine-tuned on our labels, reading each line plus 8
lines either side (`finetune_minilm.py`), plus two more of the same shape trained on where reads
START and RESUME. It only reads the 30% of lines layer 1 saw something near; on holdout 4 that costs
nothing at all and does a third of the work. A context model stacks all six detectors' scores over 15
lines either side (`context_stack.py`), and the edge heads place each region's start and end.

**Layer 3, optional, a language model on edges only** — the regions are already found; Claude is
asked, once per region and given a window around it, only where that region starts and ends. It is
never asked to detect: as a seventh detector it changed nothing (64.4% against 64.6%). Anything it
does not answer keeps layer 2's edges, so the layer degrades to the free tier and no further.

| Job | Script | What it is |
|---|---|---|
| detect, cheap | `train.py`, `sequence_model.py`, `detector_bakeoff.py` | five small models, one score per line |
| detect, fine-tuned | `finetune_minilm.py` | BGE-small on line + 8 either side; the sixth detector |
| gate | `export_candidate.py`, `serve_candidate.py` | the cheap five decide which lines layer 2 reads |
| stack | `context_stack.py` | one context model over all six, trained on out-of-fold scores |
| place edges | `edge_heads.py` + two fine-tuned models, averaged | start and resume, trained on `is_start` / `is_resume` |
| place edges, language model | `cascade.py`, `serve_candidate.py` | Claude, one window per region, trusted only near it |
| creator chapters | `candidate.py` | a chapter the creator titled "Sponsor"; free, no false alarm in any set |
| export | `export_candidate.py` | trains every part on all 205 videos into `data/production/candidate.pt` |
| serve | `predict.py serve --tier` | `free`, `candidate`, `cascade`, `qwen` |
| grade | `replay.py`, `cascade.py`, `export_candidate.py --verify` | ad time skipped, real show lost, share of videos over 60 s |

**Where the numbers come from, and what they are worth.** On holdout 4 (64 videos, 64 channels, none
in any other set), graded once as pre-registered:

| | ad time skipped | show lost per video |
|---|---|---|
| cue patterns, no model | 30.1% | 13.0 s |
| `free_tier.pt`, what the extension served until today | 40.3% | 10.8 s |
| the candidate (six detectors, averaged edge heads) | 67.6% | 8.6 s |
| the candidate + the community SponsorBlock model | 67.2% | 5.5 s |
| the cascade (Claude on edges) — **development number, not a grade** | 71.2% | 4.6 s |
| Claude reading the whole transcript | 81.9% | 5.3 s |

The cascade's row is marked because the idea was developed against that set after it was graded; its
honest test is holdout 5 (`build_holdout5.sh`, pre-registration 3), which waits on new videos.

Open problems, in order: **missed reads** are the big basket (36% of ad time, and 17% of reads are
never touched at all); reads under 30 s specifically; non-English videos, where the machine-translated
caption track holds the worst video in every measured set; and the fact that the noise band on a
205-video set is about 3 points, which is wider than anything the last ten experiments moved. That
last one is why more labelled data, not more architecture, is the only lane still worth running.


### The system as it stood on 2026-09-21 (kept for the holdout 1 and 2 numbers)

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

Claude check (whole-video sweep, the data room's random + flagged design; **Haiku, not Sonnet as first
written**: `claude_label.py` never passed `--model` to the call until 2026-09-22 10:40): of 26 randomly
chosen tier-B videos Claude found a sponsor read in 1 (a 15 s read): about 4% label noise, so the
panel is clean enough to trust. Of the 4 videos the graded tier flagged, Claude found a read in
none: they are genuine false alarms (the VideoProc tutorial among them).

**The candidate on the panel** (`candidate.py --fresh features_negatives.npz`, 132 tier-B videos from
129 channels, none in training; the shipped bundles over all 146 captioned videos):

| system | no skip at all | mean skipped | > 10 s | > 60 s |
|---|---|---|---|---|
| candidate, B = 5 / B = 10 | 98% / 97% | 0.6 / 0.7 s | 2% / 3% | 0% |
| candidate v2, B = 5 / B = 10 | 98% / 97% | 0.5 / 0.7 s | 2% | 0% |
| graded free tier (shipped) | 94% | 1.8 s | 5% | 1% |
| pooled free tier | 97% | 0.7 s | 3% | 0% |

The candidate takes 10-20 more points of ad time on sponsored videos and is as clean as the best
shipped bundle where there is no ad at all.

With the community model on the same 132 videos: **the community model alone skips something on 17%
of them** (83% untouched, mean 2.9 s, 11% over 10 s, 2% over 60 s). Candidate v3, which uses it as a
seventh detector, stays clean: 97-98% untouched, mean 0.3-0.5 s, 1-2% over 10 s, 0% over 60 s.

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

On the BGE stack (three stack seeds, one fine-tuned edge seed), averaging the MLP heads with the
fine-tuned start/resume models: B = 10 62.0 / 62.4 / 61.8% against 58.7 / 57.3 / 57.0% (+3.3 to +5.1 in
every seed), median start error 3.7-3.8 s -> 2.2-2.5 s; B = 5 mixed (49.6 / 50.3 / 51.0 against
52.1 / 48.1 / 49.3). Pre-registered as candidate v2 for holdout 4.
Three edge seeds x three BGE stack seeds, nine combinations: 49.6-51.6% (B = 5) and 59.6-63.0%
(B = 10). The averaged heads do not depend on one lucky edge model.

### The community SponsorBlock model on our videos (2026-09-22, `sponsorblock_ml.py`, `sbml_eval.py`)

xenova/sponsorblock-ml (T5-small extracts sponsor text from ~500-token chunks, a BERT classifier
filters each find; trained on SponsorBlock's database up to early 2022), run with its own chunking
and parsing code on our caption files, graded by our rule:

| set | ad time | show lost / video | videos over 60 s |
|---|---|---|---|
| pooled 205 (may overlap its training labels) | 74.3% | 18.6 s | 7.8% |
| channel set 169 (3 channels; recent uploads) | 72.7% | 4.1 s | 1.2% |

On the pooled set it fails the rule at every classifier cut (its lowest lost show is 17.0 s per
video), but it skips far more ad time. On the channel set it beats our pre-registered candidate
without chapters (64.0% at 6.3 s, B = 10) though not with chapters (91.5%). Leakage caveat: these
are popular channels with years of SponsorBlock marks, so it has likely learned their older reads
(same sponsors, same scripts); our models never saw them. `sb_dates.py` measures which videos'
labels predate its training; holdout 4 (low-review tail videos) is the fair comparison.

**Leakage measured** (`sb_dates.py`: SponsorBlock's searchSegments gives each segment's submission
time): 45 of the 205 pooled videos, and none of the 169 channel videos, had a sponsor/self-promo
segment before 2022-04. On the 160 pooled videos neither system has seen labels for:

| system | ad time | show lost / video | videos over 60 s |
|---|---|---|---|
| community SponsorBlock model | 73.4% | 17.0 s | 6.9% |
| ours, stack + BGE, B = 5 / B = 10 | 51.8% / 58.4% | 5.1 / 9.4 s | 0.0 / 1.2% |
| ours, v2 averaged heads, B = 5 / B = 10 | 49.1% / 61.5% | 5.4 / 9.5 s | 0.0 / 1.2% |

(On the 45 older videos it does 78.3% at 24.5 s.) Leakage does not explain its recall: it finds much
more ad, at nearly twice our lost show, and breaks the 2% cap. It is a different trade-off, and
a different view (long chunks, text extraction): `stack_sbml.py` tests it as a seventh detector.

### The community model as a seventh detector: the biggest single gain (2026-09-22, `stack_sbml.py`)

Its finds become a line score (the classifier's sponsor/self-promo probability on lines inside a
find). Trained AND graded only on the 160 pooled videos whose labels postdate its training, so it
cannot have learned our answers; channel-grouped 5-fold CV inside them; averaged edge heads:

| BGE seed | stack + BGE, B = 5 / B = 10 | + community model, B = 5 / B = 10 |
|---|---|---|
| 0 | 50.1% / 53.1% | **60.7% / 64.6%** |
| 1 | 51.3% / 59.8% | **55.5% / 66.3%** |
| 2 | 51.9% / 55.2% | **59.3% / 67.0%** |

+4 to +11 points at B = 5 and +7 to +12 at B = 10 in every seed, within the rule (4.8-8.3 s lost per
video). The two see different things (short line windows vs long extracted chunks), and the context
model learns when to trust each. At the community model's own cost (<= 17.0 s lost per video) the
combination ties it (73.0-74.1% vs 73.4%); below that cost the community model cannot go at all,
and the combination gives 64.6-67.0% at 7-8 s. Cost: the community model is a 77M T5 plus a BERT classifier, about
11 s per video on the GPU. **On the CPU (8 threads) it measured 7.7 and 8.8 s for two 8-minute
videos**, about 1 s per minute of video: heavier than the free tier's ~10 s, but a workable slower
free option for typical 10-20 minute videos; a 90-minute podcast would take ~1.5 min.
Fine-tuned BGE-small on the same CPU: 16.6 s for 1000 windows (~a 35-minute video) in fp32, 9.1 s with
int8 dynamic quantisation (accuracy of the int8 model not yet checked). So the candidate is ~15-20 s
per 35-minute video on a desktop CPU, and v3 (with the community model) about a minute.

### The Claude tier beats the community model (2026-09-22, `claude_eval.py`)

The extension's default reader is Claude Haiku. `claude_label.py --model haiku` (the research window
sweep, not word for word the server's prompt) on the 160 pooled videos whose labels postdate the
community model's training; graded by our rule on the same videos:

| alone, as it answers | ad time | show lost / video | videos over 60 s |
|---|---|---|---|
| **Claude Haiku** | **79.1%** | **9.3 s** | 5.0% |
| community SponsorBlock model | 73.4% | 17.0 s | 6.9% |

Claude finds more ad at about half the lost show. Both break the 2% over-60 s cap. (Caveat found later
the same day: `claude_label.py` read a failed call as "no read" until 13:05, so any failures in this
Haiku run count as misses: its 79.1% is if anything an underestimate. A first "Sonnet" run of the same
160 videos lost about half its answers that way, most likely to the subscription's usage limit, and is
discarded, not reported.)

**Sonnet, measured properly** (40 of the same videos, 62 reads, the failure-aware labeller, 0 failed
windows):

| the same 40 videos | ad time | show lost / video | over 60 s | found a read in |
|---|---|---|---|---|
| Claude Haiku (the shipped default) | 79.0% | 13.6 s | 7.5% | 39 / 40 |
| **Claude Sonnet** | **83.4%** | **9.4 s** | **5.0%** | 40 / 40 |

Sonnet takes 4 points more ad time while losing 4 s less show per video, and breaks the 60 s cap
less often: the best reader measured, and better than the community model on both axes. Its cost is
subscription usage, not latency (about 7 s per window either way). Switching the extension's default
(`server/agents.mjs`, `SPONSORSKIP_MODEL`) is Simon's call.

**On holdout 4** (the fresh 64 channels, graded with everything else): Sonnet 81.9% of ad time at
5.3 s lost per video, 1.6% of videos over 60 s, a read found in 57 of 64; English only 85.5% at
6.9 s. The community model on the same videos: 77.0% at 9.9 s, 3.1% over 60 s. Sonnet is the best
reader measured, on both axes, and the only strong one inside the 2% cap. As an extra
detector in the stack (context model trained and graded on the 160, seed 0): + Claude 59.7% (B = 5) /
67.8% (B = 10); + community model 60.7 / 64.6%; + both 51.8 / 65.5% (one seed, 160 videos: likely
too many inputs for the data). The rule-bound stacks trade some of Claude's recall for staying
under the cap.

At Claude's own cost (<= 9.3 s) the stacks with Claude inside reach only 69.4-70.0% (the context model
blurs Claude's exact segments); at the community model's cost (<= 17 s) 74.9-76.5%, level with it.
Around Claude's segments instead (`claude_combo.py`): adding the stack's regions 79.6%, vetoing weak
Claude segments 78.9%, both 79.5% at 8.9 s: nothing moves, and nothing gets under the 2% cap.
Claude's 8 videos over 60 s are mostly false alarms (6 of 8), product talk again (a "leaving GitHub"
video, a gadget review, a sports podcast, an "AI side hustle" video). Sonnet is being measured on
the same 160 to see whether a stronger model removes them.

### qwen3:8b as a detector (2026-09-22, `qwen_sweep.py`, `qwen_eval.py`, `overlap.py`)

qwen answered all 3,026 windows of the 205 pooled videos (yes/no plus the read's line range),
recorded once. Alone, as it answers: **70.4% of ad time but 50.0 s of show lost per video, 28.3% of
videos over 60 s**: high recall, poor precision. In the stack + BGE (averaged heads, three BGE
seeds): B = 5 49.8 / 56.5 / 49.8% vs 49.6 / 50.3 / 51.0% without it; B = 10 61.8 / 65.5 / 64.2% vs
62.0 / 62.4 / 61.8%: a small, inconsistent gain (0 to +3 points at B = 10). The overlap analysis with
qwen in (Simon's question from 2026-09-21): qwen catches **22.7% of all ad time that no other
detector catches**; everything together catches 79.8% but would lose 66 s of show per video.

### Fine-tuning the community T5 on our labels: worse (2026-09-22, `t5_finetune.py`, `t5ft_eval.py`)

Warm-started from Xenova/sponsorblock-small, trained in its own chunk/target format on our labels
(2 epochs, lr 5e-5, positives and negatives 50/50), channel-grouped 5-fold CV on the 160 unseen
pooled videos. Alone: 64.9% of ad time at **34.3 s** lost per video (20% of videos over 60 s),
against the original's 73.4% at 17.0 s. As the stack's seventh detector: 45.6-47.4% (B = 5) and
49.9-58.1% (B = 10) against the original's 55.5-60.7% and 64.6-67.0%. 160 videos (~850 chunks per
fold) are too few: it forgets what years of SponsorBlock data taught it and over-flags. Not adopted;
the original community model stays as the detector.

### Walking outwards from inside the read, sentence by sentence: not adopted (2026-09-22, `walk_edges.py`)

Simon's rule, and the obvious one: find the ad, then step backwards a line at a time while each line
still looks like the ad, stop when it clearly does not, then the same forwards. `widen_gate.py` had
already tried this and always chose not to walk, but on the CONTEXT model's score, which is averaged
over 31 lines and so stays high well past the true edge. This walks on the **fine-tuned per-line
score** instead, the sharp one, and stops only after PATIENCE lines in a row below a line threshold,
so a single quiet line inside the read does not end the walk. Line threshold and patience were chosen
by the pooled rule alongside the detection threshold, exactly as the shipped placement's are.

Ad time skipped, three BGE stack seeds, pooled 205 videos:

| placement | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| averaged heads (v2), B = 5 | 49.6% | 50.3% | 51.0% |
| walk outwards, B = 5 | 49.7% | 49.3% | 48.0% |
| averaged heads (v2), B = 10 | 62.0% | 62.4% | 61.8% |
| walk outwards, B = 10 | 60.6% | 56.1% | 60.9% |

It loses five of six comparisons: about 1 point at B = 5 and about 3 at B = 10. Edge error on the
agreed reads is no better either, start 2.7 s against 2.2-2.4 s for the averaged heads, end 2.6 s
against about 1.8 s. **Not adopted.**

Worth saying why, because the idea is sound and the model is not: the rule always chose PATIENCE 1,
meaning it stopped at the very first line that did not look like the ad. Every more patient setting
scored worse. A walk that stops at the first quiet line is just a threshold crossing found the slow
way, and the heads already find it while reading the whole neighbourhood at once. The sequential
version can only use what it has passed; the head sees both sides of the candidate edge at once, and
that is the information that places an edge well. This is the second test of the walk idea and the
second rejection, on a sharper score than the first.

### Layer the models: the cheap five gate the expensive three (2026-09-22, `cascade.py --gate`) — ADOPTED

Simon, 2026-09-22: "It should be layered, cheap quick model does the initial early sweep, then a
stronger cheap model to get the other markers down, then if selected, either local or Claude will
begin parsing out the sponsored sections."

The candidate was not layered. All six detectors read every line of every video, so the fine-tuned
BGE models (one detector and two edge models, 33M parameters each) were paid across the whole video,
including the nine tenths of it that is obviously the show. The gate: the five cheap detectors sweep
first, take their strongest score within 15 lines, and only lines above a cut go to the fine-tuned
three. Everything else takes that model's median score. Same stack, same fixed thresholds.

Holdout 4, 64 fresh videos, B = 10 with the averaged edge heads:

| share of lines the fine-tuned models read | ad time | show lost | over 60 s |
|---|---|---|---|
| **all of them (the system as measured)** | **68.4%** | **7.9 s** | 0.0% |
| 50% | 68.4% | 7.9 s | 0.0% |
| **30%, all three models gated** | **68.4%** | **7.9 s** | 0.0% |
| 20%, detector only | 68.3% | 7.9 s | 0.0% |
| 20%, all three gated | 66.9% | 7.9 s | 0.0% |
| 10%, detector only | 62.2% | 6.9 s | 1.6% |
| 5%, detector only | 44.4% | 3.7 s | 0.0% |

**Adopted at 30%.** The grade does not move at all there, to the decimal, so two thirds of the
expensive work was waste. 20% is fine for the detector alone but costs 1.5 points once the edge
models are gated too, because `place` searches 20 lines past a region while the gate reaches 15 --
and gating only the detector is worse overall, since the other two then still read every line. At
10% the detector itself starts missing reads. So 30% is a floor, not a dial.

The cut is the reach score's 70th percentile on the POOLED data, stored in the bundle, so a video is
never judged against itself; a video with no ad in it simply sends fewer lines through. Serving
applies it in `serve_candidate.CandidateTier.regions`.

### Layer 3: Claude places the edges of the regions we found (2026-09-22, `cascade.py --edges`)

The other half of Simon's design, and the first real gain since the community model. The cheap layers
detect; Claude is asked, once per region, only where that region starts and ends. It is handed a
window around the region and answers with two line numbers, and an answer outside the trust window
falls back to the region, so a wild number cannot cut somewhere random.

This is NOT Claude as a seventh detector, which was measured the same day and did nothing (64.4%
against 64.6%). Detection was never Claude's advantage over the stack. Placement is.

Holdout 4, 64 fresh videos, 93 regions found at B = 10, claude-haiku, 0 failed calls:

| placement of the same regions | ad time | show lost | over 60 s | full reads |
|---|---|---|---|---|
| none (the raw region) | 53.1% | 3.3 s | 0.0% | 8% |
| our averaged edge heads | 68.4% | 7.9 s | 0.0% | 38% |
| **claude-haiku edges** | **71.2%** | **4.6 s** | 0.0% | **46%** |
| Claude reading the whole video, for reference | 81.9% | 5.3 s | 1.6% | — |

**Both axes at once**: 2.8 more points of ad time AND 42% less real show lost. Nothing else measured
this month moved both. 59 of the 93 answers were taken; the rest fell outside the trust window or
said there was no read there, and kept the heads' edges.

What it costs. A full-transcript read on a median video from our corpus (17 min, 18,700 characters)
was measured at $0.037 on Haiku and $0.074 on Sonnet, ~45 s each, and the extension sends two such
calls per video. Layer 3 sends one window per region instead, about 1.5 regions per video at a few
thousand characters: roughly a tenth of the text for the same model. It does not reach Claude reading
the whole video (71.2% against 81.9%) but it loses less show doing it, and detection stays local.

**The local model cannot do this job** (`cascade.py --edges --local`, qwen3:8b, same 93 regions):
62.0% of ad time at 5.4 s lost, against our own heads' 68.4% at 7.9 s and Claude's 71.2% at 4.6 s.
It took MORE of the offered edges than Claude did (70 of 93 against 59) and placed them worse: 18%
of reads fully covered against Claude's 46%, and it broke the 60 s cap on one video. It narrows
regions rather than placing them, which buys show back by giving up ad. `--tier cascade-local` was
built and then taken out of the extension: shipping it would have offered a reader measurably worse
than the free tier it sits above.

**That first comparison was not clean, and Simon caught it**: Claude was asked with `cascade.ASK`
(which spells out that the read starts where the host leaves the subject and that the turn into the
ad belongs inside) while qwen got the older, thinner `qwen_edges.ASK`. Asked exactly what Claude was
asked (`local_edges.py --same-prompt`, same 93 regions):

| qwen3:8b placing the same regions' edges | ad time | show lost | reads fully covered |
|---|---|---|---|
| its own old prompt | 62.0% | 5.4 s | 18% |
| **Claude's prompt** | **62.6%** | **3.3 s** | **26%** |
| our averaged edge heads, for comparison | 68.4% | 7.9 s | 38% |
| the raw region, for comparison | 53.1% | 3.3 s | 8% |

The prompt was worth something -- a third fewer seconds of show lost and half again as many reads
fully covered -- and it does not close the gap: still 5.8 points of ad time behind our own heads. Note
where it lands: at 3.3 s it has exactly the RAW region's show loss, meaning it is barely widening the
region at all. The instruction was a real handicap; the model is the limit.

Worth noting against the raw row: the regions themselves only cover 53.1% of ad time. Placement is
carrying 18 points, which is why the edges were the right place to spend a language model.

### The cascade, validated honestly on pooled CV (2026-09-22, `cascade_pooled.py`)

Holdout 4's 71.2% was a development number: the idea was tuned against that same set afterward, so it
could not be trusted as a grade. This checks the cascade's DIRECTION the proper way, on the pooled
205 videos' out-of-fold scores, a coarse 3-point threshold sweep (claude-haiku, 811 distinct regions,
23 failed calls, all falling back to our own edge heads as designed):

| pooled CV | our averaged heads | claude edges (cascade) |
|---|---|---|
| B = 5 | 51.2% ad time, 5.0 s lost | **53.1%, 3.8 s** |
| B = 10 | 59.9% ad time, 7.2 s lost | **62.7%, 5.4 s** |

Better on both axes at both budgets, on the same reads (touched count is identical, so this is a
placement gain, not a detection one). Smaller than holdout 4's number, as expected once the
data-mining pressure is gone, but real and in the same direction. Thresholds chosen by the rule and
fixed in `data/cascade_thresholds.json` (B=5 0.9962, B=10 0.9898) BEFORE any fresh set is touched, so
holdout 5 (pre-registration 3) can now be graded honestly whenever the crawl produces it.

### The walk, tuned on holdout 4: still behind our own heads at every setting tried (2026-09-22, `local_edges.py --walk`)

The first walk setting (patience 2, group 3, reach 30 lines) hit 77.3% of ad time -- more than
Claude's cascade -- but lost 33.4 s of show per video and broke the 60 s cap on 22% of videos: it
does not know how to stop. A sweep of patience/group/reach on the same 93 development regions:

| patience, group, reach | ad time | show lost | over 60 s |
|---|---|---|---|
| 2, 3, 30 (first try) | 77.3% | 33.4 s | 21.9% |
| 2, 3, 12 | 72.7% | 18.5 s | 6.2% |
| 1, 3, 12 | 69.2% | 14.0 s | 3.1% |
| 1, 2, 8 | 68.5% | 11.1 s | 1.6% |
| **1, 3, 6** | **66.2%** | **10.1 s** | **1.6%** |
| our averaged edge heads, for comparison | 68.4% | 7.9 s | 0.0% |

Every setting still loses more show than our own edge heads for the same or less ad time; the
tightest (reach 6) is the closest but still costs 2.2 s more per video with a nonzero over-cap rate.
The walk fixed the counting failure (removing arithmetic let the model take MORE ad time than any
free system measured, at the loose setting) but it still cannot judge "the show has resumed"
reliably, so shrinking its leash trades the gain back away rather than keeping it. **Not adopted for
the local tier.** Worth a second look if a stop-criterion that isn't just "N misses in a row" is tried
(e.g. asking the model to compare against the region's own tone instead of a bare yes/no).

### Grounding the walk in what the show is: made it worse, not better (2026-09-22, `local_edges.py --anchor`)

Simon's diagnosis: the walk asks "is this still the show" without ever telling the model what the
show is, so a generic transitional line gives it nothing to check resumption against. His follow-up:
a video's title, or even its opening, may not describe what the SPECIFIC section after the ad is
about, especially once a host has moved to a new topic. Three groundings were tried, all at the best
swept setting (patience 1, group 3, reach 6 lines), all on the same 93 development regions:

| anchor | ad time | show lost | over 60 s | reads fully covered |
|---|---|---|---|---|
| none | 66.2% | 10.1 s | 1.6% | -- |
| title only | 69.2% | 13.8 s | 1.6% | 24% |
| title + the video's opening (5 lines) | 69.2% | 14.9 s | 3.1% | 22% |
| **title + the host's own voice right before the ad** (Simon's refinement, tone not topic) | 68.8% | 12.8 s | 1.6% | 19% |
| our averaged edge heads, for comparison | 68.4% | 7.9 s | 0.0% | 38% |

Every anchored variant took slightly more ad time than the plain walk and lost meaningfully more
show doing it. The pre-ad tone sample (built specifically against the risk that a host changes topic
across the ad break, and phrased as "how the host talks," not "what they must be talking about") was
the best of the three groundings but still worse than no anchor at all, and its full-coverage rate is
the lowest of the four. Grounding the question did not make the model more decisive about resumption;
if anything it made it more willing to call ambiguous lines "still the ad," which is the opposite of
what was intended.

**Not adopted, in any form tried.** The local walk's ceiling on this development set stays the
plain, unanchored version at 66.2% / 10.1 s, itself already behind our own heads (68.4% / 7.9 s / no
videos over cap). The bottleneck reads like model judgment, not missing context: giving it more
information did not help it decide, and in three tries it made the decision slightly worse.

### Letting the local model reason: the biggest single lever on the walk (2026-09-22, `local_edges.py --think`)

Simon asked directly whether anything obvious was being missed. Two things were. The walk reused
`confirm_check.ask_bool`, which hard-codes `think: False` -- a setting chosen for a cheap yes/no
(does this region contain a read?) and then inherited, unexamined, by a far more nuanced call (has
the show resumed?). The extension's own local reader already runs qwen with `think: true`. And no run
had ever been READ: every verdict came from aggregate ad time and show lost, never from a single
answer the model actually gave.

Reasoning on, nothing else changed, same settings (patience 1, group 3, reach 6), same 93 regions:

| local walk | ad time | show lost | over 60 s | reads fully covered |
|---|---|---|---|---|
| reasoning off | 66.2% | 10.1 s | 1.6% | -- |
| **reasoning on** | **68.2%** | **9.9 s** | 1.6% | 18% |
| our averaged edge heads | 68.4% | 7.9 s | 0.0% | 38% |

**+2 points of ad time and slightly less show lost, from one boolean.** It is the largest single
improvement any change made to the local walk, and it closes almost the whole ad-time gap to our own
edge heads (68.2% against 68.4%). **Still not adopted**: it loses 2 s more show per video and breaks
the 60 s cap on one video where the heads break it on none, and it fully covers 18% of reads against
the heads' 38% -- it is landing near the right answer more often without landing ON it.

The trace (320 questions, each with the model's reasoning, in
`data/local_edges_trace_holdout4_walk_think_B10.jsonl`) also corrected a wrong diagnosis. Hunting for
dumb mistakes turned up none: the cases that looked wrong by keyword were sponsors deliberately
bridging into the pitch through the video's own subject -- a moon-infrastructure documentary sponsor
on a post-scarcity video, a Unity course sponsor reached via a speedrun's game engine. The model's
calls there were defensible. The remaining overreach is small and sits in genuinely ambiguous seams,
not in confusion.

**The lesson worth keeping is about the method, not the model.** "The bottleneck is model judgment"
was concluded from aggregate numbers alone, with a reasoning flag left off by inheritance and not one
transcript read. Two points of ad time were sitting behind that assumption.

### Reasoning on the line-number question too (2026-09-22 evening, `local_edges.py --same-prompt --think`)

Simon's question: what about reasoning WITHOUT the walk? The line-number path (`qwen_edges.ask_edges`)
had only ever run with `think: False`. Same 93 regions, Claude's prompt, reasoning on, a trace of every
answer in `data/local_edges_trace_holdout4_sameprompt_think_B10.jsonl`. 44 min on the 3080, 4 workers.

The script's own table falls back to OUR EDGE HEADS whenever qwen's answer is missing or outside the
trust window, which happened on 29 of 93 regions (8 calls timed out, 21 answers out of window). So the
first row mixes qwen with the heads. The second pair falls back to the raw region instead, which
isolates what qwen placed itself:

| qwen3:8b, line-number question, holdout 4 (dev) | ad time | show lost | over 60 s | reads fully covered |
|---|---|---|---|---|
| reasoning on, heads fill the gaps | 66.9% | 3.3 s | 0.0% | 32% |
| reasoning off, heads fill the gaps | 62.6% | 3.3 s | 0.0% | 26% |
| **reasoning on, qwen's own answers only** (64 taken) | **63.1%** | **2.9 s** | 0.0% | 26% |
| reasoning off, qwen's own answers only (60 taken) | 57.5% | 2.7 s | 0.0% | 19% |
| average of qwen (reasoning on) and our heads | 69.2% | 7.8 s | 4.7% | 40% |
| our averaged edge heads | 68.4% | 7.9 s | 0.0% | 38% |

**Reasoning is worth +5.6 points of ad time on qwen's own placements** at 0.2 s more show lost -- a
bigger gain than on the walk. It still does not reach our own heads (63.1% against 68.4%): qwen with
reasoning is a precise but timid placer, cutting less show than anything else measured and leaving
more of each read playing. The average with the heads edges past them on both axes by noise-sized
margins and breaks the 60 s cap on 3 videos, so it is not a win either. **Not adopted.** The local
model is still not an edge placer that beats the free tier; it is a lower-show-lost trade.

### The measured system was never the shipped system (2026-09-22, `export_candidate.py`, `serve_candidate.py`)

Two days of gains lived only in evaluation code. `candidate.py` refits the six detectors on all
pooled rows, scores a fresh feature file and prints a table; `finetune_minilm.py` trained the BGE
detector, wrote its per-line scores and **threw the weights away**. So every result could be
re-measured and none of it could read a new video. `predict.py serve`, which is what the extension
calls, still loaded `free_tier.pt`: the linear marker, one context model and two edge heads, the
system from before the stack and the fine-tune.

On holdout 4, the same 64 fresh videos and the same grader:

| system | ad time skipped | show lost per video | worst single video |
|---|---|---|---|
| cue patterns, no model | 30.1% | 13.0 s | 65 s |
| **`free_tier.pt`, what the extension serves** | **40.3%** | **10.8 s** | **177 s** |
| `free_tier_pooled.pt` | 39.3% | 5.9 s | 63 s |
| candidate v2 B = 10 + chapters | 67.8% | 8.4 s | none over 60 s |
| **candidate v3 B = 10 + chapters** | **67.2%** | **5.5 s** | **none over 60 s** |

27 points of ad time and half the show loss, already paid for and never delivered. No experiment in
the last day moved the number by more than 2 points, so this is worth more than the whole remaining
search.

The export path: `finetune_minilm.py --train-only --save-model` writes the checkpoint,
`marker/data/export_chain.sh` runs it for the three fine-tuned models (inside, start, resume),
`export_candidate.py` trains the five CPU parts and the stack on all pooled rows and saves one
bundle, and `serve_candidate.py` loads it. `export_candidate.py --verify holdout4` scores holdout 4
**through the saved bundle** and checks it against the numbers above: a bundle that does not
reproduce the grade is not the system that was graded.

Serving cost beyond the old free tier: the potion embedding (8M) and fine-tuned BGE-small (33M, one
17-line window per line), both CPU. v3 adds the community T5 on top, about a minute per 35-minute
video.

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

### What v3 misses, and targeting it (2026-09-23 night, `miss_audit.py`, `target_misses.py`, `qwen_veto.py`)

**Miss audit** (v3's out-of-fold scores on the 160 pooled videos, B = 10, reproduces 64.6% / 7.1 s): of
226 reads, ad time skipped is 76.5% for English reads with a cue word, **52.4% in machine-translated
captions** (52 of 160 videos), **~37.5% for reads with no cue word** (the same in both languages, so a
separate failure), **29.2% for English self-promo**. 59.7% of missed ad time is reads never touched,
40.3% trimmed edges. No "the whole video is the ad" case exists in this data. Full table: `data/miss_audit.md`.

**Three new inputs to v3's context model, 3 seeds each, all worse or inside noise** (B = 10 against v3's
64.6 / 66.3 / 67.0%): a translated-captions flag 63.7 / 63.9 / 64.3%; Simon's island signal as a
DETECTOR (window unlike both sides; `island_check.py` had only tried it as a filter) 65.4 / 66.6 / 67.0%
with +1-2 s lost; qwen's recorded answers as an input 61.9 / 55.1 / 65.9%. With 160 videos the context
model overfits every added input; new columns are not the lever.

**Level 2 as a veto: consistent gain at B = 10.** v3 at a looser threshold, and a find is kept if its
peak clears v3's own threshold or qwen's RECORDED answer (reasoning off) said "ad" on a window over it:
**68.4 / 68.8 / 68.9%** vs 64.6 / 66.3 / 67.0% (+3.8 / +2.5 / +1.9, all three seeds), show lost
9.5 / 9.7 / 9.3 s (inside B). Requiring qwen on every find: 67.5 / 68.1 / 68.1%. At B = 5 mixed
(-2.4 to +1.8). A lower bound for the live design (reasoning on); in serving qwen only has to read the
extra finds, not every window. Not yet built; needs Simon's go.

**Reasoning on does not help the veto** (2026-09-24 early, `qwen_sweep.py --think --only`, 1,822 windows
near loose v3 finds, 288 min GPU, 0 failures; `qwen_veto.py --think`). B = 10, veto the extras:
66.7 / 66.4 / 69.1% against reasoning off 68.4 / 68.8 / 68.9% and v3 alone 64.6 / 66.3 / 67.0%; veto
all 65.8 / 62.7 / 66.6%. Reasoning makes qwen say "ad" less often, and the veto needs its recall. So
level 2 = loose v3 + qwen with reasoning OFF (cheaper too: 3.0 s a window against 9.5 s, per the two sweep logs).

**Self-promo misses, read by hand** (2026-09-24, the 25 English self-promo reads in the 160, from
`data/miss_audit.csv`). 18 are never touched. Of those, about 6 are doubtful labels: "[music] >> [music]",
a plain "smash that like button", a story about selling apparel, a bingo-sheet bit, and a "Thanks again
to DeleteMe for sponsoring" that is a sponsor mention filed as self-promo. About 6 are end-of-video
outros (last 5%: subscribe, Patreon, merch, "extended thoughts on..."), worth little to a viewer because
the video is ending. About 6 are genuine mid-video pitches for the creator's own product (an app launch,
the host's own finance site twice, live-show tickets, a wallpaper app, a friend's podcast). So the
29.2% overstates the problem: the real target is ~6 own-product pitches, too few to train on alone.
Metric question for Simon: count end-of-video outros as ad time or not.

**English sponsor reads with no cue word, read by hand** (2026-09-24, 18 missed or mostly missed in the
160). Integrated product use with no "sponsored by": 7 (microwave bowl covers, a microdermabrasion
tool, car perfume twice, pepper spray, Elgato's prompter, World of Tanks). Cold opens in the first 15%
of the video: 7 (overlaps the above). Giveaway announcements: 3. Caption failures ("VPN.", "foreign
foreign"): 2. One story-style lead-in. No brand-name signal has ever been tried (checked README and
scripts). Next idea: a sponsor-brand list mined from the text of training reads, used like the qwen veto
(keep a loose extra find only if it names a known sponsor brand) rather than as another input, since
inputs overfit.

**Brand-list veto: small, free, positive on all three seeds** (2026-09-24, `qwen_veto.py --brand`). Words
that mark sponsor reads in at least 2 OTHER channels, with at least 60% of their lines (outside the
video's own channel) inside a read, mined from the reads' text leave-one-channel-out: they flag 1.2% of
lines and 9.2% of read lines. Keeping a loose extra find only if it names one, at B = 10: **66.9 / 67.3 /
67.7%** against v3 alone 64.6 / 66.3 / 67.0% (+2.3 / +1.0 / +0.7), show lost 7.9 / 8.4 / 8.8 s. It needs no
GPU, so it could improve level 1 itself. Combined with qwen it adds nothing (68.4 / 68.9 / 68.9%): qwen
already confirms those finds. Not built into serving; small enough to be partly selection noise, so it
should be confirmed on holdout 5 before shipping.

**A separate self-promo model: it barely learns** (2026-09-24, Simon's idea, `qwen_veto.py --selfpromo`).
Logistic regression on the same 395 line features, self-promo lines only (479 lines, 32 reads in the
160), channel-grouped 5-fold: out-of-fold average precision **0.011 against 0.005 by chance**. Used as a
veto on v3's loose extras (its top 2% of lines): 66.3 / 67.4% at B = 10 (seeds 0-1) against v3's 64.6 /
66.3%, below qwen's 68.4 / 68.8%, and nothing added on top of qwen. With a model that weak, the gain is
most likely the loose threshold itself letting a few more real finds through, not self-promo detection.
32 reads (a third doubtful) are too few; more self-promo examples would have to come first (the
channel set has 18 more, but it is reserved as a test set).

### Claude's category labels: v3 skips sponsor reads, not the rest (2026-09-24, `claude_label.py --categories`, `category_audit.py`)

Simon: split the data by kind of promotion, because the creator's own product "could go either way
depending on the style of read", and SponsorBlock's selfpromo lumps "subscribe / support us" in with real
pitches. Claude Haiku labelled every window of all 205 pooled videos (164 min, 0 failed windows) into
sponsor / own_product / channel_plug / other_promo, each a "read" (the show stops for a pitch) or a
"mention". Totals: sponsor read 249 segments (13,538 s), channel_plug read 104 (1,702 s) and mention 93,
own_product read 48 (1,602 s) and mention 30, other_promo mention 35 and read 24, sponsor mention 13.
In a 10-video pilot, 4 of SponsorBlock's 5 selfpromo marks were channel plugs.

v3 (seed 0, B = 10, out-of-fold) against these labels on the 160: **sponsor reads 75.5% of lines skipped**
(28.7% of segments never touched); **own_product reads 20.8%** (73.2% never touched); other_promo reads
32.8%; channel_plug reads 11.5% (88.0% never touched); every "mention" under 16%. The own-product read is
the gap Simon pointed at: long enough to matter (1,602 s over 48 reads, ~33 s each) and almost unseen.
Simon (09-24): end-of-video plugs count as ad time, so channel_plug reads are in scope too.

### Detectors trained on Claude's categories (2026-09-24, `category_detector.py`)

Linear models on the 395 frozen line features, channel-grouped folds over all 205 videos. Out-of-fold
average precision: own-product + other-promo reads **0.037** (chance 0.010; 45 videos), all non-sponsor
promo reads **0.184** (chance 0.016; 116 videos), any promotion 0.621 (chance 0.074). SponsorBlock's
selfpromo labels gave 0.011 against 0.005: Claude's separated labels are far easier to learn.
As a veto on v3's loose extras (B = 10, 3 seeds, SponsorBlock labels): +0.0 to +1.5 points, below qwen's
+2-4; against Claude's any-promo labels, nothing. As their own region source added to v3: no threshold
meets the B = 10 rule (too imprecise alone). Structural reason: a veto can only keep what v3 found
loosely, and v3 never touches 73% of own-product reads. Next: the same target fine-tuned (BGE-small).

**Fine-tuned on Claude's non-sponsor promo reads** (BGE-small, `finetune_minilm.py --target claude-nonsponsor`,
22 min): out-of-fold average precision **0.380** (chance 0.016), against 0.184 linear and 0.011 for
SponsorBlock's selfpromo. As its own region source added to v3: fails the B = 10 rule on every seed
(v3 alone already sits at 1.9% of videos over 60 s against the 2% cap; the strictest setting adds +1.1
points for +4.9 s). As a veto on v3's loose extras (`category_detector.py --ft`), B = 10: SponsorBlock
labels 65.7 / 67.6 / 68.3% vs 64.6 / 66.3 / 67.0% (+1.1 to +1.3, +1.3 to +2.2 s); Claude any-promo labels
57.4 / 60.0 / 60.6% vs 57.5 / 60.9 / 57.5% (inconsistent). A veto cannot reach reads v3 never touches,
which is where own-product reads are. Proposed to Simon: a separate opt-in self-promo path graded on its own.

**The fine-tuned Claude-category detector as an eighth stack input** (Simon's question, 2026-09-24,
`target_misses.py --variants claudeft [--claude-target]`). A, stack learns SponsorBlock labels: B = 10
63.0 / 61.2 / 55.7% against v3's 64.6 / 66.3 / 67.0%, worse on every seed. B, stack learns and is graded on
Claude's any-promo labels, against v3's inputs retrained on the same target: B = 5 47.4 / 47.0 / 46.1% vs
40.9 / 41.7 / 43.7% (+2.4 to +6.5), B = 10 52.1 / 48.0 / 59.1% vs 57.7 / 61.5 / 59.3% (worse). Inconsistent;
not adopted. Retargeting alone (v3 inputs on Claude's labels, 57.7-61.5%) matches v3 as built on those
labels (57.5-60.9%). Every way of attaching the detector to v3 (veto, union, input) has now failed.

### Enlarging the stack's training pool to 494 videos: sharper edges, no more ad time (2026-09-24, `enlarge_pool.py`, `enlarge_claude.py`, `enlarge_stack.py`)

Every set already graded or used as a tripwire joins the pool: 494 videos, 326 channels, 619 reads, 434
stack-eligible (160 original + 274 new). All streams re-run out-of-fold over the enlarged pool; nothing
here touches a live holdout (holdout 6 is the fresh test). Seeds 0 / 1 / 2 throughout.

**Fine-tuned BGE detectors, old (205 videos) vs enlarged (494), average precision on the same 205 videos:**
inside a read 0.730 -> 0.737 (unchanged); start line 0.111 / 0.130 / 0.131 -> **0.164** (0.160 on the 289
added videos); resume line 0.192 / 0.187 / 0.194 -> **0.230** (0.206 on the added). More data sharpens the
edge heads, not the detector.

**Comparison 1, stacking step trained on 434 vs the original 160**, same (enlarged) detector streams, graded on
the 274 new videos. B = 5: 48.4 / 48.6 / 50.3% vs 47.8 / 52.7 / 52.4%. B = 10: 71.3 / 72.4 / 70.9% at
8.3 / 8.5 / 8.2 s vs 71.3 / 73.4 / 71.9% at 9.0 / 9.9 / 9.4 s. The bigger pool buys about 1.1 s less show
lost per video for about 0.7 points less ad time: the 160-video stack was NOT badly overfit.

**Comparison 2, fine-tuned Claude-category detector as an 8th input**, all 434 videos, B = 10: 69.9 / 69.8 /
70.3% vs v3's 69.5 / 70.7 / 71.1%; B = 5 57.6 / 57.8 / 59.1% vs 58.6 / 59.3 / 59.2%. Not adopted (still fails
on the bigger pool). qwen's recorded answers skipped: no sweep exists for the 274 new videos.

**Comparison 3, the stacking step's shape** (all 434, B = 5 / B = 10 ad time, mean of 3 seeds, show lost at B = 10):

| stacking step | B = 5 | B = 10 | show lost |
|---|---|---|---|
| hidden 32, wd 0.01, dropout 0.2 (recipe of record) | 59.0% | 70.4% | 8.0 s |
| linear (hidden 0) | 53.2% | 68.4% | 8.2 s |
| wd 0.1 | 59.8% | 70.0% | 7.9 s |
| wd 0.3 | 59.7% | 70.6% | 8.3 s |
| dropout 0.5 | **60.9%** | 70.7% | 8.2 s |

The hidden layer earns its place (linear loses 6 points at B = 5). Dropout 0.5 wins at B = 5 on every seed
(+1.9) and ties at B = 10: a candidate for holdout 6, not adopted on CV alone. Weight decay does nothing.

Verdict: no change to v3. Open for Simon: retrain the shipped stack on 434 videos (about 1 s less show cut,
ad time flat) and/or dropout 0.5, both to be graded once on holdout 6.

### What a flat 180 s cap would cost level 4 (2026-09-24, zero spend)
Level 4's runaway ends on holdout 5 (worst 575 s) suggested cutting every served segment at 180 s. Before
touching the test set, the cost of that rule was measured on the labelled reads it would clip: 558
SponsorBlock reads in the burned sets (pooled, channels, holdouts 2-4, tail). Median read 66 s, p90 130 s,
p99 325 s, longest 444 s.

| cap | real reads longer | real ad time a truncation would drop |
|---|---|---|
| 150 s | 47 | 7.3% |
| **180 s** | **23** | **5.0%** |
| 240 s | 10 | 2.7% |
| 300 s | 7 | 1.3% |

So a flat 180 s cap is not free: it could take up to ~5 points of ad time back from level 4's 73.2%.
Smarter variant, not yet testable: cap only when the resume quote did not snap to a caption line (a
runaway end presumably has none). Testing either needs served-reader answers on a burned set (holdout 4,
64 videos, roughly $4 of plan usage at holdout 5's rate); served answers exist only for holdout 5.

### Shrinking the three fine-tuned models: fp16 is free, int8 is not (2026-09-24, `compress_parity.py`)
The bundle's bge / edge_start / edge_resume checkpoints weigh 401 MB together (fp32). Each compression
was run through the serving code (CPU, gated, v3 at B = 10) on holdout 4 (64 videos, already graded)
and compared with the fp32 originals on the same rows.

| models | size | line-score change vs fp32 (max) | regions identical | ad time | show lost / video |
|---|---|---|---|---|---|
| fp32 (shipped) | 401 MB | - | - | 66.2% | 5.0 s |
| **fp16 weights, upcast at load** | **200 MB** | 0.008 | **51 of 51 videos** | 66.2% | 5.0 s |
| int8 dynamic (Linear layers) | 208 MB | 0.999 | 25 of 51 | 67.1% | 5.5 s |

**fp16 halves the download with no change in what gets skipped.** int8 saves nothing more (the
embedding table stays fp32) and changes half the videos' regions, so its +0.9 points is drift, not a
gain. Not applied to the bundle yet: that is decision 5.

### The self-promotion path: its own detector beside v3 (2026-09-24, `selfpromo_path.py`)
Simon wanted self-promo detection now. Every way of attaching the self-promo detector to v3 had failed
(above), so it runs beside v3 instead: the fine-tuned BGE-small trained on Claude's non-sponsor promotion
reads (own product, channel plugs, other promo) scores every line, a rule of its own marks regions, and
those are served as `selfpromo`, which the extension's "Skip self-promotion too" switch controls. Anything
overlapping a sponsor region is dropped (sponsor wins); v3's sponsor answer is unchanged (checked live on
4 holdout-4 videos).

The rule was picked on the true out-of-fold stream of the 251 videos Claude labelled (channel-grouped folds),
under a budget on real show lost (flagged seconds where Claude saw no promotion of any kind and SponsorBlock
has no sponsor). 84 minutes of such reads in 138 of the 251 videos:

| show-lost budget | rule (threshold, smoothing, bridge, min s) | self-promo read time caught | show lost / video | over 60 s | worst |
|---|---|---|---|---|---|
| 1 s | 0.80, 5, 2, 10 | 19.0% | 0.9 s | 0% | 57 s |
| **2 s (shipped)** | **0.95, 1, 0, 5** | **28.6%** | **1.9 s** | **0%** | 58 s |
| 3 s | 0.55, 1, 0, 5 | 34.8% | 2.9 s | 0.8% | 68 s |
| 5 s | 0.40, 1, 4, 5 | 40.7% | 4.9 s | 0.8% | 121 s |

For scale, v3 alone skips 11-21% of these reads. Served model: the same recipe trained on all 251 videos
(`finetune_minilm.py --target claude-nonsponsor --extra-sets features_holdout5.npz --train-only`), stored fp16
as `production/models/selfpromo.pt` (67 MB) with `selfpromo_rule.json`; without those files the path is off.
Cost: it reads every line (v3's gate would hide exactly these reads), about 30 s of CPU on a 36-minute video
at the program's 2 threads. First live finds: an end-of-video "like and subscribe" (in scope, per Simon) and
a 5 s false positive on a Patreon mention inside the topic. Its fresh grade is holdout 6, once; not yet in the
models release.
