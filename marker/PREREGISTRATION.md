# Pre-registration: the one look at the channel set (written 2026-09-22 ~05:50, before any grading)

The 169-video channel set (Sabine Hossenfelder, Wes Roth, How Money Works; 191 reads) has never been
scored by anything trained here. Five Collective rooms agreed its effective sample size is 3 channels,
so it is a **tripwire for gross overfitting**, not proof that a fix generalises. It is graded ONCE,
with everything below fixed in advance. A many-channel fresh test (the SponsorBlock tail crawl) follows.

## The candidate (fixed)

- Level 1, each trained on all 205 pooled videos: the frozen-MiniLM marker, meaning-only and
  structure-only linear models, the potion-base-8M marker, the 1-D conv sequence model (hidden 32,
  reach 7, 12 epochs), and **BAAI/bge-small-en-v1.5 fine-tuned** (line + 8 lines either side,
  2 epochs, lr 3e-5, seed 0).
- Level 2: one context model (MLP, hidden 32, seed 0) over the six detectors' context features plus
  the cue and description columns, trained on the pooled out-of-fold level-1 scores.
- Edges: the MLP start/resume heads, trained on pooled rows.
- Thresholds, fixed by the pooled CV rule on this system's out-of-fold scores:
  **B = 5: 0.9951, B = 10: 0.9870.**
- Variant: the same plus creator chapters titled for a read (adopted on pooled CV, +2 points).
- Code: `marker/candidate.py --fresh features_channels.npz --bge data/finetune_full_bge_seed0__channels.npy`.

## Baselines (fixed)

- The graded free tier `data/production/free_tier.pt` (what 0.7.0 ships as "Free marker on this PC").
- The pooled free tier `data/production/free_tier_pooled.pt`.
- Both via `predict.py grade --features features_channels.npz`.

## What is reported

Ad time skipped, show lost per video, share of videos over 60 s, false-alarm regions, reads
fully / partly / not skipped, at B = 5 and B = 10; English videos separately (all three channels are
English, so the same numbers).

## What counts as passing

- **Pass:** the candidate skips more ad time than the graded free tier at both budgets, with show lost
  per video within 3 s of its pooled-CV figure and no more than 2% of videos over 60 s.
- **Gross-overfitting tripwire:** candidate ad time at B = 10 below 45% (pooled CV was 57.0-58.7%
  over three BGE seeds) means something does not transfer, and nothing ships until it is understood.
- Whatever the result, it is reported as measured, with the 3-channel caveat attached.

## Result (graded once, 2026-09-22 ~06:55; `data/prereg_channels.log`)

169 videos, 191 reads, 3 channels, none in the pooled training data.

| system | B | ad time | show lost / video | videos over 60 s | false regions |
|---|---|---|---|---|---|
| **candidate** | 5 | **54.2%** | 5.2 s | 0.6% | 5 |
| **candidate** | 10 | **64.0%** | 6.3 s | 1.8% | 6 |
| candidate + creator chapters | 5 | 87.8% | 6.6 s | 1.8% | 5 |
| candidate + creator chapters | 10 | 91.5% | 7.6 s | 3.0% (breaks the 2% cap) | 6 |
| graded free tier (`free_tier.pt`, shipped) | | 43.7% | 8.0 s | | |
| pooled free tier (`free_tier_pooled.pt`) | | 42.6% | 5.4 s | | |
| cue patterns | | 37.0% | 22.8 s | | |

Reads fully / partly / not skipped by the candidate: 42 / 34 / 24% (B = 5), 46 / 37 / 17% (B = 10).

**Verdict against the rule written above: PASS.** The candidate beats the graded free tier at both
budgets (+10.5 and +20.3 points), show lost is within 3 s of its pooled-CV figure (CV ~5.0 s at B = 5,
~8.5 s at B = 10), and no more than 2% of videos lose over 60 s. The tripwire (under 45% at B = 10)
did not fire.

Caveats, as registered: 3 channels, so this rules out gross overfitting and does not show the size
generalises. The chapter result is channel-specific: all three creators title their sponsor
segments, so chapters nearly solve these channels; most channels do not (20 of 280 pooled reads had
such a chapter). The next test is the many-channel SponsorBlock tail crawl.

---

# Pre-registration 2: the many-channel test (written 2026-09-22 ~07:05, before the videos are crawled)

**Set ("holdout 4").** Every video the resumed SponsorBlock-tail crawl fetches (`fetch_captions.py
--limit 180`, from `candidates.json`) whose channel is in NONE of `examples.jsonl`,
`examples_tail.jsonl`, `examples_holdout2.jsonl`, `examples_holdout3.jsonl`, `examples_channels.jsonl`.
Built with `build_dataset.py --captions marker/data/captions --all-to holdout4 --keep-selfpromo-only
--not-in` all five. Expected: ~100+ videos from as many channels.

**System.** Exactly the candidate above: same six detectors trained on all 205 pooled videos, the
fine-tuned BGE-small seed 0 (a fresh `--score` run on this set), the same context model and edge
heads, **the same thresholds: B = 5 0.9951, B = 10 0.9870**. Two variants reported: plus creator
chapters (fetched with `fetch_watch_meta.py`), and English videos only.

**Baselines.** `free_tier.pt` (shipped) and `free_tier_pooled.pt`, via `predict.py grade`; the
community SponsorBlock model (`sponsorblock_ml.py`) at its default 0.5 classifier cut, if it has
been run on these videos (it was trained on SponsorBlock data up to early 2022, so any of these
videos older than that may be in its training set: reported, not hidden).

**Pass.** The candidate skips more ad time than `free_tier.pt` at both budgets, show lost per video
within 3 s of its pooled-CV figure, at most 2% of videos over 60 s. Graded once.

**Added before holdout 4 exists (2026-09-22 ~07:25): candidate v2, reported beside the candidate.**
Identical, except the edge heads are the average of the MLP start/resume heads and fine-tuned
BGE-small start/resume models (`finetune_minilm.py --target start|resume`, trained on all pooled
videos and scored on holdout 4). Its thresholds are its own pooled-CV rule choice on the
stack + BGE seed 0 scores with averaged heads, fixed now: computed by `candidate.py` from the
pooled out-of-fold scores, never from holdout 4. Pooled CV (three BGE stack seeds): 49.6-51.0%
at B = 5 and 61.8-62.4% at B = 10, against 48.1-52.1% and 57.0-58.7% with the MLP heads alone.
The original candidate stays the primary, pre-registered system.

**Added before holdout 4 is graded (2026-09-22 ~08:10): candidate v3, reported beside the others.**
Candidate v2 plus the community SponsorBlock model (xenova/sponsorblock-ml, run by
`sponsorblock_ml.py` on holdout 4) as a seventh detector. Its context model is trained only on the
160 pooled videos whose SponsorBlock labels postdate the community model's training (`sb_dates.json`),
and its thresholds come from channel-grouped CV over those 160 (`stack_sbml.py` seed 0): pooled CV
there 60.7% (B = 5) and 64.6% (B = 10), against 50.1% / 53.1% without it. Holdout-4 videos whose
labels predate 2022-04 are reported separately (the community model may have trained on them).
v3 needs the GPU (the community model is ~11 s per video); the primary system stays the candidate.

**Amendment before holdout 4 is built or graded (2026-09-22 ~08:35).** YouTube's throttling makes the
resumed tail crawl slow (16 attempts in an hour), and 64 caption files already on disk come from
channels in none of the five sets: most were fetched by the same tail crawl on 2026-09-21 after holdout
3 was built, and have never been used for training, tuning or grading. Holdout 4 is therefore every
caption file in `marker/data/captions` whose channel is in none of the five sets, as built by
`build_holdout4.sh` now; later crawl videos form a holdout 5. Nothing else changes: same systems, same
fixed thresholds, graded once.

## Result 2: holdout 4 (graded once, 2026-09-22 ~09:35; `data/prereg_holdout4.log`)

64 videos from 64 channels, none in any other set; 72 reads; 22 English, 42 on machine-translated
English tracks; 8 with SponsorBlock labels older than 2022-04.

| system | B | ad time | show lost / video | over 60 s | English only |
|---|---|---|---|---|---|
| candidate (primary) | 5 | 53.3% | 9.0 s | 1.6% | 61.6% @ 6.8 s |
| candidate (primary) | 10 | 64.6% | 11.2 s | 1.6% | 70.5% @ 6.9 s |
| candidate v2 (averaged edge heads) | 5 | 46.5% | 4.2 s | 0.0% | 60.7% @ 4.8 s |
| candidate v2 | 10 | 67.6% | 8.6 s | 0.0% | 72.9% @ 6.8 s |
| candidate v3 (+ community model) | 5 | 52.3% | 2.4 s | 0.0% | 68.2% @ 0.7 s |
| candidate v3 | 10 | 66.4% | 5.0 s | 0.0% | 71.1% @ 0.7 s |
| graded free tier (shipped) | | 40.3% | 10.8 s | | |
| pooled free tier | | 39.3% | 5.9 s | | |
| cue patterns | | 30.1% | 13.0 s | | |
| community SponsorBlock model (cut 0.5) | | 77.0% | 9.9 s | 3.1% | |
| same, on the 56 videos labelled after its training | | 75.4% | 8.0 s | 3.6% | |

Creator chapters add +0.7 to +1.4 points (these channels rarely title sponsor chapters).

**Verdicts against the rule written in advance:**
- **Candidate (primary): does not pass as registered.** It beats the shipped tier at both budgets
  (+13.0 and +24.3 points) with at most 1.6% of videos over 60 s, but at B = 5 it lost 9.0 s of show
  per video against ~5.0 s on pooled CV, outside the 3 s tolerance (B = 10: 11.2 vs ~8.5-9.0 s, inside).
  The excess sits on the machine-translated tracks: English videos lost 6.8 s.
- **Candidate v2: passes** (beats shipped at both budgets; 4.2 / 8.6 s lost, at or under CV; 0% over 60 s).
- **Candidate v3: passes**, with the best cost: 52.3% at 2.4 s and 66.4% at 5.0 s, 0% over 60 s.
- **The community SponsorBlock model skips the most ad** (75.4% on videos it cannot have trained on) at
  8.0 s lost per video, but breaks the 2% cap (3.6% of videos over 60 s). At a similar cost (8-11 s)
  it beats our v1 and v2 by about 8-10 points on these channels; v3, which includes it, trades some of
  that for half the lost show and no video over 60 s. Its thresholds were fixed on pooled CV and were
  conservative here (5.0 s used of a 10 s budget); they are not re-chosen on this set.

**Added after the graded look (2026-09-22 14:15): the readers on the same 64 videos.** Not
pre-registered, and nothing here is tuned on this set (a reader answers directly; there is no
threshold to choose), so it is reported as a measurement, not as a pass/fail:

| reader | ad time | show lost / video | over 60 s |
|---|---|---|---|
| **Claude Sonnet** (`claude_label.py --model sonnet`, 0 failed windows) | **81.9%** | **5.3 s** | 1.6% |
| community SponsorBlock model (cut 0.5) | 77.0% | 9.9 s | 3.1% |
| candidate v2 (free tier) | 67.6% | 8.6 s | 0.0% |
| candidate v3 (free tier + community model) | 66.4% | 5.0 s | 0.0% |
| graded free tier (shipped) | 40.3% | 10.8 s | |

Sonnet found a read in 57 of 64 videos and, on the 22 English ones, took 85.5% of ad time at 6.9 s.
It is the best reader measured, above the community model on both axes and inside the 2% cap the
community model breaks.

## Pre-registration 3: the cascade (written 2026-09-22 ~15:30, NOT yet graded)

**What is registered.** The marker's six detectors find the regions (the cheap five gating the
fine-tuned three at a 30% share, as `export_candidate.py` stores it), and claude-haiku is asked, once
per region, only where that region starts and ends. An answer is used only if it lands within 20
lines before the region's start / 10 lines into it, and symmetrically at the end; anything else, and
any failed call, keeps the marker's own averaged edge heads. Creator chapters apply as usual.

**Thresholds are being fixed now, on the pooled 205 videos**, by the written rule (most ad time
subject to the show budget, at most 2% of videos over 60 s), over out-of-fold stack scores with
Claude's edges in place (`cascade_pooled.py`, shares 0.012 / 0.02 / 0.03). They are written to
`data/cascade_thresholds.json` before any fresh set is touched.

**Why a new set is needed, and why it does not exist yet.** The 71.2% at 4.6 s that made this look
worth shipping was measured on holdout 4, and the idea was then developed against that same set (the
B = 5 run, the local-model comparison). Holdout 4 is therefore a DEVELOPMENT set for the cascade, not
a grade, and its number must not be quoted as one. Holdout 3 cannot stand in: it is 10 videos and 15
reads, a confidence interval of roughly ±25 points, which cannot decide anything.

So the cascade's grade waits on new videos. YouTube refused this address today at 45 s and at 180 s
between videos, after a 10-minute cool-off, so the crawl is rescheduled with a 90-minute cool-off and
4 minutes between videos.

**Registered in advance**: holdout 5 is every caption file whose channel appears in none of
`features.npz`, `features_tail.npz`, `features_holdout2.npz`, `features_holdout3.npz`,
`features_channels.npz`, `features_holdout4.npz`, built once the crawl has 40 or more such videos.
Graded once, with the thresholds above, against: the shipped free tier, the candidate with our own
edge heads, and the cascade. The comparison that decides whether the cascade ships is the cascade
against the candidate **on that set**, not on holdout 4.

## Pre-registration 4: everything that could be packaged (written 2026-09-24, before holdout 5 is built)

Simon: "Run the test and present all of the data to me and I will make decisions on what to package."
Holdout 5 is built exactly as pre-registration 3 defines it (every caption file whose channel is in
none of the six earlier sets; 45 such videos exist now, over the 40 minimum). The crawl keeps running;
videos it fetches after the build belong to a later holdout, not this one.

**Graded once, all on the same holdout-5 videos, with SponsorBlock's sponsor + selfpromo labels:**

| | system | thresholds (fixed before the set exists) |
|---|---|---|
| A | cue patterns, no model | none |
| B | `free_tier.pt` (served until 0.8.0) | its own |
| C | candidate v2 (six detectors), + chapters | `candidate.pt` thresholds_v2, B = 5 and 10 |
| D | candidate v3 (+ community model), + chapters: **level 1 in 0.8.0** | `candidate.pt` v3 thresholds |
| E | v3, loose, extras kept only if they name a brand word (`brand_words.json`, mined from all 205 pooled reads) | `veto_thresholds.json` "brand" |
| F | v3, loose, extras kept only if qwen3:8b (reasoning off) says "ad" on a window over them: **level 2 as measured** | `veto_thresholds.json` "qwen" |
| G | v3 + claude-haiku placing each region's edges: **level 3 in 0.8.0** | v3's |
| H | the cascade of pre-registration 3 (v2 regions + haiku edges) | `cascade_thresholds.json` |
| I | claude-haiku reading the whole transcript through the served code (`level4_eval.mjs`): **level 4, the default** | none |

**Reported for each:** ad time skipped, real show lost per video, worst single video, share of videos
over 60 s, reads touched, and seconds per video on this PC. Model size on disk per level. Secondary:
Claude category labels (`claude_label.py --categories`) on holdout 5, and each system's share of each
category's lines skipped, with end-of-video plugs counted as ad time (Simon, 2026-09-24).

Nothing is chosen on holdout 5. The table goes to Simon as it comes out, including anything that
fails or crashes. If a system cannot be run as specified, that is reported, not patched around.

**Fixed values (seed-0 out-of-fold picks, `qwen_veto.py --brand --save`, before the build):** E and F
at B = 10: loose 0.98964, strict 0.99593 (= v3's served threshold); at B = 5: loose 0.99838, strict
0.99837 (the rule chose almost no loosening). `brand_words.json`: 154 words. Level 1-3 grades are
reported at B = 10 as served, with B = 5 alongside.

## Result 4: holdout 5 (graded once, 2026-09-24; `data/grade_holdout5.log`, `cascade_holdout5.log`, `level4_holdout5.jsonl`)

46 videos, 61 reads, 46 channels. B = 10 unless marked. Ad time and show lost against SponsorBlock labels.

| | system | ad time | show lost / video | worst video | over 60 s | reads missed |
|---|---|---|---|---|---|---|
| A | cue patterns | 32.3% | 18.3 s | 78 s | | |
| B | free_tier.pt | 43.4% | 9.5 s | 104 s | | |
| C | v2 + chapters | 53.5% | 5.8 s | 49 s | 0% | 41% |
| D | **v3 + chapters (level 1)** | **62.7%** | **4.7 s** | 49 s | 0% | 41% |
| E | loose v3 + brand words | 63.4% | 6.0 s | 52 s | 0% | 38% |
| F | loose v3 + qwen (level 2) | 65.6% | 6.8 s | 52 s | 0% | 34% |
| G | v3 + haiku edges (level 3) | 61.9% | 7.5 s | 49 s | 0% | 44% |
| H | cascade (pre-reg 3) | 49.4% | 6.1 s | | 2.2% | 46% |
| I | Claude whole transcript (level 4, served) | 73.2% | 28.7 s | 575 s | 10.9% | 21% |

B = 5: C 39.1% / 3.9 s, D 54.1% / 2.7 s, E and F identical to D (the rule loosened nothing), G 52.6% / 3.8 s.
Level 3: haiku answered 62 of 62 regions, only 8 answers inside the acceptance window; the rest kept
our heads. Level 4: 3 of 46 reads failed (Claude killed at ~240 s); its worst losses are runaway ends
(0-637 s for a 65 s read), which the labeller's 180 s cap would have stopped; not patched on this set.
Pre-registration 3's decision: the cascade (H) does not beat the candidate (C) on this set; it does not ship.
CPU seconds per video (5 videos, 4-79 min): free 10-12, v2 18-62, v3 22-72.

## Pre-registration 5: holdout 6 (written 2026-09-26, before holdout 6 is built)

**Set.** Holdout 6 is every caption file whose channel appears in none of `examples.jsonl`,
`examples_tail.jsonl`, `examples_holdout2.jsonl` to `examples_holdout5.jsonl`, `examples_channels.jsonl`
or `examples_negatives.jsonl`. It is built once the crawl has 40 or more such videos and graded once,
with SponsorBlock's sponsor + selfpromo labels. End-of-video plugs count as ad time (Simon, 2026-09-24).

**Graded, all fixed before the set exists:**

| | system | fixed values |
|---|---|---|
| D | v3 + chapters (level 1, as served) | `candidate.pt` v3 thresholds |
| D2 | **challenger: v3 with dropout 0.5 in the stacking step**, otherwise identical (same 160 videos, same streams, same edge heads) | `data/dropout_thresholds.json`: B = 5 0.99945, B = 10 0.99611; model `production/v3_dropout.pt` |
| G | **challenger: v3 with growth instead of the edge heads** — same detection (model, thresholds) as D; once a region is found, it grows outward line by line while the v3 stack's own score stays above a fixed bar, replacing the start/resume heads entirely (`marker/grow_sweep.py`, added 2026-10-02 after the Fireship video showed heads collapsing a 65 s read to 5 s) | same v3 thresholds as D (B = 5 0.9984, B = 10 0.9959); grow bar B = 5 0.98, B = 10 0.95 — fixed on the same 160 CV videos, chosen as the lowest bar that still clears the budget with headroom, not the loosest bar the rule alone would pick |
| S | self-promo path beside v3 (`selfpromo.pt`, `selfpromo_rule.json`), graded on Claude's category labels for holdout 6 | the shipped rule |
| I | level 4 as served since 2026-09-26: whole transcript, with the unconfirmed-end cap (180 s, `segments.mjs`; a cut leaving 30 s or less of the video skips to the end) | none; the uncapped answer is reported beside it from the same calls |

**Decision rules (written now):**
- **D2 replaces D** only if, at B = 10 (the served budget), it skips more ad time **and** loses no more
  than 0.5 s more show per video **and** puts no more videos over 60 s. Otherwise v3 stays. B = 5 is reported, not decided on.
- **G replaces D** under the same rule as D2 (more ad time, no more than 0.5 s more show, no more
  videos over 60 s, at B = 10). If both D2 and G clear the rule, the one with more ad time ships;
  a tie keeps D, since it needs no retrain. G and D2 can both lose to D.
- **The cap stays** unless, on holdout 6, it removes more real ad time than the show it saves.
- **S** is reported against its CV number (28.6% of self-promo read time at 1.9 s). Nothing is decided on it here.

**What CV says about D2 before the grade (`dropout_challenger.py`, the 160 videos v3 is trained on, seeds 0-2).**
It does NOT reproduce the enlarged pool's result, where dropout 0.5 won at B = 5. Here it LOSES at B = 5
(52.6 / 54.1 / 51.0% vs v3's 60.7 / 55.5 / 59.3%, show 4.5-4.7 s vs 4.8-5.0 s) and gains a little at
B = 10 (67.4 / 67.7 / 68.1% vs 64.6 / 66.3 / 67.0%, show 7.5 / 8.3 / 8.8 s vs 7.1 / 8.3 / 7.4 s), mostly
by skipping more show. It is registered anyway at Simon's call; the rule above means a win bought only
with lost show does not ship.

**What CV says about G before the grade (`marker/grow_sweep.py`, the same 160 videos, v3's own
thresholds unchanged).** At B = 10: 67.3% of ad time at 6.3 s of show lost, over-60 1.2%, against D's
64.6% at 7.1 s, over-60 1.9% — more ad time AND less show lost AND fewer over-cap videos, the only
challenger so far to beat D on every axis on CV. At B = 5: 61.3% at 3.2 s, over-60 0.0%, against D's
60.7% at 4.9 s, over-60 1.2%. The missed-read share is unchanged (33%; growth cannot find a read the
detector never touched) but full reads rise from 35% to 37% at B = 10 and partial skips shrink, which
is the mechanism: the edge heads misplace the start on a partially-found read (as on the Fireship video,
where they kept only the last 5 s of a 65 s read), and growth from the detector's own score recovers it.
