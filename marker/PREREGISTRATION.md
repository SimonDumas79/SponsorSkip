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
