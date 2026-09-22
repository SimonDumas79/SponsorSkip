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
