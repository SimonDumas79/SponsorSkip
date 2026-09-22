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
