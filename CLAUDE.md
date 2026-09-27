# SponsorSkip

`README.md` is the full reference; `marker/README.md` has the marker pipeline's run order and the system of models.

## How Simon and Claude work on marker/

**Claude drives, Simon conducts** (his call, 2026-09-21). Claude writes the code; Simon decides what gets built and in what order. Explain what each change does and what the numbers mean in plain words, and check direction with him before a step that changes the plan.

- Run commands from the repo root: scripts open `marker/data/...` by relative paths.
- `features.npz` holds `X` (rows x 397: 384 MiniLM meaning, 3 seam, 6 cues, 2 position, 2 description columns), `y`, `video`, `line`, `split`, `is_start`, `is_resume`, `category` ("sponsor", "selfpromo" or ""), `start_seconds`, `channel`, `feature_names`. Labels cover SponsorBlock `sponsor` AND `selfpromo`.
- Eight channel-disjoint sets: `features.npz` (train/val, tuned by channel-grouped CV); the holdouts `features_tail.npz` (holdout 1), `features_holdout2.npz` to `features_holdout5.npz`; `features_channels.npz` (the three named channels); `features_negatives.npz` (sponsor-free videos). Holdouts 1 to 5 are USED (each graded once); holdout 6 is pre-registered and waits on the crawl.

## Measuring honestly

- **Grade on ad time skipped, not just region recall.** Recall counts a read as found if one line of it is skipped. `replay.py`'s `ad cov` is the share of ad seconds actually skipped; report it with the real show lost per video and the worst single video.
- Tune on cross-validation; grade each holdout once; never choose among systems by their holdout numbers.
- Every stage after the marker trains on OUT-OF-FOLD scores with the marker's own channel folds.
- Language-model answers (qwen via Ollama) are recorded once (`data/confirm_verdicts*.jsonl`, `*.edges.jsonl`) and replayed; only rules that select over recorded answers can be scored that way.
- Never leave qwen in VRAM: the scripts unload it when they finish.

## Where things stand

Holdouts are graded once each, and the results live in `marker/PREREGISTRATION.md` (results 1 to 4) with the full record in `marker/README.md`. Latest: holdout 5 (2026-09-24), level 1 (v3) 62.7% of ad time at 4.7 s of show lost per video. Holdout 6 is pre-registered (pre-registration 5) and not built yet.
