# SponsorSkip

`README.md` is the full reference; `marker/README.md` has the marker pipeline's run order and the system of models.

## How Simon and Claude work on marker/

**Claude drives, Simon conducts** (his call, 2026-09-21). Claude writes the code; Simon decides what gets built and in what order. Explain what each change does and what the numbers mean in plain words, and check direction with him before a step that changes the plan.

- Run commands from the repo root: scripts open `marker/data/...` by relative paths.
- `features.npz` holds `X` (rows x 395), `y`, `video`, `line`, `split`, `is_start`, `is_resume`, `category` ("sponsor", "selfpromo" or ""), `start_seconds`, `channel`, `feature_names`. Labels cover SponsorBlock `sponsor` AND `selfpromo`.
- Three sets, channel-disjoint: `features.npz` (train/val, tuned by channel-grouped CV), `features_tail.npz` (holdout 1: graded once on 2026-09-21, now USED), `features_holdout2.npz` (holdout 2: fresh crawl videos).

## Measuring honestly

- **Grade on ad time skipped, not just region recall.** Recall counts a read as found if one line of it is skipped. `replay.py`'s `ad cov` is the share of ad seconds actually skipped; report it with the real show lost per video and the worst single video.
- Tune on cross-validation; grade each holdout once; never choose among systems by their holdout numbers.
- Every stage after the marker trains on OUT-OF-FOLD scores with the marker's own channel folds.
- Language-model answers (qwen via Ollama) are recorded once (`data/confirm_verdicts*.jsonl`, `*.edges.jsonl`) and replayed; only rules that select over recorded answers can be scored that way.
- Never leave qwen in VRAM: the scripts unload it when they finish.

## Where things stand (2026-09-21)

Holdout 1, once (78 videos, 87 reads): cue patterns 28.0% of ad time at 19.3 s lost show per video; marker alone 31.5% at 8.3 s; free tier (context model + edge heads) 49.6% at 9.9 s; qwen tier (judge + qwen confirm + qwen edges) 63.4% at 19.9 s. Details and holdout 2 in the vault's daily note for 2026-09-21.
