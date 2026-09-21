# SponsorSkip

`README.md` is the full reference; `marker/README.md` has the marker pipeline's run order.

## How Simon and Claude work on marker/

**Claude drives, Simon conducts** (his call, 2026-09-21). Claude writes the code; Simon decides what gets built and in what order. Explain what each change does and what the numbers mean in plain words, and check direction with him before a step that changes the plan.

- Build guide for the trainer: https://claude.ai/artifact/EMfbkydo8dVdLnaC49yfbg. Part 5 (Steps 1-9) is done in `marker/train.py`; Part 6 (channel-grouped cross-validation) is next.
- Run commands from the repo root: scripts open `marker/data/...` by relative paths.
- `features.npz` holds `X` (25887 x 395), `y`, `video`, `line`, `split` (train/val/...), `is_start`, `start_seconds`, `channel`, `feature_names`. `features_tail.npz` is the low-review holdout (every row `split == "holdout"`).
- Grading lives in `marker/score.py`: `score_video(labels, flagged, slack)` grades ONE video, `smooth(flags, window)` bridges one-line gaps. Region recall is gameable at a low threshold, so always report cost (regions flagged per video) beside it.
- Current numbers (seeded): holdout 59/70 (84.3%) at `--threshold 0.95`; the gate is 90%.
