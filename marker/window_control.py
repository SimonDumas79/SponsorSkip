"""The control for finetune_minilm.py: FROZEN MiniLM reading the same window. (2026-09-22)

The fine-tuned model changed two things at once: MiniLM's weights were trained on our labels, AND it
read the line plus WINDOW lines either side instead of one line. This keeps MiniLM frozen and gives
it the same window: each line's own embedding (already in the features), the embedding of its
17-line window, and the seam, cue and position columns, into the same linear layer the marker
uses. If this closes most of the gap, the window is what helped; if not, the training is.

Same pooled 205 videos, same channel folds, same context model, edge heads and rule as
finetune_eval.py, whose frozen and fine-tuned rows it prints beside.

    python marker/window_control.py
"""

import ctypes
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score, roc_auc_score

from build_production import pooled
from context_stack import context_features
from detector_bakeoff import cv_linear, graded, line, pick
from features import MODEL_NAME
from finetune_minilm import line_texts, pairs
from stack_check import oof_seeded, sweep_fine
from train import DATA

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(4)

EMB = DATA / "window_emb.npy"


def main() -> int:
    rows, _, _ = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    if EMB.exists():
        W = np.load(EMB)
    else:
        windows = [w for _, w in pairs(rows, line_texts(rows))]
        model = SentenceTransformer(MODEL_NAME, device="cuda")
        W = model.encode(windows, batch_size=128, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        del model
        torch.cuda.empty_cache()
        np.save(EMB, W)
    X = np.column_stack([rows.X[:, :395], W])   # line meaning + seam, cues, position + the window's meaning
    p1 = cv_linear(X, rows)
    y = rows.y
    print(f"pooled: {rows.videos} videos, {rows.reads} reads\n")
    print("LINE LEVEL (average precision, ROC AUC):")
    print(f"  frozen marker, one line                 {average_precision_score(y, level1):.3f}  {roc_auc_score(y, level1):.3f}")
    print(f"  frozen MiniLM, line + 17-line window    {average_precision_score(y, p1):.3f}  {roc_auc_score(y, p1):.3f}")
    for s in sorted(int(p.stem.split("seed")[1]) for p in DATA.glob("finetune_oof_seed*.npy") if "partial" not in p.name):
        ft = np.load(DATA / f"finetune_oof_seed{s}.npy")
        print(f"  fine-tuned MiniLM, seed {s}               {average_precision_score(y, ft):.3f}  {roc_auc_score(y, ft):.3f}")

    print("\nTHROUGH THE PIPELINE (context model + shipped heads, pooled rule):")
    g_ship = graded(sweep_fine(level2, rows, p_start, p_end), rows)
    g_win = graded(sweep_fine(oof_seeded(rows, context_features(rows, p1), 0), rows, p_start, p_end), rows)
    for budget in (5, 10):
        print(line(f"B={budget:>2} frozen marker, one line (shipped)", pick(g_ship, budget)))
        print(line(f"B={budget:>2} frozen MiniLM + window", pick(g_win, budget)))
    np.save(DATA / "window_control_oof.npy", p1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
