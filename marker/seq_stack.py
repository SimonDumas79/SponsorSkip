"""One sequence model over the detectors' scores, with inside / start / resume heads on a shared trunk. (2026-09-22)

Two research suggestions in one test. (1) The end-to-end sequence model (sequence_model.py) failed
on raw line features; the boundary research warned a repeat would too, unless it read the proven
detector scores instead. (2) Multi-task: inside, start and resume heads sharing one representation,
instead of a context model plus two separately trained edge heads.

Input per line: the logits of the six level-1 detectors (the frozen marker, meaning-only,
structure-only, potion, the conv sequence model, fine-tuned MiniLM seed 0), the cue and description
columns, and position in the video. Trunk: a 2-layer bidirectional GRU over the whole video. Heads:
inside (weighted BCE), start and resume (soft labels, as edge_heads.py). Out of fold with the same
channel folds; regions from the inside head, edges placed by its own start/resume heads (the
edge_heads.place windows), graded by the pooled rule against the stack + fine-tuned reference.

    python marker/seq_stack.py
"""

import ctypes
import sys

import numpy as np
import torch
from torch import nn

from build_production import pooled
from context_stack import logit
from detector_bakeoff import CACHE, graded, line, pick
from edge_heads import place, soft
from stack_check import sweep_fine
from train import DATA

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(8)
HIDDEN, STEPS, LR, CHUNK, BATCH = 48, 1500, 2e-3, 300, 32   # trained on random 300-line chunks, 32 per step


class Trunk(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.ReLU())
        self.gru = nn.GRU(HIDDEN, HIDDEN, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.heads = nn.Linear(2 * HIDDEN, 3)

    def forward(self, x):
        h, _ = self.gru(self.inp(x))
        return self.heads(h)   # (batch, lines, 3): inside, start, resume


def main() -> int:
    rows, is_start, is_resume = pooled()
    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    bake = dict(np.load(CACHE))
    ft = dict(np.load(DATA / "finetune_streams.npz"))
    streams = [level1, bake["meaning"], bake["structure"], bake["potion"], np.load(DATA / "sequence_oof_h32_r7.npy"),
               np.load(DATA / "finetune_oof_seed0.npy")]
    names = [str(n) for n in rows.feature_names]
    extra = rows.X[:, [i for i, n in enumerate(names) if n.startswith("cue_") or n.startswith("desc_") or n == "fraction_in"]]
    X = np.column_stack([np.clip(logit(s), -8, 8) for s in streams] + [extra]).astype(np.float32)
    targets = np.column_stack([rows.y.astype(np.float32), soft(is_start, rows.video), soft(is_resume, rows.video)])
    videos = [str(v) for v in np.unique(rows.video)]
    idx = {v: np.flatnonzero(rows.video == v) for v in videos}
    chan_of = {v: rows.channel[idx[v][0]] for v in videos}

    channels = np.array(sorted(set(rows.channel)))
    np.random.default_rng(0).shuffle(channels)
    out = np.zeros((len(rows), 3), dtype=np.float32)
    pos_w = float((1 - rows.y).sum() / rows.y.sum())
    for f, fold in enumerate(np.array_split(channels, 5), 1):
        test_v = [v for v in videos if chan_of[v] in set(fold)]
        train_v = [v for v in videos if chan_of[v] not in set(fold)]
        tr_rows = np.concatenate([idx[v] for v in train_v])
        mean, std = X[tr_rows].mean(0), X[tr_rows].std(0) + 1e-6
        torch.manual_seed(0)
        model = Trunk(X.shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        bce_in = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))
        bce_edge = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0))
        rng = np.random.default_rng(0)
        Xn = (X - mean) / std
        lengths = np.array([len(idx[v]) for v in train_v], dtype=float)
        model.train()
        for step in range(STEPS):
            # videos drawn in proportion to length, then a random CHUNK-line window in each
            picks = rng.choice(len(train_v), size=BATCH, p=lengths / lengths.sum())
            xs, ts = [], []
            for k in picks:
                r = idx[train_v[k]]
                a = int(rng.integers(0, max(1, len(r) - CHUNK + 1)))
                rr = r[a:a + CHUNK]
                pad = CHUNK - len(rr)
                xs.append(np.pad(Xn[rr], ((0, pad), (0, 0))))
                ts.append(np.pad(targets[rr], ((0, pad), (0, 0)), constant_values=-1))
            x, t = torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ts))
            o = model(x)
            m = t[..., 0] >= 0
            loss = bce_in(o[..., 0][m], t[..., 0][m]) + 0.5 * (bce_edge(o[..., 1][m], t[..., 1][m]) + bce_edge(o[..., 2][m], t[..., 2][m]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            for v in test_v:
                r = idx[v]
                out[r] = torch.sigmoid(model(torch.from_numpy((X[r] - mean) / std)[None])[0]).numpy()
        print(f"  fold {f}/5 done", flush=True)
    np.save(DATA / "seq_stack_oof.npy", out)

    ref = graded(sweep_fine(ft["stack_ft0"], rows, p_start, p_end), rows)
    own = graded(sweep_fine(out[:, 0], rows, out[:, 1], out[:, 2]), rows)
    shipped_heads = graded(sweep_fine(out[:, 0], rows, p_start, p_end), rows)
    for budget in (5, 10):
        print(line(f"B={budget:>2} stack + fine-tuned s0 (reference)", pick(ref, budget)))
        print(line(f"B={budget:>2} sequence over scores, its own heads", pick(own, budget)))
        print(line(f"B={budget:>2} sequence over scores, shipped heads", pick(shipped_heads, budget)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
