"""Search for a better marker, judged by cross-validation, leaning towards not losing video.

The question every candidate is asked: of the real sponsor reads, how many does
it catch when it may only lose a few seconds of real show per video? That is
the "alone" tier, where nothing checks the marker's flags, so a false alarm
skips show the viewer wanted.

The levers
    epochs, lr, weight_decay, batch   how the model is trained
    pos_scale                         how much a missed ad line costs against a
                                      false one: 1.0 is train.py's 14x, 0.07 is equal
    hidden, dropout                   0 = linear; otherwise a two-layer network
    drop                              feature groups left out: position, seam, cues, meaning
    stack                             1 = two models, one on the meaning columns and one on the rest
                                      (seam, cues, position), each trained on its own and their
                                      logits averaged. Trained jointly they would just be one linear
                                      model; trained apart, each has to find reads on its own, so
                                      they miss different reads and the average covers both
    smooth, min_region, max_region    post-processing: bridge gaps, drop regions too
                                      short to be a read or too long to be one
    max_video                         a per-video skip allowance: the most confident
                                      regions are kept until it is spent

How a candidate is scored
    Five-fold cross-validation grouped by channel, repeated with three different
    shuffles of the channels so a lucky split cannot win. Every row is scored by
    a model that never saw its channel. Then, for each budget from 1 to 15 s of
    show lost per video, the best threshold under that budget is found and its
    recall kept; a threshold only qualifies if no single video loses more than
    WORST_CAP seconds, because an average hides the one video that loses three
    minutes. FITNESS is the average of those fifteen recalls: a candidate wins
    by catching more ads at every low budget, not at one hand-picked point.
    Beside it: recall at 5, 10 and 15 s; the worst single video at the 10 s
    threshold; and, for the checker tier, the show lost at 90% recall.

The stages (--stage)
    grid     the baseline, then every lever changed on its own
    greedy   stack the winning value of each lever, keeping what helps
    ga       a genetic algorithm over every knob, seeded with the grid's best
    all      the three in order
Results append to data/experiments.jsonl and a finished candidate is never
re-run, so a stopped search resumes where it was.

The zoo (data/zoo/)
    Every candidate that reaches the frontier (within 0.01 of the best fitness
    so far) is saved: its config, its metrics, its out-of-fold scores per seed,
    and a final model trained on every row. Saved candidates can be cloned
    later: their configs seed new mutations, and their out-of-fold scores are
    exactly what a stacked (level-2) model needs to train on without leakage.

The holdout is never touched here. Tune on cross-validation; look at the
holdout once, at the end, with train.py --holdout.
"""

import argparse
import ctypes
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from score import score_video, smooth as bridge
from train import DATA, LAST_LINE_SECONDS, SLACK, load, scaling

RESULTS = DATA / "experiments.jsonl"
LEADERBOARD = DATA / "experiments_leaderboard.md"
ZOO = DATA / "zoo"
FRONTIER = 0.01   # a candidate this close to the best fitness so far is worth keeping

GROUPS = {"meaning": (0, 384), "seam": (384, 387), "cues": (387, 393), "position": (393, 395)}

BASE = {"epochs": 20, "lr": 1e-3, "weight_decay": 0.01, "batch": 256, "pos_scale": 1.0,
        "hidden": 0, "dropout": 0.0, "drop": [], "smooth": 3, "min_region": 1, "max_region": 0, "max_video": 0,
        "stack": 0}

SPACE = {
    "epochs": [10, 20, 50, 100],
    "lr": [3e-4, 1e-3, 3e-3, 1e-2],
    "weight_decay": [0.0, 0.01, 0.1, 0.3, 1.0],
    "batch": [64, 256, 1024],
    "pos_scale": [1.0, 0.5, 0.25, 0.1, 0.07],
    "hidden": [0, 16, 32, 64],
    "dropout": [0.0, 0.2, 0.5],
    "drop": [[], ["position"], ["seam"], ["cues"], ["position", "seam"], ["meaning"]],
    "smooth": [1, 3, 5],
    "min_region": [1, 3, 4, 5, 7, 10],
    "max_region": [0, 180, 120, 90],   # seconds; a flagged region longer than this is dropped (0 = off)
    "max_video": [0, 300, 180, 120],   # seconds a video may have flagged in total: most confident regions first (0 = off)
    "stack": [0, 1],   # 1 = two models trained SEPARATELY (meaning columns; everything else), their logits averaged
}

BUDGETS = list(range(1, 16))   # seconds of show lost per video the alone tier may spend
WORST_CAP = 60.0               # and no single video may lose more than this, whatever the average
GATE = 0.90                    # recall the checker tier must keep
# Dense where it matters: with a 14x pos_weight the useful thresholds crowd above 0.9.
THRESHOLDS = np.unique(np.round(np.concatenate([
    np.arange(0.05, 0.90, 0.01), np.arange(0.90, 0.99, 0.002), np.arange(0.99, 0.9995, 0.0005)]), 4))


# ----------------------------------------------------------------------------- one candidate


def canonical(cfg: dict) -> dict:
    """Fill in levers added after a result was recorded, and blank the ones that do nothing."""
    cfg = dict(BASE, **cfg)
    if cfg["hidden"] == 0:
        cfg["dropout"] = 0.0   # dropout only exists inside a hidden layer
    return cfg


def key(cfg: dict) -> str:
    cfg = canonical(cfg)
    if not cfg["stack"]:
        del cfg["stack"]   # keeps the keys of results recorded before the lever existed
    return json.dumps(cfg, sort_keys=True)


def describe(cfg: dict) -> str:
    """The candidate as its differences from the baseline, or 'baseline'."""
    diff = [f"{k}={v}" for k, v in cfg.items() if v != BASE[k] and not (k == "dropout" and cfg["hidden"] == 0)]
    return " ".join(diff) or "baseline"


def kept_columns(cfg: dict) -> np.ndarray:
    keep = np.ones(395, dtype=bool)
    for group in cfg["drop"]:
        lo, hi = GROUPS[group]
        keep[lo:hi] = False
    return np.flatnonzero(keep)


def branches(cfg: dict) -> list[np.ndarray]:
    """The column sets trained as separate models: one set, or two when stacking."""
    columns = kept_columns(cfg)
    if not cfg.get("stack"):
        return [columns]
    meaning, rest = columns[columns < 384], columns[columns >= 384]
    return [c for c in (meaning, rest) if len(c)]


def build(cfg: dict, n_in: int) -> nn.Module:
    if cfg["hidden"] == 0:
        return nn.Linear(n_in, 1)
    return nn.Sequential(nn.Linear(n_in, cfg["hidden"]), nn.ReLU(), nn.Dropout(cfg["dropout"]),
                         nn.Linear(cfg["hidden"], 1))


def fit(cfg: dict, X_tr: np.ndarray, y_tr: np.ndarray, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X_tr).float()
    yt = torch.from_numpy(y_tr).float()
    model = build(cfg, Xt.shape[1])
    ratio = (len(yt) - yt.sum()) / yt.sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ratio * cfg["pos_scale"]]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    for _ in range(cfg["epochs"]):
        model.train()
        order = torch.randperm(len(Xt))
        for i in range(0, len(Xt), cfg["batch"]):
            idx = order[i:i + cfg["batch"]]
            optimizer.zero_grad()
            loss = loss_fn(model(Xt[idx]).squeeze(1), yt[idx])
            loss.backward()
            optimizer.step()
    return model.eval()


def cross_validate(cfg: dict, rows, seed: int, folds: int = 5) -> np.ndarray:
    """One out-of-fold probability per row, from a model that never saw the row's channel."""
    channels = np.array(sorted(set(rows.channel)))
    np.random.default_rng(seed).shuffle(channels)
    oof = np.full(len(rows), np.nan, dtype=np.float32)
    for fold in np.array_split(channels, folds):
        test = np.isin(rows.channel, fold)
        logits = np.zeros(int(test.sum()), dtype=np.float32)
        for columns in branches(cfg):
            X = rows.X[:, columns]
            mean, std = scaling(X[~test])
            model = fit(cfg, (X[~test] - mean) / std, rows.y[~test], seed)
            with torch.no_grad():
                scaled = torch.from_numpy((X[test] - mean) / std).float()
                logits += model(scaled).squeeze(1).numpy()
        oof[test] = 1.0 / (1.0 + np.exp(-logits / len(branches(cfg))))
    assert not np.isnan(oof).any()
    return oof


# ----------------------------------------------------------------------------- grading


def runs(flags: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([0], flags.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def video_meta(rows) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per video: its row numbers, its labels, and how long each caption line is on screen."""
    out = []
    for vid in np.unique(rows.video):
        r = np.flatnonzero(rows.video == vid)
        s = rows.start_seconds[r]
        out.append((r, rows.y[r], np.diff(np.append(s, s[-1] + LAST_LINE_SECONDS))))
    return out


def grade_at(probs: np.ndarray, meta, threshold: float, smooth_w: int, min_region: int, max_region: float,
             max_video: float) -> dict:
    found = total = windows = right = 0
    lost = np.zeros(len(meta))
    for k, (r, labels, on_screen) in enumerate(meta):
        flags = bridge((probs[r] >= threshold).astype(np.int8), smooth_w)
        if min_region > 1 or max_region > 0:
            for lo, hi in runs(flags):
                too_short = hi - lo < min_region
                too_long = max_region > 0 and on_screen[lo:hi].sum() > max_region
                if too_short or too_long:
                    flags[lo:hi] = 0
        if max_video > 0:
            # Keep the most confident regions until the video's skip allowance is spent.
            spent = 0.0
            for lo, hi in sorted(runs(flags), key=lambda run: -probs[r][run[0]:run[1]].max()):
                seconds = on_screen[lo:hi].sum()
                if spent + seconds > max_video:
                    flags[lo:hi] = 0
                else:
                    spent += seconds
        f, t, w, ri = score_video(labels, flags, SLACK)
        found += f
        total += t
        windows += w
        right += ri
        lost[k] = on_screen[(flags == 1) & (labels == 0)].sum()
    return {"th": float(threshold), "recall": found / max(total, 1), "windows": windows / len(meta),
            "show": float(lost.mean()), "worst": float(lost.max()), "over15": float((lost > 15).mean())}


def summarise(probs: np.ndarray, meta, cfg: dict) -> dict:
    """The whole recall-against-cost curve, boiled down to the numbers the search ranks by."""
    curve = [grade_at(probs, meta, th, cfg["smooth"], cfg["min_region"], cfg["max_region"], cfg["max_video"])
             for th in THRESHOLDS]

    def best_under(budget: float):
        """Most reads caught while the average video loses <= budget and no video loses > WORST_CAP."""
        ok = [p for p in curve if p["show"] <= budget and p["worst"] <= WORST_CAP]
        return max(ok, key=lambda p: (p["recall"], -p["show"])) if ok else None

    recalls = [(best_under(b) or {"recall": 0.0})["recall"] for b in BUDGETS]
    at10 = best_under(10) or {"th": float("nan"), "worst": float("nan"), "over15": float("nan")}
    at_gate = [p for p in curve if p["recall"] >= GATE]
    cheapest = min(at_gate, key=lambda p: p["show"]) if at_gate else None
    return {
        "fitness": float(np.mean(recalls)),
        "r5": recalls[4], "r10": recalls[9], "r15": recalls[14],
        "th10": at10["th"], "worst10": at10["worst"], "over15_10": at10["over15"],
        "show90": cheapest["show"] if cheapest else float("nan"),
        "windows90": cheapest["windows"] if cheapest else float("nan"),
        "th90": cheapest["th"] if cheapest else float("nan"),
    }


def evaluate(cfg: dict, rows, meta, seeds: tuple[int, ...]) -> tuple[dict, list[np.ndarray]]:
    oofs = [cross_validate(cfg, rows, s) for s in seeds]
    per_seed = [summarise(oof, meta, cfg) for oof in oofs]
    out = {k: float(np.nanmean([m[k] for m in per_seed])) for k in per_seed[0]}
    out["fitness_std"] = float(np.std([m["fitness"] for m in per_seed]))
    return out, oofs


def fit_full(cfg: dict, rows, seed: int = 0) -> list[dict]:
    """Train every branch of a candidate on all of these rows; each part carries its own scaling."""
    parts = []
    for columns in branches(cfg):
        X = rows.X[:, columns]
        mean, std = scaling(X)
        parts.append({"columns": columns, "mean": mean, "std": std,
                      "model": fit(cfg, (X - mean) / std, rows.y, seed=seed)})
    return parts


def predict_full(parts: list[dict], X: np.ndarray) -> np.ndarray:
    """The candidate's probability per row: the sigmoid of its branches' average logit."""
    z = np.zeros(len(X), dtype=np.float32)
    for part in parts:
        with torch.no_grad():
            scaled = torch.from_numpy((X[:, part["columns"]] - part["mean"]) / part["std"]).float()
            z += part["model"](scaled).squeeze(1).numpy()
    return 1.0 / (1.0 + np.exp(-z / len(parts)))


def save_to_zoo(cfg: dict, metrics: dict, oofs: list[np.ndarray], rows, seeds: tuple[int, ...]) -> Path:
    """Keep a frontier candidate: config, metrics, out-of-fold scores, and a model trained on every row."""
    import hashlib
    tag = f"{metrics['fitness']:.3f}_{hashlib.sha1(key(cfg).encode()).hexdigest()[:8]}"
    folder = ZOO / tag
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    for seed, oof in zip(seeds, oofs):
        np.save(folder / f"oof_seed{seed}.npy", oof)
    parts = [{"state": p["model"].state_dict(), "columns": torch.from_numpy(p["columns"]),
              "mean": torch.from_numpy(p["mean"]), "std": torch.from_numpy(p["std"])}
             for p in fit_full(cfg, rows)]   # the final score is the average of the branches' logits
    torch.save({
        "branches": parts, "config": cfg,
        "thresholds": {"alone10": float(metrics["th10"]), "checker90": float(metrics["th90"])},
        "metrics": metrics, "trained_on": "every row of features.npz", "videos": np.unique(rows.video).tolist(),
    }, folder / "model.pt")
    with (ZOO / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"tag": tag, "fitness": metrics["fitness"], "cfg": cfg, "describe": describe(cfg)}) + "\n")
    return folder


# ----------------------------------------------------------------------------- bookkeeping


def load_results() -> dict[str, dict]:
    if not RESULTS.exists():
        return {}
    done = {}
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            rec["cfg"] = canonical(rec["cfg"])
            rec["key"] = key(rec["cfg"])
            done.setdefault(rec["key"], rec)   # a duplicate under the canonical key is the same candidate
    return done


class Search:
    def __init__(self, rows, seeds, results):
        self.rows, self.meta, self.seeds, self.results = rows, video_meta(rows), seeds, results
        self.evaluated = 0
        self.best = max((r["metrics"]["fitness"] for r in results.values()), default=0.0)

    def run(self, cfg: dict, stage: str) -> dict:
        k = key(cfg)
        if k in self.results and self.results[k]["seeds"] == list(self.seeds):
            return self.results[k]["metrics"]
        t0 = time.time()
        metrics, oofs = evaluate(cfg, self.rows, self.meta, self.seeds)
        rec = {"key": k, "cfg": cfg, "stage": stage, "seeds": list(self.seeds), "metrics": metrics,
               "seconds": round(time.time() - t0, 1)}
        kept = ""
        if metrics["fitness"] >= self.best - FRONTIER or cfg == BASE:
            rec["zoo"] = save_to_zoo(cfg, metrics, oofs, self.rows, self.seeds).name
            kept = "  -> zoo"
        self.best = max(self.best, metrics["fitness"])
        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self.results[k] = rec
        self.evaluated += 1
        m = metrics
        print(f"  [{stage}] fit {m['fitness']:.3f}±{m['fitness_std']:.3f}  r@5 {m['r5']:.2f}  r@10 {m['r10']:.2f}  "
              f"r@15 {m['r15']:.2f}  worst@10 {m['worst10']:4.0f}s  show@90 {m['show90']:4.0f}s  "
              f"({rec['seconds']:.0f}s)  {describe(cfg)}{kept}", flush=True)
        return metrics

    def ranked(self) -> list[dict]:
        return sorted(self.results.values(), key=lambda r: -r["metrics"]["fitness"])


def leaderboard(search: Search, top: int = 20) -> str:
    base = search.results.get(key(BASE))
    lines = ["# Marker search leaderboard", "",
             f"Fitness = mean recall over budgets of 1-15 s of show lost per video, "
             f"{len(search.seeds)}-seed channel-grouped 5-fold CV, {len(search.results)} candidates.", "",
             "| # | fitness | r@5s | r@10s | r@15s | worst video @10s | show lost @90% recall | windows @90% | candidate |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, rec in enumerate(search.ranked()[:top], start=1):
        m = rec["metrics"]
        lines.append(f"| {i} | {m['fitness']:.3f} ±{m['fitness_std']:.3f} | {m['r5']:.1%} | {m['r10']:.1%} | {m['r15']:.1%} "
                     f"| {m['worst10']:.0f} s | {m['show90']:.0f} s | {m['windows90']:.1f} | `{describe(rec['cfg'])}` |")
    if base:
        m = base["metrics"]
        lines += ["", f"Baseline (train.py as is): fitness {m['fitness']:.3f} ±{m['fitness_std']:.3f}, "
                  f"r@5 {m['r5']:.1%}, r@10 {m['r10']:.1%}, r@15 {m['r15']:.1%}, worst@10 {m['worst10']:.0f} s, "
                  f"show@90 {m['show90']:.0f} s.",
                  f"One read is {100 / max(search.rows.reads, 1):.1f} points of recall on {search.rows.reads} reads: "
                  "differences inside ±1 read are noise."]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- stages


def stack_stage(search: Search, top: int = 8) -> None:
    print(f"\nSTACK: the {top} best candidates so far, each re-run as two separately trained models", flush=True)
    search.run(dict(BASE, stack=1), "stack")
    for rec in [r for r in search.ranked() if not r["cfg"]["stack"]][:top]:
        search.run(dict(rec["cfg"], stack=1), "stack")


def grid(search: Search) -> None:
    print("\nGRID: the baseline, then each lever on its own", flush=True)
    search.run(dict(BASE), "grid")
    for lever, values in SPACE.items():
        if lever == "dropout":
            continue   # only means something with a hidden layer; paired below
        for v in values:
            if v == BASE[lever]:
                continue
            cfg = dict(BASE, **{lever: v})
            search.run(cfg, "grid")
            if lever == "hidden":
                search.run(dict(cfg, dropout=0.2), "grid")


def greedy(search: Search) -> dict:
    print("\nGREEDY: stack each lever's best value, keep what helps", flush=True)
    base_fit = search.run(dict(BASE), "greedy")["fitness"]
    gains = []
    for lever, values in SPACE.items():
        if lever == "dropout":
            continue
        best_v, best_f = None, base_fit
        for v in values:
            if v == BASE[lever]:
                continue
            f = search.run(dict(BASE, **{lever: v}), "greedy")["fitness"]
            if f > best_f:
                best_v, best_f = v, f
        if best_v is not None:
            gains.append((best_f - base_fit, lever, best_v))
    current, current_fit = dict(BASE), base_fit
    for gain, lever, v in sorted(gains, reverse=True):
        trial = dict(current, **{lever: v})
        f = search.run(trial, "greedy")["fitness"]
        if f > current_fit:
            current, current_fit = trial, f
            print(f"  kept {lever}={v}: fitness {f:.3f}", flush=True)
        else:
            print(f"  dropped {lever}={v}: {f:.3f} did not beat {current_fit:.3f}", flush=True)
    print(f"  greedy result: {describe(current)} -> {current_fit:.3f}", flush=True)
    return current


def ga(search: Search, rng: random.Random, pop_size: int, generations: int) -> dict:
    print(f"\nGA: population {pop_size}, {generations} generations", flush=True)

    def random_cfg() -> dict:
        return {k: rng.choice(v) for k, v in SPACE.items()}

    def mutate(cfg: dict) -> dict:
        child = dict(cfg)
        for lever, values in SPACE.items():
            if rng.random() < 0.2:
                i = values.index(child[lever])
                step = rng.choice([-1, 1])
                child[lever] = values[max(0, min(len(values) - 1, i + step))] if rng.random() < 0.7 else rng.choice(values)
        return child

    def crossover(a: dict, b: dict) -> dict:
        return {k: (a if rng.random() < 0.5 else b)[k] for k in SPACE}

    def tournament(pop: list[dict]) -> dict:
        return max(rng.sample(pop, min(3, len(pop))), key=lambda c: search.run(c, "ga")["fitness"])

    seeds_from_grid = [r["cfg"] for r in search.ranked()[:max(1, pop_size - 4)]]
    population = seeds_from_grid + [random_cfg() for _ in range(pop_size - len(seeds_from_grid))]
    seen = {key(c) for c in population}
    for gen in range(1, generations + 1):
        scored = sorted(population, key=lambda c: -search.run(c, "ga")["fitness"])
        best = search.run(scored[0], "ga")
        print(f"  generation {gen}: best {best['fitness']:.3f}  {describe(scored[0])}", flush=True)
        if gen == generations:
            break
        children = scored[:2]   # elitism: the top two survive unchanged
        tries = 0
        while len(children) < pop_size and tries < 200:
            tries += 1
            child = mutate(crossover(tournament(scored), tournament(scored)))
            if key(child) not in seen:
                seen.add(key(child))
                children.append(child)
        while len(children) < pop_size:
            children.append(random_cfg())
        population = children
    return search.ranked()[0]["cfg"]


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["grid", "greedy", "ga", "stack", "all"], default="all")
    ap.add_argument("--seeds", type=int, default=3, help="fold shuffles per candidate")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads torch may use")
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--quick", action="store_true", help="smoke test: baseline only, one seed")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    if sys.platform == "win32":   # Simon is at this PC: stay out of the way
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)

    rows = load(DATA / "features.npz")
    seeds = (0,) if args.quick else tuple(range(args.seeds))
    search = Search(rows, seeds, load_results())
    print(f"{len(rows)} rows, {len(set(rows.channel))} channels, {rows.reads} reads; "
          f"{len(search.results)} candidates already on disk; seeds {seeds}; {args.threads} threads", flush=True)

    t0 = time.time()
    if args.quick:
        search.run(dict(BASE), "quick")
    else:
        if args.stage in ("grid", "all"):
            grid(search)
        if args.stage in ("greedy", "all"):
            greedy(search)
        if args.stage == "stack":
            stack_stage(search)
            ga(search, random.Random(1), args.pop, args.generations)
        if args.stage in ("ga", "all"):
            ga(search, random.Random(0), args.pop, args.generations)

    board = leaderboard(search)
    LEADERBOARD.write_text(board, encoding="utf-8")
    print("\n" + board)
    print(f"\n{search.evaluated} candidates trained in {(time.time() - t0) / 60:.1f} min; "
          f"leaderboard written to {LEADERBOARD.relative_to(DATA.parent.parent)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
