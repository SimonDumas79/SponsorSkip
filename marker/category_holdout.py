"""Pre-registration 4's secondary table: each system's share of each Claude category skipped, on a holdout.

Regions come from grade_holdout5.py (data/holdout5_regions.json) and grade_level4.py; categories from
claude_label.py --categories on the same videos. A line counts for a category if its start falls inside
one of Claude's segments of that category. End-of-video plugs count (Simon, 2026-09-24).

    python marker/category_holdout.py holdout5
"""

import collections
import json
import sys

import numpy as np

from train import DATA, load

sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "holdout5"
    T = load(DATA / f"features_{which}.npz")
    systems = json.loads((DATA / f"{which}_regions.json").read_text(encoding="utf-8"))
    systems = {k: v for k, v in systems.items() if "B=10" in k}
    systems["I  level 4 (served Claude)"] = json.loads((DATA / f"level4_{which}_regions.json").read_text())
    labels = {r["videoID"]: r["segments"] for r in
              json.loads((DATA / f"claude_labels_categories_{which}.json").read_text(encoding="utf-8"))}
    local = {str(v): np.flatnonzero(T.video == v) for v in np.unique(T.video)}
    member = collections.defaultdict(lambda: np.zeros(len(T.y), dtype=bool))
    for v, r in local.items():
        t = T.start_seconds[r]
        for s in labels.get(v, []):
            k = f"{s['category']}/{s['style']}"
            member[k][r[(t >= s["start"]) & (t < max(s["end"], s["start"] + 0.1))]] = True
    member["ANY promo (Claude)"] = np.any(np.stack(list(member.values())), axis=0)
    member["SponsorBlock labels"] = T.y.astype(bool)
    cats = sorted(member, key=lambda k: -member[k].sum())
    print(f"{which}: lines per category -- " + ", ".join(f"{k} {member[k].sum()}" for k in cats) + "\n")
    print(f"{'system':<44}" + "".join(f"{k.split('/')[0][:11] + ('/' + k.split('/')[1][:4] if '/' in k else ''):>17}" for k in cats))
    for name, kept in systems.items():
        skip = np.zeros(len(T.y), dtype=bool)
        for v, spans in kept.items():
            for a, z in spans:
                skip[local[v][a:z]] = True
        row = "".join(f"{(skip & member[k]).sum() / max(member[k].sum(), 1):17.1%}" for k in cats)
        not_promo = skip & ~member["ANY promo (Claude)"] & ~member["SponsorBlock labels"]
        print(f"{name[:44]:<44}{row}   lines skipped that nobody calls promo: {not_promo.sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
