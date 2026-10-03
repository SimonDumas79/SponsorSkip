"""Grade v3 and challenger J with soft lead-ins left out of the score. (2026-10-03)

Simon's point: SponsorBlock often starts a segment at the soft lead-in, the stretch where the show
segues toward the sponsor while it is still the show's own content ("speaking of staying safe
online..."). A marker reading for the ad cannot be expected to catch that, and nobody minds hearing
it, so missing it should not count as missed ad time.

What marks a lead-in: Claude placed every pooled read's edges without seeing SponsorBlock's
(verify_edges.py, data/edge_verify.jsonl), told to start at the first line of the promotion itself,
including a hard lead-in like "this video is sponsored by". Where SponsorBlock starts EARLIER than
Claude, the lines between are SponsorBlock's soft lead-in. Three gradings, on the same skips:

  standard     as every result so far: SponsorBlock's lines are the ad
  soft-neutral SponsorBlock's soft lead-in counts as neither ad nor show (Simon's rule)
  both-neutral every line where the two starts disagree counts as neither, either direction
               (also Claude's "a quick word from..." lines that SponsorBlock calls show)

Neutral lines are left out of ad time (missed or skipped) and out of show lost. Reads Claude could
not place keep the standard labels. Graded on the 160 v3 CV videos, so this is a CV number, not a
holdout; it changes the yardstick, not the system.

    python marker/lead_in_grade.py
"""
import datetime
import json
import sys

import numpy as np

from build_production import pooled
from candidate import sub
from experiments import runs
from grow_sweep import grow
from edge_heads import place
from replay import regions_by_video
from train import DATA, LAST_LINE_SECONDS

SYSTEMS = {  # name: (detect threshold, grow bar or None for the edge heads); PREREGISTRATION.md D and J
    "v3 B=10": (0.99593, None), "J  B=10": (0.99593, 0.95),
    "v3 B=5": (0.99838, None), "J  B=5": (0.99838, 0.98),
}


def neutral_masks(U) -> tuple[dict, dict, int]:
    """Per video, the line mask of SponsorBlock's soft lead-ins and of every start disagreement."""
    by_video = {}
    for line in (DATA / "edge_verify.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["set"] == "pooled" and r["claude_start"] is not None:
            by_video.setdefault(r["video"], []).append(r)
    soft, both, checked = {}, {}, 0
    for vid in np.unique(U.video):
        r = np.flatnonzero(U.video == vid)
        reads = runs(U.y[r])
        s, b = np.zeros(len(r), bool), np.zeros(len(r), bool)
        for e in by_video.get(str(vid), []):
            if e["read"] >= len(reads) or reads[e["read"]][0] != e["sb_start"]:
                continue  # the read numbering must line up with this pool's labels, or skip it
            checked += 1
            sb, cl = e["sb_start"], e["claude_start"]
            if cl > sb:
                s[sb:cl] = b[sb:cl] = True
            elif cl < sb:
                b[cl:sb] = True
        soft[str(vid)], both[str(vid)] = s, b
    return soft, both, checked


def grade(kept: dict, U, neutral: dict | None) -> dict:
    ad_total = ad_skipped = 0.0
    lost, lead_missed = [], 0.0
    for vid in np.unique(U.video):
        r = np.flatnonzero(U.video == vid)
        y = U.y[r] == 1
        flags = np.zeros(len(r), bool)
        for a, b in kept.get(str(vid), []):
            flags[a:b] = True
        s = U.start_seconds[r]
        on = np.diff(np.append(s, s[-1] + LAST_LINE_SECONDS))
        n = neutral[str(vid)] if neutral is not None else np.zeros(len(r), bool)
        ad_total += on[y & ~n].sum()
        ad_skipped += on[y & ~n & flags].sum()
        lost.append(on[~y & ~n & flags].sum())
        lead_missed += on[y & n & ~flags].sum()
    lost = np.array(lost)
    return {"coverage": ad_skipped / ad_total, "show": lost.mean(), "worst": lost.max(),
            "ad_total": ad_total, "lead_missed": lead_missed}


def main() -> int:
    P, _, _ = pooled()
    _, _, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    fs = np.load(DATA / "finetune_oof_edge_start_seed0.npy")
    fe = np.load(DATA / "finetune_oof_edge_resume_seed0.npy")
    d = json.load(open(DATA / "sb_dates.json"))
    cutoff = datetime.datetime(2022, 4, 1).timestamp() * 1000
    unseen = np.array([not (d.get(str(v), {}).get("first") and d[str(v)]["first"] < cutoff) for v in P.video])
    U = sub(P, unseen)
    oof3 = np.load(DATA / "stack_sbml_oof_s0.npy")[1]
    ps, pe = ((p_start + fs) / 2)[unseen], ((p_end + fe) / 2)[unseen]
    soft, both, checked = neutral_masks(U)

    def secs(masks):
        t = 0.0
        for vid in np.unique(U.video):
            r = np.flatnonzero(U.video == vid)
            s = U.start_seconds[r]
            t += np.diff(np.append(s, s[-1] + LAST_LINE_SECONDS))[masks[str(vid)]].sum()
        return t
    print(f"v3 CV videos: {U.videos} videos, {U.reads} reads; {checked} reads have Claude's start")
    print(f"soft lead-in seconds (SponsorBlock earlier than Claude): {secs(soft):.0f}; "
          f"any start disagreement: {secs(both):.0f}\n")
    print(f"{'system':<9} {'grading':<13} {'ad time':>8} {'show lost':>10} {'worst':>7}  lead-in left playing")
    for name, (th, low) in SYSTEMS.items():
        found = regions_by_video(oof3, U, th, 1)
        kept = place(found, U, ps, pe) if low is None else grow(found, U, oof3, low)
        for label, masks in (("standard", None), ("soft-neutral", soft), ("both-neutral", both)):
            g = grade(kept, U, masks)
            extra = f"  {g['lead_missed']:.0f} s" if masks is soft else ""
            print(f"{name:<9} {label:<13} {g['coverage']:8.1%} {g['show']:8.1f} s {g['worst']:5.0f} s{extra}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
