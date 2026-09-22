"""False alarms where they matter most: sponsor-FREE videos, the ones the pooled data never contains. (2026-09-22)

Every pooled video has a read, so the rule's lost-show numbers never saw a video that should be left
alone. sample_negatives.py found videos the SponsorBlock community marked (intro, outro, filler...)
but never marked a sponsor or self-promotion in; their captions are in data/captions_negatives/.
This runs the saved free-tier bundles end to end on each (predict.py's live path) and reports, per
the research round's metric: the share of videos with NO skip at all, the seconds wrongly skipped
per video, and the share losing more than 10 s and more than 60 s. English videos separately
(the likely scope). The worst videos are listed with titles, because "no sponsor mark" is not proof
of no sponsor: the list is what a Claude check should adjudicate before the number is trusted.

    python marker/negatives_eval.py
"""

import glob
import json
import sys

import numpy as np

from predict import FreeTier
from train import DATA

sys.stdout.reconfigure(encoding="utf-8")
BUNDLES = {"graded free tier (free_tier.pt)": DATA / "production" / "free_tier.pt",
           "pooled free tier (free_tier_pooled.pt)": DATA / "production" / "free_tier_pooled.pt"}


def main() -> int:
    caps = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(str(DATA / "captions_negatives" / "*.json")))]
    caps = [c for c in caps if c.get("lines")]
    cands = {c["videoID"]: c for c in json.loads((DATA / "candidates_negatives.json").read_text(encoding="utf-8"))}
    english = {c["videoID"] for c in caps if (c.get("language") or "").startswith("en")}
    print(f"{len(caps)} sponsor-free videos with captions ({len(english)} English; "
          f"tier B {sum(cands.get(c['videoID'], {}).get('tier') == 'B' for c in caps)})\n")
    out = {}
    for name, path in BUNDLES.items():
        tier = FreeTier(path)
        lost = {}
        for c in caps:
            segs = tier.segments(c)
            lost[c["videoID"]] = sum(s["end"] - s["start"] for s in segs)
        out[name] = lost
        for label, vids in (("all", [c["videoID"] for c in caps]), ("English", sorted(english))):
            x = np.array([lost[v] for v in vids])
            print(f"  {name:<40} {label:<8} n={len(x):3d}: no skip {np.mean(x == 0):5.0%}   "
                  f"mean {x.mean():5.1f} s   median {np.median(x):4.1f} s   >10 s {np.mean(x > 10):4.0%}   >60 s {np.mean(x > 60):4.0%}")
    name = next(iter(BUNDLES))
    worst = sorted(out[name].items(), key=lambda kv: -kv[1])[:10]
    titles = {c["videoID"]: (c.get("title") or "", c.get("channel") or "", c.get("language")) for c in caps}
    print(f"\n  worst videos under the {name}:")
    for v, s in worst:
        if s <= 0:
            break
        t, ch, lang = titles[v]
        print(f"    {s:5.0f} s  {v}  [{lang}] {ch[:20]:<20} {t[:60]}")
    json.dump(out, open(DATA / "negatives_eval.json", "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
