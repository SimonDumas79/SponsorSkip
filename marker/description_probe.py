"""Does the video description name the sponsor, and does that name land inside the labelled read?

One yt-dlp metadata request per video (no captions), a pause between them, stop on the first 429.
"""
import json, re, subprocess, sys, time
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
N, PAUSE = int(sys.argv[1]) if len(sys.argv) > 1 else 25, 8.0
OUT = DATA / "descriptions_probe.json"

d = np.load(DATA / "features.npz")
video, cat, y = d["video"], d["category"], d["y"]
sponsor_videos = sorted({str(v) for v in np.unique(video) if (cat[video == v] == "sponsor").any()})
rng = np.random.default_rng(0)
sample = list(rng.choice(sponsor_videos, size=min(N, len(sponsor_videos)), replace=False))

got = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
for vid in sample:
    if vid in got:
        continue
    p = subprocess.run([sys.executable, "-m", "yt_dlp", "--skip-download", "--no-warnings", "--quiet",
                        "--print", "%(description)s", f"https://www.youtube.com/watch?v={vid}"],
                       capture_output=True, text=True, encoding="utf-8", timeout=90,
                       creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW)
    if p.returncode != 0:
        print(f"{vid}: yt-dlp failed: {(p.stderr or '').strip()[-160:]}")
        if "429" in (p.stderr or ""):
            print("throttled; stopping"); break
        got[vid] = None
    else:
        got[vid] = p.stdout
    OUT.write_text(json.dumps(got, ensure_ascii=False, indent=0), encoding="utf-8")
    time.sleep(PAUSE)

# captions per video: text inside labelled reads vs outside
text = {}
with (DATA / "examples.jsonl").open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["videoID"] in got:
            text.setdefault(r["videoID"], []).append((r["i"], r["label"], r["text"]))

STOP = set("the and for you your with this that from video what how new all free get our are one out use more".split())
def brand_tokens(desc):
    """Candidate sponsor names: words near a link, a code, 'sponsor', or a percent-off line."""
    toks = set()
    for ln in desc.splitlines():
        low = ln.lower()
        if any(k in low for k in ("sponsor", "code ", "promo", "% off", "discount", "http", "partner", "thanks to", "check out")):
            for w in re.findall(r"[A-Za-z][A-Za-z0-9']{2,}", ln):
                if w.lower() not in STOP and not low.startswith("http"):
                    toks.add(w.lower())
            for dom in re.findall(r"https?://(?:www\.)?([a-z0-9-]+)\.", low):
                toks.add(dom)
    return toks

print(f"\n{len([v for v in got if got[v]])} descriptions fetched")
print(f"{'video':<12} {'mentions':>8} {'brand in read':>14} {'brand outside':>14}  best token")
n_mention = n_in = n_only_in = 0
for vid, desc in got.items():
    if not desc or vid not in text:
        continue
    low = desc.lower()
    mentions = any(k in low for k in ("sponsor", "promo code", "use code", "% off", "discount", "partner"))
    inside = " ".join(t for _, l, t in text[vid] if l).lower()
    outside = " ".join(t for _, l, t in text[vid] if not l).lower()
    best, best_in, best_out = "", 0, 0
    for tok in brand_tokens(desc):
        if len(tok) < 4:
            continue
        ci, co = inside.count(tok), outside.count(tok)
        if ci > best_in or (ci == best_in and co < best_out):
            best, best_in, best_out = tok, ci, co
    n_mention += mentions; n_in += best_in > 0; n_only_in += best_in > 0 and best_out == 0
    print(f"{vid:<12} {str(mentions):>8} {best_in:>14} {best_out:>14}  {best}")
print(f"\nsponsor words in description: {n_mention}; a description token inside the read: {n_in}; "
      f"inside and NOWHERE else: {n_only_in}")
