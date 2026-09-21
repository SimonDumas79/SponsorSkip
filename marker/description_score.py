"""Label-free scoring of the description signal on the cached descriptions."""
import json, re
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
got = {k: v for k, v in json.loads((DATA / "descriptions_probe.json").read_text(encoding="utf-8")).items() if v}
text = {}
with (DATA / "examples.jsonl").open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["videoID"] in got:
            text.setdefault(r["videoID"], []).append((r["i"], r["label"], r["text"], r["start"]))

GENERIC = set("""youtube twitter instagram tiktok facebook discord patreon twitch reddit spotify apple google amazon bit
linktr linktree goo youtu the and for you your with this that from video what how new all free get our are one out use
more code link below click here check subscribe channel merch store shop www com http https http""".split())


def tokens(desc: str) -> set[str]:
    toks = set()
    for ln in desc.splitlines():
        low = ln.lower()
        for dom in re.findall(r"https?://(?:www\.)?([a-z0-9-]{4,})\.", low):
            toks.add(dom)
        for m in re.finditer(r"\bcode\b\s*[:\-]?\s*([A-Za-z0-9]{3,})", ln):          # "code SIMON"
            toks.add(m.group(1).lower())
        if "http" in low or "code" in low or "sponsor" in low or "% off" in low or "partner" in low:
            for w in re.findall(r"\b[A-Z][A-Za-z0-9]{3,}\b", ln):                     # Capitalised names on those lines
                toks.add(w.lower())
    return {t for t in toks if t not in GENERIC and not t.isdigit()}


n_reads = n_reads_hit = n_lines_hit = n_lines_hit_in = 0
videos_with_tokens = 0
print(f"{'video':<12} {'tokens':>6} {'hit lines':>9} {'in read':>8} {'reads':>5} {'reads hit':>9}  tokens that hit")
for vid, desc in got.items():
    toks = tokens(desc)
    rows = text[vid]
    hits = [(i, l, t) for i, l, t, _ in rows if any(tok in t.lower() for tok in toks)]
    labels = np.array([l for _, l, _, _ in rows])
    padded = np.concatenate(([0], labels, [0])); edges = np.flatnonzero(np.diff(padded))
    reads = list(zip(edges[::2], edges[1::2]))
    hit_idx = {i for i, _, _ in hits}
    reads_hit = sum(1 for a, b in reads if any(i in hit_idx for i in range(a, b)))
    which = sorted({tok for tok in toks if any(tok in t.lower() for _, _, t in hits)})
    n_reads += len(reads); n_reads_hit += reads_hit
    n_lines_hit += len(hits); n_lines_hit_in += sum(1 for _, l, _ in hits if l)
    videos_with_tokens += bool(toks)
    print(f"{vid:<12} {len(toks):>6} {len(hits):>9} {sum(1 for _, l, _ in hits if l):>8} {len(reads):>5} {reads_hit:>9}  {', '.join(which)[:60]}")
print(f"\n{len(got)} videos, {videos_with_tokens} with any token; reads with a hit line: {n_reads_hit}/{n_reads}; "
      f"hit lines inside a read: {n_lines_hit_in}/{n_lines_hit} ({n_lines_hit_in / max(n_lines_hit, 1):.0%} precision per line)")
