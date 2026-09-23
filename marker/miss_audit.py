"""Diagnose which sponsor/self-promo reads candidate v3 fails on. (2026-09-23)

Read-only: no model, bundle or production file is changed here. Primary diagnosis is honest
out-of-fold: candidate v3's context model (stack_sbml.py / export_candidate.py --add-v3) is
trained and graded ONLY on the 160 pooled videos whose SponsorBlock labels postdate the
community model's training (sb_dates.json, cutoff 2022-04-01). Its saved out-of-fold scores
(data/stack_sbml_oof_s0.npy, seed 0, index 1 = "+ SponsorBlock model") are reused as-is -- no
retraining -- together with the pooled rule's B=10 threshold, recomputed the same way
export_candidate.py --add-v3 recomputes it (stack_check.sweep_fine + detector_bakeoff.pick).
Regions are placed with edge_heads.place using the v2 averaged edge heads (MLP OOF + fine-tuned
OOF, restricted to the same 160 videos), exactly candidate v3's own placement.

A second, clearly-labelled table repeats the same read-level breakdown on holdout 4 (64 fresh
videos, already graded once), scored through the saved marker/data/production/candidate.pt
bundle's v3 (the same path export_candidate.py --verify-v3 holdout4 checks).

Writes marker/data/miss_audit.md and marker/data/miss_audit.csv. Nothing else.

    python marker/miss_audit.py
"""

import ctypes
import datetime
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from build_production import SETS, pooled
from detector_bakeoff import graded, pick
from edge_heads import place
from experiments import runs
from replay import regions_by_video
from stack_check import sweep_fine
from train import DATA, LAST_LINE_SECONDS, Rows, load

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(8)

OUT_MD = DATA / "miss_audit.md"
OUT_CSV = DATA / "miss_audit.csv"
CUTOFF = datetime.datetime(2022, 4, 1).timestamp() * 1000

# ----------------------------------------------------------------------------- type thresholds
VERY_SHORT_S = 20.0
EDGE_FRAC = 0.08          # within this fraction of the video's start/end
CHOPPED_MIN_LINES = 10
CHOPPED_MAX_SEC_PER_LINE = 2.0
VIDEO_IS_AD_SHARE = 0.5   # the read covers more than this share of the whole video
REPEAT_CHANNEL_MIN = 3    # a channel with this many missed/mostly-missed reads
MOSTLY_MISSED = 0.5       # < 50% of ad seconds skipped


def load_field(field: str) -> np.ndarray:
    """Concatenate a raw npz field across the pooled sets, same order as build_production.pooled()."""
    return np.concatenate([np.load(DATA / f)[field] for f in SETS])


def unseen_mask(video: np.ndarray) -> np.ndarray:
    d = json.load(open(DATA / "sb_dates.json", encoding="utf-8"))
    return np.array([not (d.get(str(v), {}).get("first") and d[str(v)]["first"] < CUTOFF) for v in video])


def load_caption_meta(cap_dir: Path) -> dict:
    meta = {}
    if not cap_dir.exists():
        return meta
    for fp in cap_dir.glob("*.json"):
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        vid = d.get("videoID") or fp.stem
        meta[vid] = {"channel_name": d.get("channel"), "duration": d.get("duration"),
                     "caption_language": d.get("language")}
    return meta


def build_text_index(video_ids: set, files: list) -> dict:
    idx = {}
    for fname in files:
        path = DATA / fname
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                if r["videoID"] in video_ids:
                    idx[(r["videoID"], int(r["i"]))] = r["text"]
    return idx


def cue_columns(feature_names) -> list:
    names = [str(n) for n in feature_names]
    return [i for i, n in enumerate(names) if n.startswith("cue_")]


def fraction_column(feature_names) -> int | None:
    names = [str(n) for n in feature_names]
    return names.index("fraction_in") if "fraction_in" in names else None


def build_reads(rows, kept: dict, oof_scores: np.ndarray, category_arr: np.ndarray, line_arr: np.ndarray,
                text_index: dict, cap_meta: dict, lang_map: dict, dataset: str) -> list:
    cue_cols = cue_columns(rows.feature_names)
    frac_col = fraction_column(rows.feature_names)
    out = []
    for vid in np.unique(rows.video):
        vid_s = str(vid)
        r = np.flatnonzero(rows.video == vid)
        y_v = rows.y[r]
        s = rows.start_seconds[r]
        on_screen = np.diff(np.append(s, s[-1] + LAST_LINE_SECONDS))
        meta = cap_meta.get(vid_s, {})
        video_duration = meta.get("duration") or float(s[-1] + LAST_LINE_SECONDS)
        channel_name = meta.get("channel_name")
        lang_info = lang_map.get(vid_s)
        flags = np.zeros(len(r), dtype=np.int8)
        for lo, hi in kept.get(vid_s, []):
            flags[lo:hi] = 1
        for a, b in runs(y_v):
            idx = r[a:b]
            duration = float(on_screen[a:b].sum())
            skipped = float(on_screen[a:b][flags[a:b] == 1].sum())
            share = skipped / duration if duration > 0 else 0.0
            cats = [c for c in category_arr[idx].tolist() if c]
            category = Counter(cats).most_common(1)[0][0] if cats else "unknown"
            max_score = float(oof_scores[idx].max())
            cue_fire = bool(rows.X[idx][:, cue_cols].sum(axis=1).max() > 0) if cue_cols else False
            num_lines = int(b - a)
            rhythm = duration / num_lines if num_lines else None
            if frac_col is not None:
                frac_pos = float(rows.X[idx[0], frac_col])
            else:
                frac_pos = float(s[a] / max(video_duration, 1e-9))
            texts = [text_index.get((vid_s, int(line_arr[i]))) for i in idx]
            text200 = " ".join(t for t in texts if t)[:200]
            translated = None
            caption_lang = meta.get("caption_language")
            if lang_info is not None:
                lv = lang_info.get("lang")
                translated = (lv is not None) and (lv != "en")
            elif caption_lang is not None:
                translated = caption_lang not in ("en", "en-US", "en-GB", "en-orig")
            bucket = "full" if share >= 0.95 else ("missed" if share == 0 else "partial")
            out.append({
                "dataset": dataset, "video": vid_s, "channel": str(rows.channel[idx[0]]),
                "channel_name": channel_name, "category": category,
                "duration_s": round(duration, 1), "video_duration_s": round(video_duration, 1),
                "position_frac": round(frac_pos, 3), "num_lines": num_lines,
                "sec_per_line": round(rhythm, 2) if rhythm is not None else None,
                "share_skipped": round(share, 3), "bucket": bucket,
                "max_v3_score": round(max_score, 4), "cue_fire": cue_fire,
                "translated": translated, "manual_captions": "unknown",
                "text_200": text200,
            })
    return out


def assign_types(read: dict, repeat_channels: set) -> list:
    tags = []
    if read["duration_s"] < VERY_SHORT_S:
        tags.append("very_short (<20s)")
    if read["category"] == "selfpromo":
        tags.append("selfpromo")
    if not read["cue_fire"]:
        tags.append("no_cue_words")
    if read["position_frac"] < EDGE_FRAC or read["position_frac"] > 1 - EDGE_FRAC:
        tags.append("at_video_edge")
    if read["translated"] is True:
        tags.append("translated_captions")
    if read["num_lines"] >= CHOPPED_MIN_LINES and read["sec_per_line"] is not None \
            and read["sec_per_line"] < CHOPPED_MAX_SEC_PER_LINE:
        tags.append("many_short_lines")
    if read["video_duration_s"] > 0 and read["duration_s"] / read["video_duration_s"] > VIDEO_IS_AD_SHARE:
        tags.append("video_is_the_ad")
    if read["channel"] in repeat_channels:
        tags.append("repeat_channel")
    if not tags:
        tags.append("other")
    return tags


def type_table(reads: list) -> tuple:
    """Type breakdown over the missed + mostly-missed group (<50% skipped)."""
    total_missed_all = sum(r["duration_s"] * (1 - r["share_skipped"]) for r in reads)
    group = [r for r in reads if r["share_skipped"] < MOSTLY_MISSED]
    channel_counts = Counter(r["channel"] for r in group)
    repeat_channels = {c for c, n in channel_counts.items() if n >= REPEAT_CHANNEL_MIN}
    for r in group:
        r["types"] = assign_types(r, repeat_channels)
    tally = {}
    for r in group:
        for t in r["types"]:
            tally.setdefault(t, {"count": 0, "missed_s": 0.0, "examples": []})
            tally[t]["count"] += 1
            tally[t]["missed_s"] += r["duration_s"] * (1 - r["share_skipped"])
            if len(tally[t]["examples"]) < 3 and r["text_200"]:
                tally[t]["examples"].append(r["text_200"])
    rows_out = []
    for t, info in tally.items():
        share = info["missed_s"] / total_missed_all if total_missed_all > 0 else 0.0
        rows_out.append((t, info["count"], info["missed_s"], share, info["examples"]))
    rows_out.sort(key=lambda x: -x[2])
    return rows_out, total_missed_all, group, repeat_channels


def fully_vs_trimmed(reads: list) -> dict:
    fully = sum(r["duration_s"] * (1 - r["share_skipped"]) for r in reads if r["share_skipped"] == 0)
    total = sum(r["duration_s"] * (1 - r["share_skipped"]) for r in reads)
    trimmed = total - fully
    return {"fully_missed_s": fully, "trimmed_s": trimmed, "total_missed_s": total}


SIGNALS = {
    "very_short (<20s)": "the region-smoothing / minimum-region-length rule likely eats these; a lower "
        "minimum run length or a dedicated short-read threshold could recover them",
    "selfpromo": "a feature for first-person merch/Patreon/course language and channel-owned-brand mentions, "
        "trained specifically on selfpromo positives rather than pooled with sponsor",
    "no_cue_words": "the cue regexes cannot help here by definition; only the semantic (meaning) detector or "
        "the community model's chunk-level text extraction can catch these -- worth checking if the BGE or "
        "community stream alone flags them even when the stack's threshold does not",
    "at_video_edge": "context_stack.py pads outside the video with a fixed low logit; a read starting at "
        "line 0 or ending at the last line has an artificially short context window on one side",
    "translated_captions": "the machine-translated track is noisier text; either widen training data with "
        "translated examples or fall back to a cheaper/looser detector (as scope_english.py already flags) "
        "for non-English videos",
    "many_short_lines": "smooth() only bridges gaps up to a small window; a read chopped into many short "
        "captions needs a larger bridge window or a rolling-max reach tuned for choppy cadence",
    "video_is_the_ad": "these look structurally different (near-constant high cue/topic density for most of "
        "the video); a whole-video classifier (sequence model) feature such as 'share of video flagged by "
        "cheap detectors' could catch it before the line-level context model ever runs",
    "repeat_channel": "a handful of channels with a consistent format the stack keeps missing; a per-channel "
        "calibration or a small labelled sample from exactly these channels would help more than more general data",
    "other": "no single candidate signal explains these; needs manual review of the text examples",
}


def cv_pipeline():
    rows, _, _ = pooled()
    category_all = load_field("category")
    line_all = load_field("line")
    mask = unseen_mask(rows.video)
    print(f"pooled: {rows.videos} videos; unseen (v3's own CV universe): {int(mask.sum())} rows' worth of "
          f"videos -> {len(set(rows.video[mask].tolist()))} videos", flush=True)

    U = Rows(rows.X[mask], rows.y[mask], rows.video[mask], rows.channel[mask], rows.start_seconds[mask],
              rows.split[mask], rows.feature_names)
    category_U = category_all[mask]
    line_U = line_all[mask]

    oof3 = np.load(DATA / "stack_sbml_oof_s0.npy")[1]
    assert len(oof3) == len(U), f"stack_sbml_oof_s0.npy length {len(oof3)} != unseen rows {len(U)}"

    level1, level2, p_start, p_end = np.load(DATA / "pooled_oof.npy")
    fs_oof = np.load(DATA / "finetune_oof_edge_start_seed0.npy")
    fe_oof = np.load(DATA / "finetune_oof_edge_resume_seed0.npy")
    heads_start = ((p_start + fs_oof) / 2)[mask]
    heads_end = ((p_end + fe_oof) / 2)[mask]

    g3 = graded(sweep_fine(oof3, U, heads_start, heads_end), U)
    picked = pick(g3, 10)
    if picked is None:
        raise SystemExit("no B=10 threshold met the pooled rule on the unseen set -- cannot diagnose")
    th10, kept10, grade10 = picked
    print(f"reproduced v3 CV @ B=10: threshold {th10:.4f}, ad time {grade10['coverage']:.1%}, "
          f"show lost {grade10['show']:.1f} s/video, videos: {U.videos}, reads: {U.reads}", flush=True)

    cap_meta = load_caption_meta(DATA / "captions_pooled_unseen")
    lang_map = json.load(open(DATA / "video_language.json", encoding="utf-8"))
    text_index = build_text_index(set(str(v) for v in U.video),
                                  ["examples.jsonl", "examples_tail.jsonl", "examples_holdout2.jsonl"])

    reads = build_reads(U, kept10, oof3, category_U, line_U, text_index, cap_meta, lang_map, "cv_unseen160")
    return reads, grade10


def holdout4_pipeline():
    from sbml_eval import load_preds
    from serve_candidate import Candidate
    from stack_sbml import sbml_stream
    OUT = DATA / "production" / "candidate.pt"
    if not OUT.exists():
        print("holdout4: no candidate.pt bundle found, skipping the secondary table", flush=True)
        return None, None
    T = load(DATA / "features_holdout4.npz")
    category_T = np.load(DATA / "features_holdout4.npz")["category"]
    line_T = np.load(DATA / "features_holdout4.npz")["line"]
    potT = np.load(DATA / "features_potion_holdout4.npz")["X"]
    potT = potT[:, :potT.shape[1] - 2]
    bge_T = np.load(DATA / "finetune_full_bge_seed0__holdout4.npy")
    fs_T = np.load(DATA / "finetune_full_edge_start_seed0__holdout4.npy")
    fe_T = np.load(DATA / "finetune_full_edge_resume_seed0__holdout4.npy")
    c = Candidate(OUT)
    if c.v3 is None:
        print("holdout4: bundle has no v3, skipping the secondary table", flush=True)
        return None, None
    st = {**c.streams(T, potT, bge_T), "sbml": sbml_stream(T, load_preds())}
    ctx3, ps, pe = c.score_v3(T, st)
    th = c.v3["thresholds"][10]
    kept = place(regions_by_video(ctx3, T, th, 1), T, (ps + fs_T) / 2, (pe + fe_T) / 2)
    from replay import grade_regions
    g = grade_regions(kept, T)
    print(f"reproduced v3 holdout4 @ B=10: threshold {th:.4f}, ad time {g['coverage']:.1%}, "
          f"show lost {g['show']:.1f} s/video (README records 67.2% / 5.5 s WITH creator chapters, "
          f"not applied here)", flush=True)

    cap_meta = load_caption_meta(DATA / "captions_holdout4")
    lang_map = json.load(open(DATA / "video_language.json", encoding="utf-8"))
    text_index = build_text_index(set(str(v) for v in T.video), ["examples_holdout4.jsonl"])
    reads = build_reads(T, kept, ctx3, category_T, line_T, text_index, cap_meta, lang_map, "holdout4")
    return reads, g


def write_csv(all_reads: list) -> None:
    import csv
    fields = ["dataset", "video", "channel", "channel_name", "category", "duration_s", "video_duration_s",
              "position_frac", "num_lines", "sec_per_line", "share_skipped", "bucket", "max_v3_score",
              "cue_fire", "translated", "manual_captions", "text_200"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_reads:
            w.writerow(r)


def md_type_table(rows_out: list) -> str:
    lines = ["| type | reads | missed ad-seconds | share of total missed | signal that might catch it |",
             "|---|---|---|---|---|"]
    for t, count, missed_s, share, _ in rows_out:
        lines.append(f"| {t} | {count} | {missed_s:.0f} s | {share:.1%} | {SIGNALS.get(t, '')} |")
    return "\n".join(lines)


def md_examples(rows_out: list) -> str:
    out = ["", "### Examples per type", ""]
    for t, count, missed_s, share, examples in rows_out:
        out.append(f"**{t}** ({count} reads, {share:.1%} of missed ad time)")
        for ex in examples:
            out.append(f"- \"{ex}\"")
        out.append("")
    return "\n".join(out)


def main() -> int:
    cv_reads, cv_grade = cv_pipeline()
    cv_types, cv_total_missed, cv_group, cv_repeat = type_table(cv_reads)
    cv_split = fully_vs_trimmed(cv_reads)

    h4_reads, h4_grade = holdout4_pipeline()
    all_reads = list(cv_reads) + (list(h4_reads) if h4_reads else [])
    write_csv(all_reads)

    md = []
    md.append("# candidate v3 miss audit (2026-09-23)")
    md.append("")
    md.append("Primary table: out-of-fold, the 160 pooled videos v3's own CV was chosen on "
              "(sb_dates.json cutoff 2022-04-01), B=10, v2 averaged edge heads, seed 0 -- "
              "the exact path export_candidate.py --add-v3 uses. Reproduced grade: "
              f"ad time {cv_grade['coverage']:.1%}, show lost {cv_grade['show']:.1f} s/video, "
              f"{cv_grade['found']}/{cv_grade['total']} reads touched, {len(cv_reads)} reads total.")
    md.append("")
    md.append(f"'Missed / mostly-missed' = share of the read's ad seconds skipped < {MOSTLY_MISSED:.0%}. "
              f"{len(cv_group)} of {len(cv_reads)} reads qualify. A read may match more than one type, "
              "so shares below do not sum to 100%.")
    md.append("")
    md.append("## Types, ranked by share of missed ad time (CV, primary)")
    md.append("")
    md.append(md_type_table(cv_types))
    md.append(md_examples(cv_types))
    md.append("## Fully missed vs trimmed edges (CV, primary, ALL reads)")
    md.append("")
    md.append(f"Total missed ad time: {cv_split['total_missed_s']:.0f} s over {len(cv_reads)} reads. "
              f"Fully missed (0% skipped): {cv_split['fully_missed_s']:.0f} s "
              f"({cv_split['fully_missed_s']/cv_split['total_missed_s']:.1%}). "
              f"Trimmed edges (partially skipped): {cv_split['trimmed_s']:.0f} s "
              f"({cv_split['trimmed_s']/cv_split['total_missed_s']:.1%}).")
    md.append("")

    if h4_reads:
        h4_types, h4_total_missed, h4_group, h4_repeat = type_table(h4_reads)
        h4_split = fully_vs_trimmed(h4_reads)
        md.append("## SECOND TABLE -- holdout 4 (already graded once; not the primary evidence)")
        md.append("")
        md.append(f"Reproduced grade: ad time {h4_grade['coverage']:.1%}, show lost {h4_grade['show']:.1f} "
                  f"s/video (no creator chapters applied here, so this reads lower than the README's "
                  f"67.2%/5.5s figure, which includes them), {len(h4_reads)} reads, "
                  f"{len(h4_group)} missed/mostly-missed.")
        md.append("")
        md.append(md_type_table(h4_types))
        md.append(md_examples(h4_types))
        md.append(f"Fully missed: {h4_split['fully_missed_s']:.0f} s "
                  f"({h4_split['fully_missed_s']/max(h4_split['total_missed_s'],1e-9):.1%}); "
                  f"trimmed: {h4_split['trimmed_s']:.0f} s "
                  f"({h4_split['trimmed_s']/max(h4_split['total_missed_s'],1e-9):.1%}).")
        md.append("")

    md.append("## Could not determine")
    md.append("")
    md.append("- Manual vs auto-generated captions: not recoverable from what is stored (captionFile is just "
              "a filename like `<id>.en.json3`; yt-dlp does not mark which of --write-subs/--write-auto-subs "
              "supplied it). Left as 'unknown' in the CSV.")
    md.append("- 'translated' relies on video_language.json's detected spoken language; a few videos not in "
              "that file fall back to the caption file's own recorded `language` field, which is only present "
              "on a minority of caption files fetched with the newer fetch_captions.py.")
    md.append("")
    md.append(f"Files written: `{OUT_MD}`, `{OUT_CSV}`.")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {OUT_MD}")
    print(f"wrote {OUT_CSV} ({len(all_reads)} reads)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
