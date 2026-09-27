/**
 * The agent's instructions, the transcript it reads, and the checks its
 * answer has to pass before anything gets skipped. Pure functions: no I/O,
 * so the tests can drive them directly.
 */

export const SYSTEM_PROMPT = `You find the sponsored segments in a YouTube video by reading its entire transcript.

A SPONSOR segment is a paid promotion the creator reads for a third party: "this video is sponsored by", "thanks to X for sponsoring", "use code", "link in the description" for a product or service, a mid-roll ad read, a brand deal. It includes the lead-in sentence that introduces the sponsor and runs until the creator returns to the video's actual content.
A SELFPROMO segment is the creator promoting their own things: merch, Patreon or channel memberships, their own course, app or other channel. Not a subscribe or like reminder, and not a mention of a previous video.
Content that merely discusses a product as the video's actual topic (a review, a tutorial about that software) is NOT a sponsor segment.

Timestamps: each transcript line starts with [seconds]. A segment's "start" is the timestamp of its first line. Its "end" is the timestamp of the FIRST LINE AFTER the segment, the line where the real content resumes (or the video's length if it runs to the end). Be precise, since the player jumps from start straight to end.

Reply with ONLY a JSON object, no prose:
{"segments": [{"start": <seconds>, "end": <seconds>, "category": "sponsor" | "selfpromo", "quote": "<the first 6-12 words of the segment, copied exactly>", "resume_quote": "<the first 6-12 words right AFTER the segment, where the real content resumes, copied exactly; empty if the segment runs to the end>", "confidence": <0.0-1.0>}]}
If there are none, reply {"segments": []}.`;

/** Transcript → the numbered text the agent reads. Lines are merged into ~CHUNK_S chunks to save tokens. */
export const CHUNK_S = 6;
export function buildPrompt({ title, channel, lengthSeconds, transcript }) {
  const chunks = [];
  let current = null;
  for (const line of transcript) {
    const text = line.text.replace(/\s+/g, " ").trim();
    if (!text) continue;
    if (current && line.start - current.start < CHUNK_S) current.text += ` ${text}`;
    else chunks.push((current = { start: line.start, text }));
  }
  const header = [
    title ? `Title: ${title}` : null,
    channel ? `Channel: ${channel}` : null,
    lengthSeconds ? `Length: ${Math.round(lengthSeconds)} seconds` : null,
  ].filter(Boolean);
  return [...header, "", "Transcript:", ...chunks.map((c) => `[${Math.round(c.start)}] ${c.text}`)].join("\n");
}

/** First balanced {...} in the text, parsed; null if there is none. */
export function extractJson(text) {
  const cleaned = String(text).replace(/```(?:json)?/g, "");
  const start = cleaned.indexOf("{");
  if (start === -1) return null;
  let depth = 0;
  let inString = false;
  for (let i = start; i < cleaned.length; i++) {
    const c = cleaned[i];
    if (inString) {
      if (c === "\\") i++;
      else if (c === '"') inString = false;
    } else if (c === '"') inString = true;
    else if (c === "{") depth++;
    else if (c === "}" && --depth === 0) {
      try {
        return JSON.parse(cleaned.slice(start, i + 1));
      } catch {
        return null;
      }
    }
  }
  return null;
}

/**
 * The agent's reply → segments safe to skip. Drops anything malformed,
 * backwards, shorter than MIN_S or outside the video; clamps to the length;
 * merges overlaps; sorts. A reply that isn't JSON yields [] (skip nothing),
 * never a throw: a wrong answer here costs the viewer content.
 */
export const MIN_S = 3;
export function parseSegments(text, duration) {
  const parsed = extractJson(text);
  const raw = Array.isArray(parsed?.segments) ? parsed.segments : [];
  const limit = Number.isFinite(duration) && duration > 0 ? duration : Infinity;
  const clean = [];
  for (const s of raw) {
    const start = Number(s?.start);
    let end = Number(s?.end);
    if (!Number.isFinite(start) || !Number.isFinite(end)) continue;
    if (start < 0 || start >= limit) continue;
    end = Math.min(end, limit);
    if (end - start < MIN_S) continue;
    const category = s.category === "selfpromo" ? "selfpromo" : "sponsor";
    const confidence = Number.isFinite(Number(s.confidence)) ? Math.max(0, Math.min(1, Number(s.confidence))) : null;
    clean.push({
      start,
      end,
      category,
      quote: typeof s.quote === "string" ? s.quote.slice(0, 200) : null,
      resumeQuote: typeof s.resume_quote === "string" && s.resume_quote.trim() ? s.resume_quote.slice(0, 200) : null,
      confidence,
    });
  }
  clean.sort((a, b) => a.start - b.start);
  return mergeOverlaps(clean);
}

export function mergeOverlaps(clean) {
  const merged = [];
  for (const s of clean) {
    const last = merged.at(-1);
    if (last && s.start <= last.end && s.category === last.category) last.end = Math.max(last.end, s.end);
    else merged.push(s);
  }
  return merged;
}

const norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9' ]+/g, " ").replace(/\s+/g, " ").trim();

/** Start time of the caption line where `quote` begins, near `around`; null if not found. */
function lineOf(lines, quote, around, window) {
  const words = norm(quote ?? "").split(" ").filter(Boolean).slice(0, 4);
  if (words.length < 2) return null;
  const needle = words.join(" ");
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (l.start < around - window) continue;
    if (l.start > around + window) break;
    const joined = `${l.text} ${lines[i + 1]?.text ?? ""}`;
    const at = joined.indexOf(needle);
    // The quote must START in this line, not only in the next one.
    if (at !== -1 && at <= l.text.length) return l.start;
  }
  return null;
}

/**
 * Tighten each segment to exact caption lines. The agent reads ~6 s chunks,
 * so its start can be a chunk early and its end a chunk late. Its quotes pin
 * both: `quote` (the segment's first words) moves the start to the line where
 * the read begins, and `resumeQuote` (the first words after it) moves the end
 * to the line where the show resumes. Each bound moves at most `window`
 * seconds, and a segment never shrinks below MIN_S.
 *
 * An end the resume quote could not confirm is also capped at UNCONFIRMED_CAP_S
 * (Simon, 2026-09-26). Level 4's worst losses were runaway ends (575 s of show
 * for one read on holdout 5). A flat 180 s cap would clip ~5% of real ad time,
 * since 23 of 558 labelled reads run longer; on holdout 4 this rule changed no
 * real read (marker/level4_cap.py, README "Level 4's end cap"). If the cut would
 * leave END_SLIVER_S or less of the video, the skip runs to the end instead
 * (Simon, 2026-09-26): of 494 labelled videos, 26 end on a read, all <= 160 s,
 * so a capped skip that near the end is an outro, not the show.
 */
export const UNCONFIRMED_CAP_S = 180;
export const END_SLIVER_S = 30;
export function snapStarts(segments, transcript, window = 15, duration = null) {
  const lines = transcript.map((l) => ({ start: l.start, text: norm(l.text) }));
  return segments.map((seg) => {
    let { start, end } = seg;
    const s = lineOf(lines, seg.quote, seg.start, window);
    if (s !== null && s < end - MIN_S) start = s;
    const e = lineOf(lines, seg.resumeQuote, seg.end, window);
    if (e !== null && e > start + MIN_S) end = e;
    else if (end - start > UNCONFIRMED_CAP_S) {
      end = start + UNCONFIRMED_CAP_S;
      if (Number.isFinite(duration) && duration - end <= END_SLIVER_S) end = duration;
    }
    return { ...seg, start, end };
  });
}
