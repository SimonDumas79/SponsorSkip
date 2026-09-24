#!/usr/bin/env node
/**
 * SponsorSkip backend: an agent reads a YouTube video's whole transcript and
 * returns the sponsor segments to skip.
 *
 * Listens on 127.0.0.1 only (default port 4790). Zero npm dependencies; needs
 * yt-dlp (`python -m pip install --user yt-dlp`) for captions. Agents
 * (agents.mjs): Claude through the Claude Code CLI on the user's subscription,
 * and the local GPU model through Ollama. No API keys. Claude reads first by
 * default; `?reader=local` puts the GPU first (see readWithAgents for why
 * that isn't the default).
 * Results are cached per video in ./cache, so a rewatch costs nothing, and
 * every reading and every failure is appended to ./backend.log.
 *
 *   GET /health             which agents can run right now, and whether the
 *                           level-1 model files are in (models.mjs)
 *   GET /quick/:videoId     instant: the cached result, else SponsorBlock's
 *                           community segments (hash-prefix lookup), marked
 *                           interim, so an early sponsor read is covered
 *                           while the agent works
 *   GET /analyze/:videoId   the agent's result (cached, or computed now).
 *                           ?reader=claude|local  ?fresh=1 ignores the cache.
 *                           No English captions → SponsorBlock.
 *   GET /progress/:videoId  what the reader has found SO FAR, while /analyze
 *                           is still running. Claude reads the transcript in
 *                           overlapping parts, so a sponsor read in the
 *                           opening minute can be skipped long before the
 *                           whole video has been read.
 */
import crypto from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { AGENTS, claudeAvailable, localBlocker, readClaude, readLocal, readMarker } from "./agents.mjs";
import { ensureModels, modelStatus } from "./models.mjs";
import { mergeOverlaps, parseSegments, snapStarts } from "./segments.mjs";
import { getTranscript } from "./youtube.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const PORT = Number(process.env.SPONSORSKIP_PORT) || 4790;
const CACHE = path.join(root, "cache");
const VIDEO_ID = /^[\w-]{11}$/;
fs.mkdirSync(CACHE, { recursive: true });

// One timestamped line per reading and per failure, on stdout, which the
// launcher appends to backend.log. Without it a transient failure (a caption
// fetch that gets rate-limited, say) vanishes with the popup that showed it,
// and the next session can only guess at what went wrong.
const log = (...parts) => console.log([new Date().toISOString(), ...parts.filter(Boolean)].join(" "));

const cachePath = (id) => path.join(CACHE, `${id}.json`);
function readCache(id) {
  try {
    return JSON.parse(fs.readFileSync(cachePath(id), "utf8"));
  } catch {
    return null;
  }
}

async function sponsorBlock(id) {
  // Hash-prefix lookup: the server sees 4 hex characters, never the video id.
  const prefix = crypto.createHash("sha256").update(id).digest("hex").slice(0, 4);
  const r = await fetch(`https://sponsor.ajay.app/api/skipSegments/${prefix}?categories=${encodeURIComponent('["sponsor","selfpromo"]')}`, {
    signal: AbortSignal.timeout(8000),
  });
  if (r.status === 404) return [];
  if (!r.ok) throw new Error(`sponsorblock ${r.status}`);
  const match = (await r.json()).find((v) => v.videoID === id);
  return (match?.segments ?? []).map((s) => ({ start: s.segment[0], end: s.segment[1], category: s.category, quote: null, confidence: null }));
}

const inFlight = new Map();
// What a reading has found so far, while it is still going: videoId -> what
// /progress answers. Only ever a partial answer; the cache holds the finished
// one. Dropped as soon as the job settles.
const partial = new Map();

// Which step a reading is on, for the popup's progress strip. The point is
// that "found nothing" and "it broke" must never look the same: the steps run
// to a named end either way.
const STEPS = {
  captions: "Fetching captions",
  opening: "Reading the opening",
  full: "Reading the whole transcript",
  gpu: "Reading on your GPU",
  verify: "Claude checking that answer",
  marker: "Reading with the free marker (no language model)",
  markerQwen: "The marker finds, your GPU checks each find",
  markerCandidate: "Reading with six detectors and the fine-tuned model",
  markerCascade: "Found the reads, now placing their edges exactly",
  markerV3: "Reading with seven detectors (v3)",
  markerV3Cascade: "v3 found the reads, Claude is placing their edges",
};
function stage(id, step, extra = {}) {
  const was = partial.get(id) ?? { segments: [] };
  partial.set(id, { ...was, ...extra, step, label: STEPS[step] ?? step });
}
function analyze(id, { reader = "marker-v3", fresh = false } = {}) {
  // A reading is cached per video, but the popup lets you switch reader: a cached reading by a
  // different reader is not the answer that was asked for, so read again. (Old cache files and
  // SponsorBlock fallbacks carry no reader and are kept.)
  let cached = fresh ? null : readCache(id);
  if (cached?.reader && cached.reader !== reader) cached = null;
  if (cached) {
    log(id, "cached", `${cached.segments?.length ?? 0} segment(s)`);
    return Promise.resolve(cached);
  }
  if (inFlight.has(id)) return inFlight.get(id);
  const job = (async () => {
    const started = Date.now();
    log(id, `reading (reader ${reader}${fresh ? ", fresh" : ""})`);
    stage(id, "captions");
    let video;
    try {
      video = await getTranscript(id);
    } catch (e) {
      // Couldn't fetch captions (rate limit, network, a YouTube change): fall
      // back to SponsorBlock for now, and DON'T cache, so the next visit retries.
      log(id, "FAILED captions:", String(e.message).slice(0, 200), "- serving SponsorBlock, not cached, retries next visit");
      return { videoId: id, segments: await sponsorBlock(id).catch(() => []), source: "sponsorblock", reason: `couldn't fetch captions (${String(e.message).slice(0, 120)})`, retryLater: true };
    }
    let result;
    if (!video.transcript) {
      result = { videoId: id, title: video.title, segments: await sponsorBlock(id), source: "sponsorblock", reason: "no English captions" };
    } else {
      result = await readWithAgents(id, video, reader);
    }
    result.analyzedAt = new Date().toISOString();
    result.seconds = Math.round((Date.now() - started) / 1000);
    fs.writeFileSync(cachePath(id), JSON.stringify(result, null, 2));
    log(
      id,
      `${result.source} ${result.segments.length} segment(s) in ${result.seconds}s`,
      result.reason && `- ${result.reason}`,
      result.channel && `| ${result.channel}: ${String(result.title ?? "").slice(0, 60)}`,
    );
    for (const t of result.tried ?? []) {
      log(id, ` tried ${t.agent}: ${t.outcome}`, t.seconds && `(${t.seconds}s)`, t.costUsd && `$${t.costUsd}`);
    }
    return result;
  })().finally(() => {
    inFlight.delete(id);
    partial.delete(id);
  });
  inFlight.set(id, job);
  return job;
}

/**
 * Which agent reads, and in what order. Reader "claude": Claude reads; the
 * local GPU model only if Claude can't run. "local": the GPU reads first, and
 * Claude re-reads when the local answer disagrees with SponsorBlock.
 *
 * Why Claude is the default (measured 2026-09-18 on two Dwarkesh Patel
 * episodes, 6 sponsor reads between them): Claude Haiku found all 6, with
 * starts within 1-2 s. qwen3:8b found 2 of 6 whole in ~30k-character parts,
 * and with ~12k parts it found only the closing call-to-action lines, leaving
 * 40-60 s of each ad playing. Claude-first was picked on that evidence. Since
 * 2026-09-24 the free marker (level 1) is the program's default and every
 * reader here is opt-in, because each one costs the user GPU time or Claude usage.
 */
async function readWithAgents(id, video, reader) {
  const tried = [];
  // Merge again after snapping: snapStarts moves edges onto caption lines, so
  // two near-identical readings of one sponsor read (the opening pass and the
  // full read) can end up overlapping only once their edges have been snapped.
  const clean = (text) => mergeOverlaps(snapStarts(parseSegments(text, video.lengthSeconds), video.transcript));
  const base = { videoId: id, title: video.title, channel: video.channel, transcriptLines: video.transcript.length, reader };

  async function viaLocal() {
    stage(id, "gpu");
    const blocker = await localBlocker();
    if (blocker) {
      tried.push({ agent: AGENTS.local, outcome: `skipped: ${blocker}` });
      return null;
    }
    try {
      const t0 = Date.now();
      const { text, parts } = await readLocal(video);
      const segments = clean(text);
      tried.push({ agent: AGENTS.local, outcome: `${segments.length} segment(s) from ${parts} part(s)`, seconds: Math.round((Date.now() - t0) / 1000) });
      return segments;
    } catch (e) {
      tried.push({ agent: AGENTS.local, outcome: `failed: ${e.message}` });
      return null;
    }
  }
  async function viaClaude(firstStep = "opening") {
    try {
      const t0 = Date.now();
      stage(id, firstStep);
      // Each part is published as it lands, so the page can start skipping
      // what has been found while the rest is still being read.
      const { text, costUsd, parts, failed } = await readClaude(video, root, (p) => {
        const soFar = clean(p.text);
        stage(id, "full", { segments: soFar, part: p.part, parts: p.parts, source: AGENTS.claude });
        log(id, `part ${p.part}/${p.parts}: ${soFar.length} segment(s) so far`);
      });
      const segments = clean(text);
      tried.push({
        agent: AGENTS.claude,
        outcome: `${segments.length} segment(s) from ${parts} part(s)${failed ? `; ${failed}` : ""}`,
        seconds: Math.round((Date.now() - t0) / 1000),
        costUsd,
      });
      return segments;
    } catch (e) {
      tried.push({ agent: AGENTS.claude, outcome: `failed: ${e.message}` });
      return null;
    }
  }
  // The marker readers, opt-in. marker-qwen needs the GPU; when the GPU is busy or warm it runs as
  // the free tier instead, and says so. SponsorBlock only if the marker itself fails.
  if (reader === "marker" || reader === "marker-qwen" || reader === "marker-candidate" ||
      reader === "marker-cascade" || reader === "marker-v3" || reader === "marker-v3-cascade") {
    let tier = reader === "marker-qwen" ? "qwen" : reader === "marker-candidate" ? "candidate"
      : reader === "marker-cascade" ? "cascade" : reader === "marker-v3" ? "v3"
      : reader === "marker-v3-cascade" ? "v3-cascade" : "free";
    if (tier === "qwen") {
      const blocker = await localBlocker();
      if (blocker) {
        tried.push({ agent: AGENTS["marker-qwen"], outcome: `skipped: ${blocker}; running the free marker instead` });
        tier = "free";
      }
    }
    const agent = AGENTS[`marker-${tier}`] ?? (tier === "qwen" ? AGENTS["marker-qwen"] : AGENTS.marker);
    try {
      const t0 = Date.now();
      stage(id, tier === "qwen" ? "markerQwen" : tier === "candidate" ? "markerCandidate"
        : tier === "v3" ? "markerV3" : tier === "v3-cascade" ? "markerV3Cascade"
        : tier.startsWith("cascade") ? "markerCascade" : "marker");
      const segments = await readMarker(id, video, root, { tier });
      tried.push({ agent, outcome: `${segments.length} segment(s)`, seconds: Math.round((Date.now() - t0) / 1000) });
      return { ...base, segments, source: agent, tried };
    } catch (e) {
      tried.push({ agent, outcome: `failed: ${e.message}` });
      return { ...base, segments: await sponsorBlock(id).catch(() => []), source: "sponsorblock", reason: "the marker could not read it", tried };
    }
  }

  // A reading "agrees" with SponsorBlock when every community segment has one of ours within 60 s.
  const agrees = (ours, community) => community.every((c) => ours.some((s) => Math.abs(s.start - c.start) < 60));

  if (reader === "local") {
    const local = await viaLocal();
    // The setting promises "Claude checks it", so Claude has to actually
    // check. It used to be asked only when the local answer DISAGREED with
    // SponsorBlock -- and `[].every()` is true, so on a video SponsorBlock
    // doesn't cover, the local answer was accepted unverified. That is exactly
    // the video this extension exists for. An empty local answer was accepted
    // the same way, because `[]` is truthy. Both checked for here.
    if (local && local.length) {
      const community = await sponsorBlock(id).catch(() => []);
      if (community.length && agrees(local, community)) return { ...base, segments: local, source: AGENTS.local, tried };
      tried.push({
        agent: "sponsorblock",
        outcome: community.length
          ? `local missed ${community.filter((c) => !local.some((s) => Math.abs(s.start - c.start) < 60)).length} community segment(s); asking Claude`
          : "SponsorBlock has nothing to check against; asking Claude",
      });
    } else if (local) {
      tried.push({ agent: "sponsorblock", outcome: "the GPU found nothing, which is unchecked on its own; asking Claude" });
    }
    const claude = await viaClaude("verify");
    if (claude) return { ...base, segments: claude, source: AGENTS.claude, tried };
    if (local) return { ...base, segments: local, source: AGENTS.local, tried };
  } else {
    const claude = await viaClaude();
    if (claude) return { ...base, segments: claude, source: AGENTS.claude, tried };
    const local = await viaLocal();
    if (local) return { ...base, segments: local, source: AGENTS.local, tried };
  }
  return { ...base, segments: await sponsorBlock(id).catch(() => []), source: "sponsorblock", reason: "no agent could read it", tried };
}

// Browser extensions only. Firefox treats an extension's host permissions as
// opt-in, so its requests can arrive as ordinary cross-origin fetches; this
// answers them. Web pages get no CORS header, so no site can make the backend
// spend Claude usage (a page can't forge its Origin).
const EXTENSION_ORIGIN = /^(chrome-extension|moz-extension|extension):\/\/[\w-]+$/;
function send(res, status, body, type = "application/json") {
  const headers = { "content-type": type, "cache-control": "no-store" };
  // The origin rides on res, not a module variable: requests overlap (an
  // analyze can take a minute), so a shared variable would answer the wrong one.
  if (res.origin && EXTENSION_ORIGIN.test(res.origin)) {
    headers["access-control-allow-origin"] = res.origin;
    headers.vary = "Origin";
  }
  res.writeHead(status, headers);
  res.end(type === "application/json" ? JSON.stringify(body) : body);
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  res.origin = req.headers.origin ?? null;
  try {
    if (req.method !== "GET") return send(res, 405, { error: "GET only" });
    if (url.pathname === "/health") {
      const [blocker, claude] = await Promise.all([localBlocker(), claudeAvailable()]);
      return send(res, 200, {
        ok: true,
        agents: [
          { name: AGENTS.local, ready: !blocker, note: blocker ?? "ready" },
          { name: AGENTS.claude, ready: claude, note: claude ? "ready" : "Claude Code CLI not found (install it and log in)" },
        ],
        models: modelStatus(),
      });
    }
    const m = /^\/(quick|analyze|progress)\/([\w-]+)$/.exec(url.pathname);
    if (!m) return send(res, 404, { error: "not found" });
    const [, route, id] = m;
    if (!VIDEO_ID.test(id)) return send(res, 400, { error: "bad video id" });
    if (route === "progress") {
      // Cheap on purpose: a Map lookup or one cached file. The page polls this
      // every couple of seconds while a reading runs.
      const p = partial.get(id);
      if (p) return send(res, 200, { videoId: id, ...p, partial: true, done: false });
      // Only fall back to the cache when nothing is running: during a ?fresh=1
      // re-read the cached answer is the very thing being replaced.
      if (inFlight.has(id)) return send(res, 200, { videoId: id, segments: [], partial: true, done: false, waiting: true });
      const cached = readCache(id);
      return send(res, 200, cached ? { ...cached, done: true } : { videoId: id, segments: [], done: false, waiting: true });
    }
    if (route === "quick") {
      const cached = readCache(id);
      if (cached) return send(res, 200, cached);
      const segments = await sponsorBlock(id).catch(() => []);
      return send(res, 200, { videoId: id, segments, source: "sponsorblock", interim: true });
    }
    const asked = url.searchParams.get("reader");
    const reader = ["local", "marker", "marker-qwen", "marker-candidate",
                    "marker-cascade", "marker-v3", "marker-v3-cascade", "claude"].includes(asked) ? asked : "marker-v3";
    return send(res, 200, await analyze(id, { reader, fresh: url.searchParams.get("fresh") === "1" }));
  } catch (error) {
    log("FAILED", url.pathname, String(error.message).slice(0, 200));
    send(res, 500, { error: error.message });
  }
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`SponsorSkip backend on http://127.0.0.1:${PORT} (readers: ${AGENTS.claude}; ${AGENTS.local})`);
  ensureModels(root, log);
});
