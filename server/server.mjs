#!/usr/bin/env node
/**
 * SponsorSkip backend: an agent reads a YouTube video's whole transcript and
 * returns the sponsor segments to skip.
 *
 * Listens on 127.0.0.1 only (default port 4790). Zero npm dependencies; needs
 * yt-dlp (`python -m pip install --user yt-dlp`) for captions. Agents
 * (agents.mjs): Claude through the Claude Code CLI on Simon's subscription,
 * and the local GPU model through Ollama. No API keys. Claude reads first by
 * default; `?reader=local` puts the GPU first (see readWithAgents for why
 * that isn't the default).
 * Results are cached per video in ./cache, so a rewatch costs nothing.
 *
 *   GET /health             which agents can run right now
 *   GET /quick/:videoId     instant: the cached result, else SponsorBlock's
 *                           community segments (hash-prefix lookup), marked
 *                           interim, so an early sponsor read is covered
 *                           while the agent works
 *   GET /analyze/:videoId   the agent's result (cached, or computed now).
 *                           ?reader=claude|local  ?fresh=1 ignores the cache.
 *                           No English captions → SponsorBlock.
 */
import crypto from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { AGENTS, claudeAvailable, localBlocker, readClaude, readLocal } from "./agents.mjs";
import { parseSegments, snapStarts } from "./segments.mjs";
import { getTranscript } from "./youtube.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const PORT = Number(process.env.SPONSORSKIP_PORT) || 4790;
const CACHE = path.join(root, "cache");
const VIDEO_ID = /^[\w-]{11}$/;
fs.mkdirSync(CACHE, { recursive: true });

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
function analyze(id, { reader = "claude", fresh = false } = {}) {
  const cached = fresh ? null : readCache(id);
  if (cached) return Promise.resolve(cached);
  if (inFlight.has(id)) return inFlight.get(id);
  const job = (async () => {
    const started = Date.now();
    let video;
    try {
      video = await getTranscript(id);
    } catch (e) {
      // Couldn't fetch captions (rate limit, network, a YouTube change): fall
      // back to SponsorBlock for now, and DON'T cache, so the next visit retries.
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
    return result;
  })().finally(() => inFlight.delete(id));
  inFlight.set(id, job);
  return job;
}

/**
 * Which agent reads, and in what order. Default "claude": Claude reads; the
 * local GPU model only if Claude can't run. "local": the GPU reads first, and
 * Claude re-reads when the local answer disagrees with SponsorBlock.
 *
 * Why Claude is the default (measured 2026-09-18 on two Dwarkesh Patel
 * episodes, 6 sponsor reads between them): Claude Haiku found all 6, with
 * starts within 1-2 s. qwen3:8b found 2 of 6 whole in ~30k-character parts,
 * and with ~12k parts it found only the closing call-to-action lines, leaving
 * 40-60 s of each ad playing. Simon picked Claude-first on that evidence.
 */
async function readWithAgents(id, video, reader) {
  const tried = [];
  const clean = (text) => snapStarts(parseSegments(text, video.lengthSeconds), video.transcript);
  const base = { videoId: id, title: video.title, channel: video.channel, transcriptLines: video.transcript.length, reader };

  async function viaLocal() {
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
  async function viaClaude() {
    try {
      const t0 = Date.now();
      const { text, costUsd } = await readClaude(video, root);
      const segments = clean(text);
      tried.push({ agent: AGENTS.claude, outcome: `${segments.length} segment(s)`, seconds: Math.round((Date.now() - t0) / 1000), costUsd });
      return segments;
    } catch (e) {
      tried.push({ agent: AGENTS.claude, outcome: `failed: ${e.message}` });
      return null;
    }
  }
  // A reading "agrees" with SponsorBlock when every community segment has one of ours within 60 s.
  const agrees = (ours, community) => community.every((c) => ours.some((s) => Math.abs(s.start - c.start) < 60));

  if (reader === "local") {
    const local = await viaLocal();
    if (local) {
      const community = await sponsorBlock(id).catch(() => []);
      if (agrees(local, community)) return { ...base, segments: local, source: AGENTS.local, tried };
      tried.push({ agent: "sponsorblock", outcome: `local missed ${community.filter((c) => !local.some((s) => Math.abs(s.start - c.start) < 60)).length} community segment(s); asking Claude` });
    }
    const claude = await viaClaude();
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
      });
    }
    const m = /^\/(quick|analyze)\/([\w-]+)$/.exec(url.pathname);
    if (!m) return send(res, 404, { error: "not found" });
    const [, route, id] = m;
    if (!VIDEO_ID.test(id)) return send(res, 400, { error: "bad video id" });
    if (route === "quick") {
      const cached = readCache(id);
      if (cached) return send(res, 200, cached);
      const segments = await sponsorBlock(id).catch(() => []);
      return send(res, 200, { videoId: id, segments, source: "sponsorblock", interim: true });
    }
    const reader = url.searchParams.get("reader") === "local" ? "local" : "claude";
    return send(res, 200, await analyze(id, { reader, fresh: url.searchParams.get("fresh") === "1" }));
  } catch (error) {
    send(res, 500, { error: error.message });
  }
});

server.listen(PORT, "127.0.0.1", () => console.log(`SponsorSkip backend on http://127.0.0.1:${PORT} (readers: ${AGENTS.claude}; ${AGENTS.local})`));
