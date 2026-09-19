#!/usr/bin/env node
/**
 * SponsorSkip backend: an agent reads a YouTube video's whole transcript and
 * returns the sponsor segments to skip.
 *
 * Listens on 127.0.0.1 only (default port 4790). Zero npm dependencies; needs
 * yt-dlp (`python -m pip install --user yt-dlp`) for captions and the Claude
 * Code CLI for the agent. The agent is Claude Haiku via `claude -p`, on
 * Simon's own login, with no tools, no MCP servers and none of his hooks.
 * Results are cached per video in ./cache, so a rewatch costs nothing.
 *
 *   GET /health
 *   GET /quick/:videoId     instant: the cached result, else SponsorBlock's
 *                           community segments (hash-prefix lookup), marked
 *                           interim, so an early sponsor read is covered
 *                           while the agent works
 *   GET /analyze/:videoId   the agent's result (cached, or computed now,
 *                           ~5-40 s). No English captions → SponsorBlock.
 *   GET /sponsorskip.user.js  the userscript, for one-click install
 */
import { spawn } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildPrompt, parseSegments, snapStarts, SYSTEM_PROMPT } from "./segments.mjs";
import { getTranscript } from "./youtube.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const PORT = Number(process.env.SPONSORSKIP_PORT) || 4790;
const CACHE = path.join(root, "cache");
const MODEL = process.env.SPONSORSKIP_MODEL || "haiku";
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

/** One agent call: the transcript in, raw text out. */
function runClaude(prompt) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      "claude",
      [
        "-p",
        "--model", MODEL,
        "--tools", "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--setting-sources", "project",
        "--output-format", "json",
        "--system-prompt", SYSTEM_PROMPT,
      ],
      // No shell: claude is a native .exe, and a shell would have to re-quote
      // the multi-line system prompt.
      { cwd: root, windowsHide: true },
    );
    let out = "";
    let err = "";
    const timer = setTimeout(() => child.kill(), 180_000);
    child.stdout.on("data", (d) => (out += d));
    child.stderr.on("data", (d) => (err += d));
    child.on("error", reject);
    child.on("close", (code) => {
      clearTimeout(timer);
      try {
        const json = JSON.parse(out);
        if (json.is_error) return reject(new Error(`claude: ${json.result ?? "error"}`));
        resolve({ text: json.result ?? "", costUsd: json.total_cost_usd ?? null });
      } catch {
        reject(new Error(`claude exited ${code}: ${(err || out).slice(0, 300)}`));
      }
    });
    child.stdin.end(prompt);
  });
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
function analyze(id) {
  const cached = readCache(id);
  if (cached) return Promise.resolve(cached);
  if (inFlight.has(id)) return inFlight.get(id);
  const job = (async () => {
    const started = Date.now();
    const video = await getTranscript(id);
    let result;
    if (!video.transcript) {
      result = { videoId: id, title: video.title, segments: await sponsorBlock(id), source: "sponsorblock", reason: "no English captions" };
    } else {
      const { text, costUsd } = await runClaude(buildPrompt(video));
      const segments = snapStarts(parseSegments(text, video.lengthSeconds), video.transcript);
      result = {
        videoId: id,
        title: video.title,
        channel: video.channel,
        segments,
        source: `claude-${MODEL}`,
        transcriptLines: video.transcript.length,
        costUsd,
      };
    }
    result.analyzedAt = new Date().toISOString();
    result.seconds = Math.round((Date.now() - started) / 1000);
    fs.writeFileSync(cachePath(id), JSON.stringify(result, null, 2));
    return result;
  })().finally(() => inFlight.delete(id));
  inFlight.set(id, job);
  return job;
}

function send(res, status, body, type = "application/json") {
  res.writeHead(status, { "content-type": type, "cache-control": "no-store" });
  res.end(type === "application/json" ? JSON.stringify(body) : body);
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  try {
    if (req.method !== "GET") return send(res, 405, { error: "GET only" });
    if (url.pathname === "/health") return send(res, 200, { ok: true, model: MODEL });
    if (url.pathname === "/sponsorskip.user.js") {
      return send(res, 200, fs.readFileSync(path.join(root, "userscript", "sponsorskip.user.js"), "utf8"), "text/javascript");
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
    return send(res, 200, await analyze(id));
  } catch (error) {
    send(res, 500, { error: error.message });
  }
});

server.listen(PORT, "127.0.0.1", () => console.log(`SponsorSkip backend on http://127.0.0.1:${PORT} (agent: claude ${MODEL})`));
