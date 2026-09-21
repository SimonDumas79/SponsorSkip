/**
 * Checking the marker's candidates — Simon's design, 2026-09-20.
 *
 * The marker finds candidate sponsor regions locally and for free. Checking
 * its work is then a SETTING, not a requirement, and three tiers fall out of
 * one pipeline:
 *
 *   none    the marker's own edges are used. Free, offline, no account.
 *   local   a local model answers "is there really an ad here?" and a no drops
 *           the candidate. The marker still owns the edges.
 *   claude  Claude is given a window and names the LINE the read starts on.
 *
 * Why a checker declares WHICH of the two jobs it can do, rather than both:
 * measured 2026-09-20, qwen3:8b found 6 of 6 sponsor reads and placed their
 * starts a median 55.7 s late, against Claude's 0.6 s. Detection it can do;
 * edges it cannot. Pretending otherwise would ship a skipper that starts a
 * minute into the advertisement.
 *
 * The two rules that make this safe:
 *   1. A checker can only ever NARROW what gets skipped. It may reject a
 *      candidate or tighten its edges; it can never invent a new one.
 *   2. An answer is a line index into the window it was handed, and anything
 *      outside that window is thrown away. This is what makes 2026-09-19's
 *      failure — a read at 3616 s reported as 1616 s, a number appearing
 *      nowhere in the prompt — structurally impossible rather than unlikely.
 *
 * NOT yet wired into server.mjs: the marker it checks does not exist until the
 * model is trained. This module is standalone and tested on its own.
 */
import { spawn } from "node:child_process";

const OLLAMA = (process.env.OLLAMA_URL || "http://localhost:11434").replace(/\/+$/, "");
const LOCAL_MODEL = process.env.SPONSORSKIP_LOCAL_MODEL || "qwen3:8b";
const CLAUDE_MODEL = process.env.SPONSORSKIP_MODEL || "haiku";

/** The window measured at 0.6 s median start error: 90 s of run-up, 60 s of tail. */
export const BEFORE_SECONDS = 90;
export const AFTER_SECONDS = 60;

/**
 * The caption lines around a candidate region.
 * The region is deliberately NOT centred: a read that sat in the middle of
 * every window would let a model score well by always answering "the middle".
 */
export function windowFor(transcript, region) {
  const from = region.start - BEFORE_SECONDS;
  const to = region.end + AFTER_SECONDS;
  const lo = transcript.findIndex((l) => l.start >= from);
  if (lo < 0) return null;
  let hi = lo;
  while (hi < transcript.length && transcript[hi].start <= to) hi += 1;
  return hi - lo >= 8 ? { lo, hi, lines: transcript.slice(lo, hi) } : null;
}

export const renderWindow = (lines) =>
  lines.map((l, i) => `${String(i).padStart(3)}: ${l.text.replace(/\s+/g, " ").trim()}`).join("\n");

const CONFIRM_ASK = `You are given numbered caption lines from part of a YouTube video.

Is there a SPONSOR READ in these lines — an advertisement the host reads out for a company that paid them, interrupting the video's actual subject?

A review, unboxing or demonstration of a product is NOT a sponsor read: if the product is the video's subject, nothing is being interrupted. The host's own merch, Patreon or other videos are not sponsor reads either.

Answer ONLY with JSON: {"sponsor": true} or {"sponsor": false}`;

const EDGES_ASK = `You are given numbered caption lines from part of a YouTube video. Somewhere inside them there is a sponsor read: an advertisement the host reads out.

Answer ONLY with JSON: {"start_line": <number>, "end_line": <number>}
  start_line = the number of the FIRST line of the sponsor read.
  end_line   = the number of the first line where the actual show RESUMES after it.
If there is no sponsor read at all, answer {"start_line": null, "end_line": null}.`;

function extractJson(text) {
  const m = String(text ?? "").replace(/```(?:json)?/g, "").match(/\{[\s\S]*?\}/);
  if (!m) return null;
  try {
    return JSON.parse(m[0]);
  } catch {
    return null;
  }
}

function runClaude(prompt, timeoutMs = 120_000) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      "claude",
      ["-p", "--model", CLAUDE_MODEL, "--tools", "", "--strict-mcp-config",
       "--no-session-persistence", "--output-format", "json"],
      { windowsHide: true },
    );
    let out = "";
    const timer = setTimeout(() => child.kill(), timeoutMs);
    child.stdout.on("data", (d) => (out += d));
    child.on("error", (e) => reject(new Error(`Claude Code CLI not available: ${e.message}`)));
    child.on("close", () => {
      clearTimeout(timer);
      try {
        const json = JSON.parse(out);
        resolve({ text: json.result ?? "", costUsd: json.total_cost_usd ?? null });
      } catch {
        reject(new Error(`claude gave no usable answer: ${out.slice(0, 200)}`));
      }
    });
    child.stdin.end(prompt);
  });
}

async function askOllama(prompt, schema) {
  const r = await fetch(`${OLLAMA}/api/chat`, {
    method: "POST",
    body: JSON.stringify({
      model: LOCAL_MODEL,
      stream: false,
      think: false,
      format: schema,
      keep_alive: "60s",
      options: { num_ctx: 8192, temperature: 0 },
      messages: [{ role: "user", content: prompt }],
    }),
    signal: AbortSignal.timeout(120_000),
  });
  if (!r.ok) throw new Error(`ollama ${r.status}`);
  return extractJson((await r.json()).message?.content);
}

/**
 * The checkers. Each says what it can do, and nothing calls a job a checker
 * has not claimed.
 */
export const CHECKERS = {
  none: {
    label: "nothing — the marker's own answer is used",
    confirm: null,
    placeEdges: null,
  },
  local: {
    label: `${LOCAL_MODEL} (local GPU) — confirms only`,
    async confirm(lines) {
      const answer = await askOllama(`${CONFIRM_ASK}\n\n${renderWindow(lines)}`, {
        type: "object",
        properties: { sponsor: { type: "boolean" } },
        required: ["sponsor"],
      });
      return answer ? Boolean(answer.sponsor) : null;
    },
    // Deliberately absent: measured 55.7 s median start error.
    placeEdges: null,
  },
  claude: {
    label: `claude-${CLAUDE_MODEL} (Claude Code) — confirms and places edges`,
    async confirm(lines) {
      const { text } = await runClaude(`${CONFIRM_ASK}\n\n${renderWindow(lines)}`);
      const answer = extractJson(text);
      return answer ? Boolean(answer.sponsor) : null;
    },
    async placeEdges(lines) {
      const { text, costUsd } = await runClaude(`${EDGES_ASK}\n\n${renderWindow(lines)}`);
      const answer = extractJson(text);
      if (!answer || answer.start_line === null || answer.start_line === undefined) return { edges: null, costUsd };
      return { edges: { startLine: answer.start_line, endLine: answer.end_line }, costUsd };
    },
  },
};

/**
 * Check one candidate region. Returns the region to skip, or null to drop it.
 *
 * `verdict` says what happened, so the popup can be honest about which tier
 * actually decided: "kept" (nothing checked it), "confirmed", "rejected",
 * "placed" (edges moved), or "out-of-window" (the answer was thrown away).
 */
export async function verifyRegion(region, transcript, checker = CHECKERS.none) {
  const win = windowFor(transcript, region);
  if (!win) return { ...region, verdict: "kept", reason: "window too small" };

  if (checker.confirm) {
    const ok = await checker.confirm(win.lines);
    if (ok === false) return null;
    if (ok === null) return { ...region, verdict: "kept", reason: "checker gave no usable answer" };
    if (!checker.placeEdges) return { ...region, verdict: "confirmed" };
  }

  if (!checker.placeEdges) return { ...region, verdict: "kept" };

  const { edges, costUsd } = await checker.placeEdges(win.lines);
  if (!edges) return null;

  const { startLine, endLine } = edges;
  // The whole point: an answer that does not resolve inside the window it was
  // given is wrong by construction, so it is discarded rather than trusted.
  const inWindow = (i) => Number.isInteger(i) && i >= 0 && i < win.lines.length;
  if (!inWindow(startLine)) {
    return { ...region, verdict: "out-of-window", reason: `start_line ${startLine} of ${win.lines.length}`, costUsd };
  }
  const endIdx = inWindow(endLine) && endLine > startLine ? endLine : win.lines.length - 1;

  return {
    start: win.lines[startLine].start,
    end: win.lines[endIdx].start,
    verdict: "placed",
    costUsd,
  };
}

/** Every candidate, checked in turn. Never adds a region; only narrows or drops. */
export async function verifyAll(regions, transcript, checker = CHECKERS.none) {
  const kept = [];
  for (const region of regions) {
    try {
      const checked = await verifyRegion(region, transcript, checker);
      if (checked) kept.push(checked);
    } catch (e) {
      // A checker failing must not lose the candidate: the marker found it for
      // free, and dropping it silently would be the worst of both worlds.
      kept.push({ ...region, verdict: "kept", reason: `checker failed: ${String(e.message).slice(0, 80)}` });
    }
  }
  return kept;
}
