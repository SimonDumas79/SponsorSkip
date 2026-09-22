/**
 * The two agents that can read a transcript (the order is chosen in
 * server.mjs, readWithAgents: Claude first by default):
 *
 *   claude: Claude Haiku through the Claude Code CLI (`claude -p`), on
 *           Simon's subscription. No API key, no tools, no MCP servers, no hooks.
 *           Reads the opening on its own first, so the start of a video is
 *           covered within seconds, then reads the whole transcript.
 *   local:  qwen3:8b on Simon's GPU through Ollama. Free, but measured well
 *           below Claude on this task (2 of 6 reads found). The transcript is
 *           read in parts that fit its 16k context, the model is unloaded as
 *           soon as the video is done, it's skipped when the GPU is busy or
 *           warm, and it's abandoned mid-run if the GPU gets hot.
 *
 * And one reader that is not an agent at all (opt-in, ?reader=marker):
 *
 *   marker: the free tier trained in marker/ (marker/predict.py serve): a
 *           linear marker, a context model and start/resume heads, on the
 *           CPU, no language model. Graded on two holdouts of unseen channels
 *           at about half the ad time skipped for 7.5-9.9 s of real show lost
 *           per video (see marker/README.md).
 *   marker-qwen: the same marker, with qwen3 on the GPU confirming each
 *           region and placing its edges (predict.py serve --tier qwen).
 *           Graded at 63.4% of ad time on both holdouts, for 19.9 s and
 *           9.9 s of real show lost per video.
 */
import { execFile, spawn } from "node:child_process";
import os from "node:os";
import path from "node:path";
import { SYSTEM_PROMPT, buildPrompt, extractJson } from "./segments.mjs";

const OLLAMA = (process.env.OLLAMA_URL || "http://localhost:11434").replace(/\/+$/, "");
const LOCAL_MODEL = process.env.SPONSORSKIP_LOCAL_MODEL || "qwen3:8b";
const CLAUDE_MODEL = process.env.SPONSORSKIP_MODEL || "haiku";
const CHUNK_CHARS = Number(process.env.SPONSORSKIP_CHUNK_CHARS) || 30_000; // ~8k tokens per part, leaving room in a 16k context for thinking and the answer
// How much of the opening Claude reads on its own first, so a sponsor read at
// the start is skippable within seconds. Small on purpose: it is a head start,
// not the answer (see readClaude).
const HEAD_SECONDS = Number(process.env.SPONSORSKIP_HEAD_SECONDS) || 420;
const OVERLAP_S = 120; // parts overlap so a read that straddles a cut is seen whole at least once
const GPU_BUSY_MIB = 2500;
const GPU_WARM_C = 78;
const GPU_ABORT_C = 83;

// JSON schema Ollama constrains the answer to (the thinking stays free-form).
const SCHEMA = {
  type: "object",
  properties: {
    segments: {
      type: "array",
      items: {
        type: "object",
        properties: {
          start: { type: "number" },
          end: { type: "number" },
          category: { type: "string", enum: ["sponsor", "selfpromo"] },
          quote: { type: "string" },
          resume_quote: { type: "string" },
          confidence: { type: "number" },
        },
        required: ["start", "end", "category", "quote"],
      },
    },
  },
  required: ["segments"],
};

function execText(cmd, args, timeout = 10_000) {
  return new Promise((resolve) => execFile(cmd, args, { timeout, windowsHide: true }, (e, out) => resolve(e ? null : String(out).trim())));
}

export async function gpuState() {
  const out = await execText("nvidia-smi", ["--query-gpu=memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"]);
  if (!out) return null;
  const [memMiB, util, tempC] = out.split("\n")[0].split(",").map((x) => Number(x.trim()));
  return { memMiB, util, tempC };
}

/** Machine-wide CPU busy %, over one second (no pounding the CPU while other things run). */
async function cpuBusy(sampleMs = 1000) {
  const snap = () => os.cpus().reduce((a, c) => ((a.idle += c.times.idle), (a.total += c.times.user + c.times.nice + c.times.sys + c.times.idle + c.times.irq), a), { idle: 0, total: 0 });
  const a = snap();
  await new Promise((r) => setTimeout(r, sampleMs));
  const b = snap();
  return b.total - a.total > 0 ? Math.round(100 * (1 - (b.idle - a.idle) / (b.total - a.total))) : 0;
}

/** Why the local agent shouldn't run right now, or null if it can. */
export async function localBlocker() {
  let tags;
  try {
    tags = await (await fetch(`${OLLAMA}/api/tags`, { signal: AbortSignal.timeout(3000) })).json();
  } catch {
    return "Ollama is not running";
  }
  if (!(tags.models ?? []).some((m) => m.name === LOCAL_MODEL)) return `${LOCAL_MODEL} is not installed in Ollama`;
  const g = await gpuState();
  if (g && g.memMiB > GPU_BUSY_MIB) return `the GPU is in use (${g.memMiB} MiB)`;
  if (g && g.tempC >= GPU_WARM_C) return `the GPU is warm (${g.tempC} °C)`;
  const cpu = await cpuBusy();
  if (cpu > 60) return `the CPU is busy with something else (${cpu}%)`;
  return null;
}

/** Split a transcript into overlapping parts of at most ~CHUNK_CHARS of text. */
export function chunkTranscript(transcript, maxChars = CHUNK_CHARS, overlapS = OVERLAP_S) {
  const parts = [];
  let i = 0;
  while (i < transcript.length) {
    let chars = 0;
    let j = i;
    while (j < transcript.length && (chars < maxChars || j === i)) chars += transcript[j++].text.length + 8;
    parts.push(transcript.slice(i, j));
    if (j >= transcript.length) break;
    // Step back so the next part starts OVERLAP_S before this one ended.
    const cut = transcript[j - 1].start - overlapS;
    let k = j - 1;
    while (k > i + 1 && transcript[k - 1].start > cut) k--;
    i = Math.max(i + 1, k);
  }
  return parts;
}

async function unload() {
  await fetch(`${OLLAMA}/api/generate`, { method: "POST", body: JSON.stringify({ model: LOCAL_MODEL, keep_alive: 0 }), signal: AbortSignal.timeout(15_000) }).catch(() => {});
}

/** Local agent: reads each part, returns every raw segment it found (unvalidated). */
export async function readLocal(video) {
  const parts = chunkTranscript(video.transcript);
  const found = [];
  let hot = null;
  const watch = setInterval(async () => {
    const g = await gpuState();
    if (g && g.tempC >= GPU_ABORT_C) hot = `the GPU reached ${g.tempC} °C`;
  }, 3000);
  try {
    for (const [n, part] of parts.entries()) {
      if (hot) throw new Error(hot);
      const header =
        parts.length > 1
          ? `\n(This is part ${n + 1} of ${parts.length} of the transcript, from [${Math.round(part[0].start)}] to [${Math.round(part.at(-1).start)}]. Report only segments that start in this part.)`
          : "";
      const r = await fetch(`${OLLAMA}/api/chat`, {
        method: "POST",
        body: JSON.stringify({
          model: LOCAL_MODEL,
          stream: false,
          think: true,
          format: SCHEMA,
          // Stay loaded between parts of this video only; unloaded explicitly below.
          keep_alive: "30s",
          options: { num_ctx: 16384, temperature: 0 },
          messages: [
            { role: "system", content: SYSTEM_PROMPT },
            { role: "user", content: buildPrompt({ ...video, transcript: part }) + header },
          ],
        }),
        signal: AbortSignal.timeout(240_000),
      });
      if (!r.ok) throw new Error(`Ollama ${r.status}`);
      const json = await r.json();
      const parsed = extractJson(json.message?.content ?? "");
      if (!parsed || !Array.isArray(parsed.segments)) throw new Error(`unusable answer on part ${n + 1}`);
      found.push(...parsed.segments);
    }
  } finally {
    clearInterval(watch);
    await unload();
  }
  return { text: JSON.stringify({ segments: found }), parts: parts.length };
}

/** One `claude -p` call: the raw answer text and what it cost. */
function runClaude(prompt, cwd) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      "claude",
      [
        "-p",
        "--model", CLAUDE_MODEL,
        "--tools", "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--setting-sources", "project",
        "--output-format", "json",
        "--system-prompt", SYSTEM_PROMPT,
      ],
      // No shell: claude is a native .exe, and a shell would have to re-quote
      // the multi-line system prompt.
      { cwd, windowsHide: true },
    );
    let out = "";
    let err = "";
    const timer = setTimeout(() => child.kill(), 180_000);
    child.stdout.on("data", (d) => (out += d));
    child.stderr.on("data", (d) => (err += d));
    child.on("error", (e) => reject(new Error(`Claude Code CLI not available: ${e.message}`)));
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

/**
 * Claude agent via the Claude Code CLI, on Simon's subscription.
 *
 * Two passes: a short one over the opening, then the whole transcript.
 *
 * The point of the first pass is that a sponsor read in the opening minutes
 * becomes skippable within seconds, instead of only once the whole video has
 * been read. It is small, so it comes back quickly, and it is handed straight
 * to the page through onPart.
 *
 * Why not read the whole thing in overlapping parts, which is the obvious way
 * to do this: it was measured on 2026-09-20, on a 62-minute episode, and it
 * lost on every count. Four overlapping 24k-character parts took 181 s and
 * $0.14, and the FIRST part did not land until 110 s -- slower to a first
 * answer than just reading the whole transcript, which takes 30-90 s and
 * $0.06-0.09. Each part also only sees its own slice, and the whole-transcript
 * read is what the 6-of-6 accuracy was measured on.
 *
 * So the second pass is the unchanged whole-transcript read, and it is the
 * answer. It covers everything the first pass covered, so a read straddling
 * the boundary is always seen whole by it, and nothing depends on stitching
 * parts together. The opening pass is best-effort: if it fails, it is ignored.
 */
export async function readClaude(video, cwd, onPart) {
  const head = video.transcript.filter((l) => l.start <= HEAD_SECONDS);
  const useHead = head.length > 0 && head.length < video.transcript.length;
  let costUsd = 0;

  if (useHead) {
    try {
      const note = `
(This is only the first ${Math.round(HEAD_SECONDS / 60)} minutes of a longer transcript. Report only segments that start within it.)`;
      const first = await runClaude(buildPrompt({ ...video, transcript: head }) + note, cwd);
      costUsd += first.costUsd || 0;
      const parsed = extractJson(first.text);
      if (parsed && Array.isArray(parsed.segments)) onPart?.({ text: first.text, part: 1, parts: 2 });
    } catch {
      // A head start, not the answer. If it fails, the full read still runs.
    }
  }

  // The answer. Throwing here is right: the caller falls back to the other reader.
  const full = await runClaude(buildPrompt(video), cwd);
  costUsd += full.costUsd || 0;
  return { text: full.text, costUsd: costUsd || null, parts: useHead ? 2 : 1, failed: null };
}

export const AGENTS = {
  local: `${LOCAL_MODEL} (local GPU)`,
  claude: `claude-${CLAUDE_MODEL} (Claude Code)`,
  marker: "marker free tier (CPU, no language model)",
  "marker-candidate": "marker candidate: six detectors + fine-tuned BGE (CPU)",
  "marker-cascade": `marker finds, claude-${CLAUDE_MODEL} places each edge`,
  "marker-cascade-local": `marker finds, ${LOCAL_MODEL} places each edge`,
  "marker-qwen": `marker + ${LOCAL_MODEL} checks (local GPU)`,
};

const PYTHON = process.env.SPONSORSKIP_PYTHON || "python";

/**
 * The free tier: marker/predict.py reads the transcript on stdin and answers
 * {segments} on stdout. It loads PyTorch and MiniLM on every call (a few
 * seconds), which is fine beside a Claude read of 30-90 s.
 */
// The candidate tier runs three encoders on the CPU. With the cheap-five gate it reads about a third
// of the lines, so a median video is ~20 s, but the longest transcript in our corpus is 3,119 lines
// and would crowd the old 180 s.
const MARKER_TIMEOUTS = { qwen: 600_000, cascade: 420_000, "cascade-local": 600_000, candidate: 300_000, free: 180_000 };

export function readMarker(id, video, root, { tier = "free", timeoutMs = MARKER_TIMEOUTS[tier] ?? 180_000 } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON, [path.join(root, "marker", "predict.py"), "serve", "--tier", tier], {
      cwd: root,
      windowsHide: true,
      env: { ...process.env, PYTHONUTF8: "1", OMP_NUM_THREADS: "2" },
    });
    let out = "";
    let err = "";
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error(`timed out after ${timeoutMs / 1000} s`));
    }, timeoutMs);
    child.stdout.on("data", (d) => (out += d));
    child.stderr.on("data", (d) => (err += d));
    child.on("error", (e) => {
      clearTimeout(timer);
      reject(e);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) return reject(new Error(`exited ${code}: ${err.trim().split("\n").pop() ?? ""}`));
      try {
        const segments = JSON.parse(out).segments ?? [];
        resolve(segments.map((s) => ({ ...s, quote: null, confidence: null })));
      } catch {
        reject(new Error("no JSON on stdout"));
      }
    });
    child.stdin.end(
      JSON.stringify({ videoID: id, channel: video.channel, duration: video.lengthSeconds, lines: video.transcript,
                       chapters: video.chapters ?? null, description: video.description ?? null }),
      "utf8",
    );
  });
}

/** Is the Claude Code CLI on this PC? (Cheap: no model call.) */
export async function claudeAvailable() {
  return (await execText("claude", ["--version"], 15_000)) !== null;
}
