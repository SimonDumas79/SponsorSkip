/**
 * Can a small model place the EDGES of a sponsor read when it is handed the
 * read already located, and asked for a LINE INDEX instead of a timestamp?
 *
 * Six reads across two Dwarkesh episodes, ground truth from SponsorBlock.
 * Each read is given as a marker-centred window (90 s before the real start,
 * 60 s after the real end, so the ad is NOT centred and the model still has to
 * find the boundaries). The model answers with the line the ad begins on and
 * the line the show resumes on. qwen3:8b and Claude Haiku get the same window
 * and the same question, so the comparison is like for like.
 *
 * NOT a unit test: it spends Claude usage, uses the GPU and fetches captions.
 * Run it deliberately: `node marker/edge_windows.mjs`. This is the harness to
 * score a trained detector against, so the numbers stay comparable.
 */
import { spawn } from "node:child_process";
import { getTranscript } from "../server/youtube.mjs";

const OLLAMA = "http://localhost:11434";
const MODEL = "qwen3:8b";

const TRUTH = [
  { id: "6AgOfiZOWiY", start: 1254.8, end: 1322.4 },
  { id: "6AgOfiZOWiY", start: 2343.9, end: 2422.3 },
  { id: "6AgOfiZOWiY", start: 3615.5, end: 3678.1 },
  { id: "X50zezLFWWI", start: 1333.4, end: 1408.4 },
  { id: "X50zezLFWWI", start: 2554.2, end: 2631.9 },
  { id: "X50zezLFWWI", start: 4399.3, end: 4471.1 },
];

const SCHEMA = {
  type: "object",
  properties: { start_line: { type: ["integer", "null"] }, end_line: { type: ["integer", "null"] } },
  required: ["start_line", "end_line"],
};

const ASK = `You are given numbered caption lines from part of a podcast. Somewhere inside them there is a sponsor read: an advertisement the host reads out.

Answer ONLY with JSON: {"start_line": <number>, "end_line": <number>}
  start_line = the number of the FIRST line of the sponsor read.
  end_line   = the number of the first line where the actual show RESUMES after it.
If there is no sponsor read at all, answer {"start_line": null, "end_line": null}.`;

function windowFor(transcript, truth) {
  const from = truth.start - 90;
  const to = truth.end + 60;
  return transcript.filter((l) => l.start >= from && l.start <= to);
}

const render = (lines) => lines.map((l, i) => `${String(i).padStart(3)}: ${l.text.replace(/\s+/g, " ").trim()}`).join("\n");

async function askQwen(lines) {
  const r = await fetch(`${OLLAMA}/api/chat`, {
    method: "POST",
    body: JSON.stringify({
      model: MODEL,
      stream: false,
      think: true,
      format: SCHEMA,
      keep_alive: "60s",
      options: { num_ctx: 16384, temperature: 0 },
      messages: [{ role: "user", content: `${ASK}\n\n${render(lines)}` }],
    }),
    signal: AbortSignal.timeout(180_000),
  });
  if (!r.ok) throw new Error(`ollama ${r.status}`);
  return JSON.parse((await r.json()).message.content);
}

function askClaude(lines) {
  return new Promise((resolve, reject) => {
    const child = spawn("claude", ["-p", "--model", "haiku", "--tools", "", "--strict-mcp-config", "--no-session-persistence", "--output-format", "json"], { windowsHide: true });
    let out = "";
    const timer = setTimeout(() => child.kill(), 120_000);
    child.stdout.on("data", (d) => (out += d));
    child.on("error", reject);
    child.on("close", () => {
      clearTimeout(timer);
      try {
        const text = JSON.parse(out).result ?? "";
        const m = text.replace(/```(?:json)?/g, "").match(/\{[\s\S]*?\}/);
        resolve(JSON.parse(m[0]));
      } catch (e) {
        reject(new Error(`unusable: ${out.slice(0, 200)}`));
      }
    });
    child.stdin.end(`${ASK}\n\n${render(lines)}`);
  });
}

const at = (lines, i) => (Number.isInteger(i) && lines[i] ? lines[i].start : null);
const err = (got, want) => (got === null ? null : +(got - want).toFixed(1));
const show = (e) => (e === null ? "  MISSED" : `${e > 0 ? "+" : ""}${e}s`);

const transcripts = {};
const rows = [];
for (const [n, truth] of TRUTH.entries()) {
  transcripts[truth.id] ??= (await getTranscript(truth.id)).transcript;
  const lines = windowFor(transcripts[truth.id], truth);
  const row = { n: n + 1, id: truth.id, at: Math.round(truth.start), lines: lines.length };

  for (const [name, fn] of [["qwen", askQwen], ["claude", askClaude]]) {
    const t0 = Date.now();
    try {
      const a = await fn(lines);
      row[name] = { s: err(at(lines, a.start_line), truth.start), e: err(at(lines, a.end_line), truth.end), secs: Math.round((Date.now() - t0) / 1000) };
    } catch (e) {
      row[name] = { s: null, e: null, secs: Math.round((Date.now() - t0) / 1000), fail: e.message.slice(0, 60) };
    }
  }
  rows.push(row);
  console.log(`read ${row.n} (${row.id} @ ${row.at}s, ${row.lines} lines)  qwen start ${show(row.qwen.s)} end ${show(row.qwen.e)} (${row.qwen.secs}s)   claude start ${show(row.claude.s)} end ${show(row.claude.e)} (${row.claude.secs}s)`);
}

console.log("\n=== summary (6 reads, error vs SponsorBlock) ===");
for (const name of ["qwen", "claude"]) {
  const got = rows.filter((r) => r[name].s !== null);
  const abs = (k) => got.map((r) => Math.abs(r[name][k]));
  const med = (xs) => (xs.length ? xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)] : NaN);
  const within = (k, s) => got.filter((r) => Math.abs(r[name][k]) <= s).length;
  console.log(
    `${name.padEnd(7)} found ${got.length}/6   start: median ${med(abs("s"))}s, worst ${Math.max(...abs("s"), 0)}s, within 5s ${within("s", 5)}/6   ` +
      `end: median ${med(abs("e"))}s, worst ${Math.max(...abs("e"), 0)}s   avg ${Math.round(rows.reduce((a, r) => a + r[name].secs, 0) / rows.length)}s/window`,
  );
}
for (const r of rows) for (const n of ["qwen", "claude"]) if (r[n].fail) console.log(`  ${n} failed on read ${r.n}: ${r[n].fail}`);

// Never leave a model in VRAM.
await fetch(`${OLLAMA}/api/generate`, { method: "POST", body: JSON.stringify({ model: MODEL, keep_alive: 0 }) }).catch(() => {});
console.log("\nlocal model unloaded");
