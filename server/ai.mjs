/**
 * "Your AI": the one place SponsorSkip talks to a language model (2026-09-24).
 *
 * The person picks their AI in the popup (Manage AI model). Two kinds:
 *
 *   An agent CLI they are already signed in to, run on this PC with the prompt on stdin.
 *   Their subscription pays; SponsorSkip never sees a key.
 *     claude   Claude Code        claude -p
 *     codex    Codex CLI          codex exec -
 *     gemini   Gemini CLI         gemini -p
 *     ollama   Ollama, this PC    its local HTTP API (nothing leaves the PC)
 *     custom   any command that reads a prompt on stdin and prints the answer on stdout
 *
 *   An OpenAI-compatible API (OpenAI, OpenRouter, LM Studio, a llama.cpp server...): a base URL,
 *   a model name and their own key. Billed per token by that provider. The key is kept in
 *   ai-config.json next to the program (gitignored) and is never sent back to the popup.
 *
 * Only Claude Haiku through Claude Code has measured accuracy on this task (README); the rest
 * run the same prompt unmeasured, and the popup says so.
 */
import { execFile, spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CONFIG = path.join(root, "ai-config.json");
const OLLAMA = (process.env.OLLAMA_URL || "http://localhost:11434").replace(/\/+$/, "");

export const PROVIDERS = {
  claude: { label: "Claude Code", model: process.env.SPONSORSKIP_MODEL || "haiku", bin: "claude" },
  codex: { label: "Codex CLI", model: "", bin: "codex" },
  gemini: { label: "Gemini CLI", model: "", bin: "gemini" },
  ollama: { label: "Ollama, this PC", model: "qwen3:8b" },
  custom: { label: "Custom command", model: "" },
  openai: { label: "API endpoint", model: "gpt-4o-mini" },
};

const DEFAULT = { provider: "claude", model: "", command: "", url: "", key: "" };

export function loadConfig() {
  try {
    const c = { ...DEFAULT, ...JSON.parse(fs.readFileSync(CONFIG, "utf8")) };
    return PROVIDERS[c.provider] ? c : { ...DEFAULT };
  } catch {
    return { ...DEFAULT };
  }
}

/**
 * Save what the popup sent. Three rules keep this route from being a way in (2026-09-24 review):
 *  - A custom command is NEVER taken from here. It runs through a shell, and any installed browser
 *    extension can reach this route, so the command comes only from editing ai-config.json by hand.
 *  - The model name reaches a command line (Codex and Gemini run through a shell on Windows), so it
 *    may only hold model-name characters.
 *  - A stored API key stays only while the provider AND the URL are unchanged, so a key is only ever
 *    sent to the URL it was entered with. An empty key field keeps it; the popup never holds it.
 */
const MODEL_NAME = /^[\w.:\/@-]{0,120}$/;
export function saveConfig(input) {
  const was = loadConfig();
  const provider = PROVIDERS[input?.provider] ? input.provider : "claude";
  const str = (v, max = 500) => (typeof v === "string" ? v.trim().slice(0, max) : "");
  const url = str(input.url).replace(/\/+$/, "");
  const sameTarget = provider === was.provider && url === was.url;
  const next = {
    provider,
    model: str(input.model, 120),
    command: was.command,
    url,
    key: str(input.key) || (sameTarget ? was.key : ""),
  };
  if (!MODEL_NAME.test(next.model)) throw new Error("a model name can only hold letters, digits and . : / @ - _");
  if (next.url && !/^https?:\/\//.test(next.url)) throw new Error("the API URL must start with http:// or https://");
  fs.writeFileSync(CONFIG, JSON.stringify(next, null, 2));
  return publicConfig(next);
}

/** What the popup may see: everything but the key. */
export function publicConfig(c = loadConfig()) {
  return { provider: c.provider, model: c.model, command: c.command, url: c.url, hasKey: !!c.key, label: describe(c),
           measured: c.provider === "claude" && modelOf(c) === "haiku" };
}

const modelOf = (c) => c.model || PROVIDERS[c.provider].model;

/** "claude-haiku (Claude Code)", "gpt-4o-mini (API: api.openai.com)"... for logs and the popup. */
export function describe(c = loadConfig()) {
  const m = modelOf(c);
  if (c.provider === "claude") return `claude-${m} (Claude Code)`;
  if (c.provider === "openai") {
    let host = c.url;
    try {
      host = new URL(c.url).host;
    } catch {}
    return `${m || "model"} (API: ${host || "no URL set"})`;
  }
  if (c.provider === "custom") return `custom command (${(c.command.split(/\s+/)[0] || "not set").slice(0, 40)})`;
  return m ? `${m} (${PROVIDERS[c.provider].label})` : PROVIDERS[c.provider].label;
}

function execText(cmd, args, timeout = 15_000) {
  return new Promise((resolve) =>
    execFile(cmd, args, { timeout, windowsHide: true, shell: process.platform === "win32" && cmd !== "claude" },
             (e, out) => resolve(e ? null : String(out).trim())));
}

/** Can the chosen AI run right now? Cheap: no model call. */
export async function available(c = loadConfig()) {
  switch (c.provider) {
    case "claude":
    case "codex":
    case "gemini": {
      const v = await execText(PROVIDERS[c.provider].bin, ["--version"]);
      return v ? { ready: true, note: "ready" } : { ready: false, note: `${PROVIDERS[c.provider].label} not found (install it and sign in)` };
    }
    case "ollama":
      try {
        const tags = await (await fetch(`${OLLAMA}/api/tags`, { signal: AbortSignal.timeout(3000) })).json();
        const m = modelOf(c);
        return tags.models?.some((t) => t.name === m || t.name === `${m}:latest`)
          ? { ready: true, note: "ready" }
          : { ready: false, note: `Ollama is running but has no ${m} (ollama pull ${m})` };
      } catch {
        return { ready: false, note: "Ollama isn't running on this PC" };
      }
    case "custom":
      return c.command ? { ready: true, note: "set (use Test to check it)" } : { ready: false, note: "no command set" };
    case "openai":
      return c.url && c.key && modelOf(c) ? { ready: true, note: "set (use Test to check it)" }
        : { ready: false, note: "needs a URL, a model and a key" };
  }
  return { ready: false, note: "unknown AI" };
}

/** Run a process with `input` on stdin; resolve its stdout. */
function run(cmd, args, input, { timeoutMs, shell = false, cwd = root } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { cwd, windowsHide: true, shell });
    let out = "";
    let err = "";
    const timer = setTimeout(() => child.kill(), timeoutMs);
    child.stdout.on("data", (d) => (out += d));
    child.stderr.on("data", (d) => (err += d));
    child.on("error", (e) => {
      clearTimeout(timer);
      reject(new Error(`${cmd} not available: ${e.message}`));
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) return reject(new Error(`${cmd} exited ${code}: ${(err || out).trim().slice(-300)}`));
      resolve(out);
    });
    child.stdin.on("error", () => {}); // a command that exits without reading stdin must not crash the program
    child.stdin.end(input, "utf8");
  });
}

// CLIs other than Claude Code take no separate system prompt here, so it leads the input.
const joined = (system, user) => `${system}\n\n---\n\n${user}`;

/**
 * One question to the person's AI: { text, costUsd }. Throws when it cannot answer; callers fall
 * back (the reader to SponsorBlock, the edge placer to the marker's own edges).
 */
export async function ask(system, user, { timeoutMs = 180_000, config = loadConfig() } = {}) {
  const c = config;
  const model = modelOf(c);
  switch (c.provider) {
    case "claude": {
      const out = await run("claude", ["-p", "--model", model, "--tools", "", "--strict-mcp-config", "--no-session-persistence",
        "--setting-sources", "project", "--output-format", "json", "--system-prompt", system], user, { timeoutMs });
      const json = JSON.parse(out);
      if (json.is_error) throw new Error(`claude: ${json.result ?? "error"}`);
      return { text: json.result ?? "", costUsd: json.total_cost_usd ?? null };
    }
    case "codex": {
      // exec is Codex's non-interactive mode; "-" reads the prompt from stdin. Read-only sandbox:
      // it only has to answer, never touch files.
      const args = ["exec", "--skip-git-repo-check", "--sandbox", "read-only", ...(model ? ["-m", model] : []), "-"];
      return { text: await run("codex", args, joined(system, user), { timeoutMs, shell: process.platform === "win32" }), costUsd: null };
    }
    case "gemini": {
      // -p is appended to whatever arrives on stdin. One word, because on Windows the npm shim needs a
      // shell, and a shell would split a sentence into separate arguments.
      const args = [...(model ? ["-m", model] : []), "-p", "Answer."];
      return { text: await run("gemini", args, joined(system, user), { timeoutMs, shell: process.platform === "win32" }), costUsd: null };
    }
    case "custom": {
      if (!c.command) throw new Error("no custom command set");
      return { text: await run(c.command, [], joined(system, user), { timeoutMs, shell: true }), costUsd: null };
    }
    case "ollama": {
      const r = await fetch(`${OLLAMA}/api/chat`, {
        method: "POST",
        body: JSON.stringify({ model, stream: false, keep_alive: 0, options: { temperature: 0, num_ctx: 32768 },
                               messages: [{ role: "system", content: system }, { role: "user", content: user }] }),
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (!r.ok) throw new Error(`Ollama ${r.status}`);
      return { text: (await r.json()).message?.content ?? "", costUsd: null };
    }
    case "openai": {
      if (!c.url || !c.key) throw new Error("the API endpoint needs a URL and a key");
      const r = await fetch(`${c.url}/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json", authorization: `Bearer ${c.key}` },
        body: JSON.stringify({ model, temperature: 0, messages: [{ role: "system", content: system }, { role: "user", content: user }] }),
        signal: AbortSignal.timeout(timeoutMs),
      });
      const json = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(`API ${r.status}: ${String(json.error?.message ?? "").slice(0, 200)}`);
      return { text: json.choices?.[0]?.message?.content ?? "", costUsd: null };
    }
  }
  throw new Error("unknown AI");
}

/** The popup's Test button: a tiny question, timed. */
export async function test(config = loadConfig()) {
  const t0 = Date.now();
  try {
    const { text } = await ask("You are a connectivity check. Reply with exactly: OK", "Reply with exactly: OK", { timeoutMs: 90_000, config });
    const seconds = Math.round((Date.now() - t0) / 100) / 10;
    return /\bok\b/i.test(text) ? { ok: true, seconds } : { ok: false, seconds, error: `answered, but not as asked: ${text.trim().slice(0, 120)}` };
  } catch (e) {
    return { ok: false, error: String(e.message).slice(0, 300) };
  }
}
