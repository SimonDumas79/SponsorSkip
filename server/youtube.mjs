/**
 * A video's transcript via yt-dlp (python -m yt_dlp): one call prints the
 * title, length and channel and writes the English captions as json3.
 *
 * Why yt-dlp and not the page's own endpoints: as of 2026-09-18 YouTube
 * refuses get_transcript ("Precondition check failed") and serves empty
 * timedtext without a proof-of-origin token, from a script and even from an
 * automated browser. yt-dlp's maintainers keep up with that; we don't have to.
 */
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

/**
 * The caption tracks to fetch: English as published, the auto-generated
 * original, and regional English. NOT "en.*": that pattern also matches
 * YouTube's machine translations into English from every other language
 * (en-pl, en-it, …). On a channel that publishes in many languages that's
 * dozens of downloads, and YouTube answered 429 (Kurzgesagt, 2026-09-19).
 */
export const SUB_LANGS = "en,en-orig,en-US,en-GB";

/** json3 caption events → [{ start, text }]. Word-level auto captions join per event. */
export function parseJson3(json) {
  return (json?.events ?? [])
    .filter((e) => Array.isArray(e.segs) && Number.isFinite(e.tStartMs))
    .map((e) => ({ start: e.tStartMs / 1000, text: e.segs.map((s) => s.utf8 ?? "").join("").replace(/\s+/g, " ").trim() }))
    .filter((l) => l.text);
}

/** Which caption file to read: English as published, then the auto-generated original. */
export function pickCaptionFile(files, videoId) {
  const order = [`${videoId}.en.json3`, `${videoId}.en-orig.json3`, `${videoId}.en-en.json3`];
  for (const name of order) if (files.includes(name)) return name;
  return files.find((f) => f.startsWith(`${videoId}.en`) && f.endsWith(".json3")) ?? null;
}

function run(cmd, args, cwd, timeoutMs = 90_000) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { cwd, windowsHide: true });
    let out = "";
    let err = "";
    const timer = setTimeout(() => child.kill(), timeoutMs);
    child.stdout.on("data", (d) => (out += d));
    child.stderr.on("data", (d) => (err += d));
    child.on("error", reject);
    child.on("close", (code) => {
      clearTimeout(timer);
      code === 0 ? resolve(out) : reject(new Error(`yt-dlp exited ${code}: ${err.trim().split("\n").pop()?.slice(0, 200) ?? ""}`));
    });
  });
}

/** { title, channel, lengthSeconds, transcript } — transcript null when the video has no English captions. */
export async function getTranscript(videoId, { python = process.env.SPONSORSKIP_PYTHON || "python" } = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "sponsorskip-"));
  try {
    const out = await run(
      python,
      [
        "-m", "yt_dlp",
        "--skip-download", "--no-simulate", "--no-warnings", "--quiet",
        "--write-subs", "--write-auto-subs", "--sub-langs", SUB_LANGS, "--sub-format", "json3",
        "--print", "%(title)s\t%(duration)s\t%(channel)s\t%(language)s",
        "-o", "%(id)s.%(ext)s",
        `https://www.youtube.com/watch?v=${videoId}`,
      ],
      dir,
    );
    const [title, duration, channel, language] = out.trim().split("\n")[0].split("\t");
    const file = pickCaptionFile(fs.readdirSync(dir), videoId);
    const transcript = file ? parseJson3(JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"))) : null;
    return {
      title: title || null,
      channel: channel && channel !== "NA" ? channel : null,
      lengthSeconds: Number(duration) || null,
      // The spoken language YouTube reports ("ru", "en", …; null when unknown). When it is not English the
      // "en" track is a machine translation, the class behind the worst video in every measured set.
      language: language && language !== "NA" ? language : null,
      transcript: transcript?.length ? transcript : null,
      captionFile: file,
    };
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}
