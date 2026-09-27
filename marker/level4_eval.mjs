#!/usr/bin/env node
/**
 * Level 4 exactly as the extension serves it (2026-09-24): server/agents.mjs readAI (the AI picked in the popup) on a video's
 * stored captions, then the same clean-up server.mjs applies (parse, snap starts to caption lines,
 * merge overlaps). The research labeller (claude_label.py) uses a different prompt and windowing, so
 * its numbers are not the served default's. Writes one JSON line per video; resumable.
 *
 *   node marker/level4_eval.mjs <list of caption files .json> <out .jsonl> [workers]
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readAI } from "../server/agents.mjs";
import { mergeOverlaps, parseSegments, snapStarts } from "../server/segments.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const [listPath, outPath, workersArg] = process.argv.slice(2);
const files = JSON.parse(fs.readFileSync(listPath, "utf8"));
const done = new Set(
  fs.existsSync(outPath)
    ? fs.readFileSync(outPath, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l).videoID)
    : [],
);
const todo = files.filter((f) => !done.has(path.basename(f, ".json")));
console.log(`${done.size} done, ${todo.length} to read with the served Claude reader`);

async function one(file) {
  const caps = JSON.parse(fs.readFileSync(file, "utf8"));
  const video = { title: caps.title, channel: caps.channel, lengthSeconds: caps.duration,
                  transcript: caps.lines.map((l) => ({ start: l.start, text: l.text })) };
  const t0 = Date.now();
  let rec;
  try {
    const { text, costUsd, parts, failed } = await readAI(video);
    const raw = parseSegments(text, video.lengthSeconds);
    const segments = mergeOverlaps(snapStarts(raw, video.transcript, 15, video.lengthSeconds));
    // raw: before snapping, with the quotes, so an end-cap rule can ask whether the resume quote was found.
    rec = { videoID: caps.videoID, segments, raw, costUsd, parts, failed: failed ?? null };
  } catch (e) {
    rec = { videoID: caps.videoID, segments: null, error: String(e.message).slice(0, 200) };
  }
  rec.seconds = Math.round((Date.now() - t0) / 1000);
  fs.appendFileSync(outPath, JSON.stringify(rec) + "\n");
  console.log(`  ${caps.videoID}  ${rec.segments ? rec.segments.length + " segment(s)" : "FAILED " + rec.error}  ${rec.seconds}s`);
}

const queue = [...todo];
await Promise.all(Array.from({ length: Number(workersArg) || 4 }, async () => {
  while (queue.length) await one(queue.shift());
}));
