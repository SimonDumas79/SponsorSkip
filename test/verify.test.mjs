/**
 * Offline checks of the checker layer. No models, no network: every checker
 * here is a stub, so these run in `npm test` beside the rest.
 *
 * What is actually being defended:
 *  - a checker can narrow or drop a candidate, never invent one;
 *  - an answer outside its own window is thrown away (the 2026-09-19 failure);
 *  - a checker that only confirms never gets asked to place edges;
 *  - a checker that throws loses nothing, because the marker found it for free.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { CHECKERS, verifyAll, verifyRegion, windowFor, renderWindow } from "../server/verify.mjs";

/** One caption line every 2 s, so line index N is at 2N seconds. */
const transcript = Array.from({ length: 200 }, (_, i) => ({ start: i * 2, text: `line ${i}` }));
const region = { start: 200, end: 260 };

test("the window runs 90 s before the region and 60 s after, and is not centred on it", () => {
  const win = windowFor(transcript, region);
  assert.equal(win.lines[0].start, 110);
  assert.equal(win.lines.at(-1).start, 320);
  // The read starts 90 s into a 210 s window: closer to the front than the middle.
  const middle = (win.lines[0].start + win.lines.at(-1).start) / 2;
  assert.ok(region.start < middle, "region must not sit at the centre of its window");
});

test("a region with too few lines around it is kept rather than checked", async () => {
  const shortTranscript = [{ start: 0, text: "a" }, { start: 2, text: "b" }];
  const out = await verifyRegion({ start: 0, end: 2 }, shortTranscript, CHECKERS.claude);
  assert.equal(out.verdict, "kept");
});

test("lines are numbered from zero within the window, not from the transcript", () => {
  const win = windowFor(transcript, region);
  const rendered = renderWindow(win.lines).split("\n");
  assert.ok(rendered[0].startsWith("  0: "));
  assert.ok(rendered[1].startsWith("  1: "));
});

test("a confirm-only checker drops a candidate it rejects", async () => {
  const checker = { confirm: async () => false, placeEdges: null };
  assert.equal(await verifyRegion(region, transcript, checker), null);
});

test("a confirm-only checker keeps the marker's own edges when it agrees", async () => {
  const checker = { confirm: async () => true, placeEdges: null };
  const out = await verifyRegion(region, transcript, checker);
  assert.equal(out.verdict, "confirmed");
  assert.equal(out.start, region.start, "edges must be untouched by a checker that only confirms");
  assert.equal(out.end, region.end);
});

test("an edge-placing checker moves the edges to the lines it named", async () => {
  // Window starts at 110 s, one line every 2 s, so line 10 is 130 s.
  const checker = { confirm: null, placeEdges: async () => ({ edges: { startLine: 10, endLine: 40 }, costUsd: 0.01 }) };
  const out = await verifyRegion(region, transcript, checker);
  assert.equal(out.verdict, "placed");
  assert.equal(out.start, 130);
  assert.equal(out.end, 190);
});

test("an answer outside its own window is thrown away, not trusted", async () => {
  // This is the 2026-09-19 failure in miniature: a plausible number from nowhere.
  const checker = { confirm: null, placeEdges: async () => ({ edges: { startLine: 9999, endLine: 10_000 }, costUsd: 0 }) };
  const out = await verifyRegion(region, transcript, checker);
  assert.equal(out.verdict, "out-of-window");
  assert.equal(out.start, region.start, "a rejected answer must leave the marker's region alone");
});

test("a negative line index is out of window too", async () => {
  const checker = { confirm: null, placeEdges: async () => ({ edges: { startLine: -1, endLine: 5 }, costUsd: 0 }) };
  assert.equal((await verifyRegion(region, transcript, checker)).verdict, "out-of-window");
});

test("an unusable end line falls back to the window's end rather than inverting the segment", async () => {
  const checker = { confirm: null, placeEdges: async () => ({ edges: { startLine: 20, endLine: 3 }, costUsd: 0 }) };
  const out = await verifyRegion(region, transcript, checker);
  assert.ok(out.end > out.start, "a segment must never end before it starts");
});

test("a checker that throws loses no candidates", async () => {
  const checker = { confirm: async () => { throw new Error("ollama down"); }, placeEdges: null };
  const kept = await verifyAll([region], transcript, checker);
  assert.equal(kept.length, 1);
  assert.equal(kept[0].verdict, "kept");
  assert.match(kept[0].reason, /checker failed/);
});

test("checking never invents a region", async () => {
  const checker = { confirm: async () => true, placeEdges: null };
  const regions = [{ start: 200, end: 260 }, { start: 300, end: 340 }];
  const kept = await verifyAll(regions, transcript, checker);
  assert.ok(kept.length <= regions.length, "a checker may only narrow or drop");
});

test("the default checker is 'none', which changes nothing", async () => {
  const out = await verifyRegion(region, transcript);
  assert.equal(out.verdict, "kept");
  assert.equal(out.start, region.start);
  assert.equal(CHECKERS.none.confirm, null);
  assert.equal(CHECKERS.none.placeEdges, null);
});

test("the local checker declares that it cannot place edges", () => {
  // Measured 2026-09-20: qwen3:8b placed starts a median 55.7 s late.
  assert.equal(CHECKERS.local.placeEdges, null);
  assert.equal(typeof CHECKERS.local.confirm, "function");
  assert.equal(typeof CHECKERS.claude.placeEdges, "function");
});
