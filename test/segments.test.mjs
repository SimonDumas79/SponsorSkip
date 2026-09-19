// Offline checks for the parts that decide what gets skipped. Run: node --test
import assert from "node:assert/strict";
import test from "node:test";
import { buildPrompt, extractJson, MIN_S, parseSegments, snapStarts } from "../server/segments.mjs";
import { parseJson3, pickCaptionFile } from "../server/youtube.mjs";

test("extractJson finds the first balanced object, around prose and fences", () => {
  assert.deepEqual(extractJson('Sure! ```json\n{"segments": [{"quote": "a } in a string"}]}\n``` done {"x":1}'), {
    segments: [{ quote: "a } in a string" }],
  });
  assert.equal(extractJson("no json here"), null);
  assert.equal(extractJson('{"broken": '), null);
});

test("parseSegments: garbage skips nothing, never throws", () => {
  assert.deepEqual(parseSegments("I could not find any.", 600), []);
  assert.deepEqual(parseSegments('{"segments": "nope"}', 600), []);
});

test("parseSegments drops bad rows, clamps, sorts, merges same-category overlaps", () => {
  const reply = JSON.stringify({
    segments: [
      { start: 300, end: 360, category: "sponsor", quote: "later", confidence: 2 },
      { start: 60, end: 120, category: "sponsor", quote: "first" },
      { start: 110, end: 130, category: "sponsor" }, // overlaps the first → merged
      { start: 200, end: 201, category: "sponsor" }, // shorter than MIN_S → dropped
      { start: 50, end: 40 }, // backwards → dropped
      { start: "x", end: 9 }, // not a number → dropped
      { start: 590, end: 700, category: "selfpromo" }, // clamped to the length
      { start: 650, end: 700 }, // starts after the video → dropped
    ],
  });
  const segs = parseSegments(reply, 600);
  assert.deepEqual(
    segs.map((s) => [s.start, s.end, s.category]),
    [
      [60, 130, "sponsor"],
      [300, 360, "sponsor"],
      [590, 600, "selfpromo"],
    ],
  );
  assert.equal(segs[1].confidence, 1, "confidence is clamped to 0..1");
  assert.ok(segs.every((s) => s.end - s.start >= MIN_S));
});

test("buildPrompt merges lines into ~6 s chunks with [seconds] markers", () => {
  const prompt = buildPrompt({
    title: "T",
    lengthSeconds: 100,
    transcript: [
      { start: 0, text: "hello" },
      { start: 2.5, text: "there" },
      { start: 7, text: "next chunk" },
    ],
  });
  assert.match(prompt, /^Title: T\nLength: 100 seconds\n\nTranscript:\n\[0\] hello there\n\[7\] next chunk$/);
});

test("snapStarts moves a start to the line where the quote begins", () => {
  const transcript = [
    { start: 100, text: "and that is the main idea" },
    { start: 103, text: "This video is sponsored by Acme," },
    { start: 106, text: "the best widgets money can buy" },
  ];
  const [s] = snapStarts([{ start: 100, end: 160, quote: "This video is sponsored by Acme", category: "sponsor" }], transcript);
  assert.equal(s.start, 103);
});

test("snapStarts leaves a start alone when the quote is missing or far away", () => {
  const transcript = [
    { start: 10, text: "unrelated words" },
    { start: 200, text: "this video is sponsored by acme" },
  ];
  const seg = { start: 12, end: 60, quote: "This video is sponsored by Acme" };
  assert.equal(snapStarts([seg], transcript)[0].start, 12, "outside the 15 s window");
  assert.equal(snapStarts([{ ...seg, quote: null }], transcript)[0].start, 12);
});

test("snapStarts only matches a quote that starts in the line, not one that begins in the next", () => {
  const transcript = [
    { start: 50, text: "we will be right back after" },
    { start: 53, text: "this video is sponsored by acme" },
  ];
  const [s] = snapStarts([{ start: 50, end: 90, quote: "this video is sponsored by acme" }], transcript);
  assert.equal(s.start, 53);
});

test("parseJson3 joins word-level segments per event and drops blank ones", () => {
  const lines = parseJson3({
    events: [
      { tStartMs: 0, segs: [{ utf8: "Hello" }, { utf8: " world" }] },
      { tStartMs: 1500, segs: [{ utf8: "\n" }] },
      { tStartMs: 2000 },
      { tStartMs: 3000, segs: [{ utf8: "  again  " }] },
    ],
  });
  assert.deepEqual(lines, [
    { start: 0, text: "Hello world" },
    { start: 3, text: "again" },
  ]);
});

test("pickCaptionFile prefers published English over auto-generated", () => {
  assert.equal(pickCaptionFile(["abc.en-orig.json3", "abc.en.json3"], "abc"), "abc.en.json3");
  assert.equal(pickCaptionFile(["abc.en-orig.json3", "abc.de.json3"], "abc"), "abc.en-orig.json3");
  assert.equal(pickCaptionFile(["abc.en-GB.json3"], "abc"), "abc.en-GB.json3");
  assert.equal(pickCaptionFile(["abc.de.json3"], "abc"), null);
});

test("snapStarts also moves the end to where the show resumes", () => {
  const transcript = [
    { start: 100, text: "This video is sponsored by Acme." },
    { start: 130, text: "Use code DWARKESH for ten percent off." },
    { start: 136, text: "Okay, back to the question of scaling." },
    { start: 142, text: "So what happens next?" },
  ];
  const [s] = snapStarts(
    [{ start: 100, end: 142, quote: "This video is sponsored by Acme", resumeQuote: "Okay, back to the question of scaling", category: "sponsor" }],
    transcript,
  );
  assert.deepEqual([s.start, s.end], [100, 136]);
});
