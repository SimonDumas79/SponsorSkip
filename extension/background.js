// Talks to the SponsorSkip program on this PC (127.0.0.1:4790) and to
// SponsorBlock, for the content script and the popup. Requests go from here
// rather than from the YouTube page so the page's CORS and local-network
// rules never apply.
const api = globalThis.browser ?? globalThis.chrome;
const BASE = "http://127.0.0.1:4790";
const SB = "https://sponsor.ajay.app";
const USER_AGENT = `SponsorSkip/${api.runtime.getManifest().version}`;

async function call(route, timeoutMs, body) {
  try {
    // x-sponsorskip: the program refuses any request without it, which is what stops web pages (server.mjs).
    const headers = { "x-sponsorskip": "1", ...(body === undefined ? {} : { "content-type": "application/json" }) };
    const init = body === undefined ? { headers } : { method: "POST", headers, body: JSON.stringify(body) };
    const r = await fetch(BASE + route, { ...init, signal: AbortSignal.timeout(timeoutMs) });
    const json = await r.json();
    return r.ok ? { ok: true, data: json } : { ok: false, error: json.error || `HTTP ${r.status}` };
  } catch (e) {
    const offline = e.name === "TypeError" || /NetworkError|Failed to fetch/.test(String(e.message));
    return { ok: false, offline, error: offline ? "SponsorSkip isn't running on this PC" : e.message };
  }
}

// --- SponsorBlock -------------------------------------------------------------

async function sha256Hex(text) {
  const bytes = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(bytes)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** SponsorBlock's segments for a video. Looked up by hash prefix, so SponsorBlock never sees the id. */
async function sbLookup(id) {
  try {
    const prefix = (await sha256Hex(id)).slice(0, 4);
    const r = await fetch(`${SB}/api/skipSegments/${prefix}?categories=${encodeURIComponent('["sponsor","selfpromo"]')}`, { signal: AbortSignal.timeout(10_000) });
    if (r.status === 404) return { ok: true, data: [] };
    if (!r.ok) return { ok: false, error: `SponsorBlock ${r.status}` };
    const match = (await r.json()).find((v) => v.videoID === id);
    return { ok: true, data: (match?.segments ?? []).map((s) => ({ start: s.segment[0], end: s.segment[1], category: s.category, votes: s.votes })) };
  } catch (e) {
    return { ok: false, error: `SponsorBlock unreachable: ${e.message}` };
  }
}

/**
 * The private SponsorBlock user ID for these submissions. Created once and
 * kept in synced storage, so every browser signed into the same account
 * submits as one user. You can paste your existing SponsorBlock ID instead.
 * It is a secret: SponsorBlock only ever publishes a hash of it.
 */
async function sbUserId() {
  const { sbUserId } = await api.storage.sync.get({ sbUserId: "" });
  if (sbUserId && sbUserId.length >= 30) return sbUserId;
  const fresh = crypto.randomUUID().replace(/-/g, "") + crypto.randomUUID().replace(/-/g, "").slice(0, 8);
  await api.storage.sync.set({ sbUserId: fresh });
  return fresh;
}

/**
 * Submit ONE segment the user has previewed and approved in the popup.
 * SponsorBlock forbids automated submissions (wiki: "Automating
 * submissions"). The popup enforces that both edges were previewed first,
 * and nothing in this extension calls this without a click on Submit.
 */
async function sbSubmit({ id, start, end, category, duration }) {
  try {
    const r = await fetch(`${SB}/api/skipSegments`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        videoID: id,
        userID: await sbUserId(),
        userAgent: USER_AGENT,
        videoDuration: duration || 0,
        segments: [{ segment: [Number(start.toFixed(3)), Number(end.toFixed(3))], category, actionType: "skip" }],
      }),
      signal: AbortSignal.timeout(15_000),
    });
    const text = await r.text();
    if (r.ok) return { ok: true };
    const reason =
      { 400: "SponsorBlock rejected it as invalid", 403: "SponsorBlock refused it", 409: "SponsorBlock already has this segment", 429: "Too many submissions; try again in a bit" }[r.status] ??
      `SponsorBlock returned ${r.status}`;
    return { ok: false, error: `${reason}${text ? `: ${text.slice(0, 160)}` : ""}` };
  } catch (e) {
    return { ok: false, error: `SponsorBlock unreachable: ${e.message}` };
  }
}

// Four levels (2026-09-22); level 1 is the default and 2-4 are opt-in (2026-09-24). Settings saved by 0.7.x name readers that are gone; each maps to
// the level that replaced it. Level 2 (your GPU checks v3) is not built yet, so marker-qwen lands on 1.
const READERS = ["marker-v3", "marker-v3-cascade", "claude"];
const LEGACY_READERS = { marker: "marker-v3", "marker-candidate": "marker-v3", "marker-qwen": "marker-v3",
                         "marker-cascade": "marker-v3-cascade", local: "claude" };
const readerFor = (r) => (READERS.includes(r) ? r : LEGACY_READERS[r] ?? "marker-v3");
// Migrate a saved 0.7.x setting as soon as the worker starts, so the page's own toast (which reads the
// setting directly) names the level that will actually run, even if the popup is never opened.
api.storage.sync.get({ reader: "marker-v3" }).then(({ reader }) => {
  if (readerFor(reader) !== reader) api.storage.sync.set({ reader: readerFor(reader) });
});

api.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  const route = {
    health: () => call("/health", 20_000),
    quick: () => call(`/quick/${msg.id}`, 15_000),
    analyze: () => call(`/analyze/${msg.id}?reader=${readerFor(msg.reader)}${msg.fresh ? "&fresh=1" : ""}`, 600_000),
    progress: () => call(`/progress/${msg.id}`, 10_000),
    aiGet: () => call("/ai-config", 10_000),
    aiSave: () => call("/ai-config", 10_000, msg.config),
    aiTest: () => call("/ai-test", 120_000, {}),
    sbLookup: () => sbLookup(msg.id),
    sbSubmit: () => sbSubmit(msg),
  }[msg.type];
  if (!route) return false;
  route().then(sendResponse);
  return true; // keeps the channel open for the async reply (Chrome)
});
