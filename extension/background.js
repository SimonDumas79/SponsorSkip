// Talks to the SponsorSkip program on this PC (127.0.0.1:4790) for the
// content script and the popup. Requests go from here rather than from the
// YouTube page so the page's CORS and local-network rules never apply.
const api = globalThis.browser ?? globalThis.chrome;
const BASE = "http://127.0.0.1:4790";

async function call(route, timeoutMs) {
  try {
    const r = await fetch(BASE + route, { signal: AbortSignal.timeout(timeoutMs) });
    const json = await r.json();
    return r.ok ? { ok: true, data: json } : { ok: false, error: json.error || `HTTP ${r.status}` };
  } catch (e) {
    const offline = e.name === "TypeError" || /NetworkError|Failed to fetch/.test(String(e.message));
    return { ok: false, offline, error: offline ? "SponsorSkip isn't running on this PC" : e.message };
  }
}

api.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  const route = {
    health: () => call("/health", 20_000),
    quick: () => call(`/quick/${msg.id}`, 15_000),
    analyze: () => call(`/analyze/${msg.id}?reader=${msg.reader === "local" ? "local" : "claude"}${msg.fresh ? "&fresh=1" : ""}`, 300_000),
  }[msg.type];
  if (!route) return false;
  route().then(sendResponse);
  return true; // keeps the channel open for the async reply (Chrome)
});
