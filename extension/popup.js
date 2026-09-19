const api = globalThis.browser ?? globalThis.chrome;
const DEFAULTS = { enabled: true, skipSelfpromo: false, reader: "claude" };
const $ = (id) => document.getElementById(id);
const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

async function loadSettings() {
  const s = { ...DEFAULTS, ...(await api.storage.sync.get(DEFAULTS)) };
  $("enabled").checked = s.enabled;
  $("skipSelfpromo").checked = s.skipSelfpromo;
  for (const r of document.querySelectorAll('input[name="reader"]')) r.checked = r.value === s.reader;
}
$("enabled").addEventListener("change", (e) => api.storage.sync.set({ enabled: e.target.checked }));
$("skipSelfpromo").addEventListener("change", (e) => api.storage.sync.set({ skipSelfpromo: e.target.checked }));
for (const r of document.querySelectorAll('input[name="reader"]')) r.addEventListener("change", (e) => api.storage.sync.set({ reader: e.target.value }));

async function connection() {
  const h = await api.runtime.sendMessage({ type: "health" });
  if (!h.ok) {
    $("conn-dot").className = "dot bad";
    $("conn-title").textContent = "Not connected on this PC";
    $("conn-detail").textContent = "SponsorSkip needs its small program running here, signed in to Claude through Claude Code.";
    $("setup").hidden = false;
    return;
  }
  const [claude, local] = [h.data.agents.find((a) => /claude/i.test(a.name)), h.data.agents.find((a) => /GPU/.test(a.name))];
  $("conn-dot").className = `dot ${claude?.ready ? "ok" : "warn"}`;
  $("conn-title").textContent = claude?.ready ? "Connected: Claude via Claude Code" : "Connected, but Claude Code isn't available";
  $("conn-detail").textContent = `Local GPU: ${local?.note ?? "unknown"}.`;
  $("setup").hidden = !!claude?.ready;
}

async function thisVideo() {
  const [tab] = await api.tabs.query({ active: true, currentWindow: true });
  if (!tab?.url?.startsWith("https://www.youtube.com/watch")) return;
  let state;
  try {
    state = await api.tabs.sendMessage(tab.id, { type: "state" });
  } catch {
    return; // the page hasn't loaded the script yet
  }
  if (!state?.videoId) return;
  $("video").hidden = false;
  const r = state.result;
  $("video-source").textContent = !r
    ? "Reading…"
    : r.error
      ? r.error
      : r.interim
        ? "Using SponsorBlock while Claude reads…"
        : `${r.source}${r.seconds ? `, ${r.seconds}s` : ""}${r.reason ? ` (${r.reason})` : ""}`;
  $("segments").replaceChildren(
    ...(r?.segments ?? []).map((s) => {
      const li = document.createElement("li");
      li.textContent = `${s.category === "selfpromo" ? "Self-promo" : "Sponsor"} ${fmt(s.start)}–${fmt(s.end)}`;
      const q = document.createElement("span");
      q.className = "muted small";
      q.textContent = s.quote ? `“${s.quote.slice(0, 32)}…”` : "";
      li.append(q);
      return li;
    }),
  );
  $("reread").onclick = () => {
    api.tabs.sendMessage(tab.id, { type: "reread" });
    $("video-source").textContent = "Reading again…";
    $("segments").replaceChildren();
  };
}

loadSettings();
connection();
thisVideo();
