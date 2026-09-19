// SponsorSkip on YouTube pages: sends the video id to the program on this PC
// (through the background worker), then skips each sponsor segment the
// playhead enters, with an Undo. It stays still during YouTube's own ads.
(() => {
  "use strict";
  const api = globalThis.browser ?? globalThis.chrome;
  const DEFAULTS = { enabled: true, skipSelfpromo: false, reader: "claude" };
  let settings = { ...DEFAULTS };
  let videoId = null;
  let result = null; // the latest result for this video, for the popup
  let segments = [];
  const undone = new Set();

  const send = (msg) => api.runtime.sendMessage(msg);
  const categories = () => (settings.skipSelfpromo ? ["sponsor", "selfpromo"] : ["sponsor"]);
  const pick = (r) => (r?.segments || []).filter((s) => categories().includes(s.category));
  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const key = (s) => `${Math.round(s.start)}-${Math.round(s.end)}`;

  api.storage.sync.get(DEFAULTS).then((s) => (settings = { ...DEFAULTS, ...s }));
  api.storage.onChanged.addListener((changes) => {
    for (const [k, v] of Object.entries(changes)) settings[k] = v.newValue;
    segments = pick(result);
  });

  // --- one small status line over the player --------------------------------
  let toastEl = null;
  let toastTimer = null;
  function toast(text, action) {
    const host = document.getElementById("movie_player") || document.body;
    if (!toastEl || !host.contains(toastEl)) {
      toastEl = document.createElement("div");
      toastEl.style.cssText =
        "position:absolute;left:12px;bottom:64px;z-index:60;background:rgba(0,0,0,.8);color:#fff;font:13px/1.4 Roboto,Arial,sans-serif;padding:6px 10px;border-radius:6px;";
      host.appendChild(toastEl);
    }
    toastEl.textContent = text;
    if (action) {
      const b = document.createElement("button");
      b.textContent = action.label;
      b.style.cssText = "margin-left:10px;background:#3ea6ff;color:#000;border:0;border-radius:4px;padding:2px 8px;cursor:pointer;font:inherit;";
      b.onclick = (e) => {
        e.stopPropagation();
        action.run();
        toastEl.style.display = "none";
      };
      toastEl.appendChild(b);
    }
    toastEl.style.display = "block";
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastEl && (toastEl.style.display = "none"), action ? 8000 : 4000);
  }

  // --- skipping --------------------------------------------------------------
  setInterval(() => {
    if (!settings.enabled || !segments.length) return;
    const video = document.querySelector("video.html5-main-video") || document.querySelector("video");
    if (!video || document.getElementById("movie_player")?.classList.contains("ad-showing")) return;
    const t = video.currentTime;
    for (const s of segments) {
      if (undone.has(key(s)) || t < s.start || t >= s.end - 0.5) continue;
      video.currentTime = s.end;
      toast(`Skipped ${s.category === "selfpromo" ? "self-promo" : "sponsor"} ${fmt(s.start)}–${fmt(s.end)}`, {
        label: "Undo",
        run: () => {
          undone.add(key(s));
          video.currentTime = s.start;
        },
      });
    }
  }, 250);

  // --- per video: SponsorBlock at once, then Claude's reading ---------------
  async function load(id, { fresh = false } = {}) {
    videoId = id;
    result = null;
    segments = [];
    undone.clear();
    if (!fresh) {
      const quick = await send({ type: "quick", id });
      if (videoId !== id) return;
      if (!quick.ok) {
        result = { error: quick.error, offline: quick.offline };
        if (settings.enabled) toast(`SponsorSkip: ${quick.error}`);
        return;
      }
      result = quick.data;
      segments = pick(result);
      if (!quick.data.interim) return announce();
    }
    if (settings.enabled) toast(settings.reader === "local" ? "SponsorSkip: reading the transcript on your GPU…" : "SponsorSkip: Claude is reading the transcript…");
    const full = await send({ type: "analyze", id, reader: settings.reader, fresh });
    if (videoId !== id) return;
    if (!full.ok) {
      if (settings.enabled) toast(`SponsorSkip: ${full.error}`);
      return;
    }
    result = full.data;
    segments = pick(result);
    announce();
  }
  function announce() {
    if (!settings.enabled) return;
    const via = /claude/i.test(result.source) ? "Claude read the transcript" : /GPU/.test(result.source) ? "read on your GPU" : "SponsorBlock";
    const n = segments.length;
    toast(n ? `SponsorSkip: ${n} to skip (${via})` : `SponsorSkip: nothing to skip (${via})`);
  }

  // The popup asks what this tab knows, or to re-read the video.
  api.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg.type === "state") sendResponse({ videoId, result, segments });
    if (msg.type === "reread" && videoId) load(videoId, { fresh: true });
  });

  // YouTube is a single-page app: watch the URL.
  setInterval(() => {
    const id = location.pathname === "/watch" ? new URL(location.href).searchParams.get("v") : null;
    if (id === videoId) return;
    if (!id) {
      videoId = null;
      result = null;
      segments = [];
      return;
    }
    load(id);
  }, 500);
})();
