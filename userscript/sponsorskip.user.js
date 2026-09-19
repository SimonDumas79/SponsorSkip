// ==UserScript==
// @name         SponsorSkip (agent-read)
// @namespace    simondumas.sponsorskip
// @version      0.2.0
// @description  Skips sponsor reads on YouTube. An agent reads the video's whole transcript to find them (local backend on 127.0.0.1:4790).
// @match        https://www.youtube.com/*
// @grant        GM_xmlhttpRequest
// @connect      127.0.0.1
// @run-at       document-idle
// @noframes
// ==/UserScript==

// The page side is deliberately thin: it sends the video id and skips what
// comes back. Fetching the transcript and running the agent happen in the
// backend (yt-dlp + Claude), because YouTube refuses transcript requests
// from scripts in the page.
(() => {
  "use strict";
  const API = "http://127.0.0.1:4790";
  const SKIP = new Set(["sponsor"]); // add "selfpromo" to also skip merch/Patreon plugs

  let videoId = null;
  let segments = [];
  const undone = new Set(); // segments the viewer chose to watch, by "start-end"

  function api(route) {
    return new Promise((resolve, reject) => {
      GM_xmlhttpRequest({
        method: "GET",
        url: API + route,
        timeout: 240000,
        onload: (r) => {
          try {
            const json = JSON.parse(r.responseText);
            r.status >= 400 ? reject(new Error(json.error || `HTTP ${r.status}`)) : resolve(json);
          } catch (e) {
            reject(e);
          }
        },
        onerror: () => reject(new Error("backend not running (start SponsorSkip)")),
        ontimeout: () => reject(new Error("backend timed out")),
      });
    });
  }

  // --- one small status/toast line over the player ---------------------------
  let toastEl = null;
  let toastTimer = null;
  function toast(text, action) {
    const host = document.getElementById("movie_player") || document.body;
    if (!toastEl || !host.contains(toastEl)) {
      toastEl = document.createElement("div");
      toastEl.style.cssText =
        "position:absolute;left:12px;bottom:64px;z-index:60;background:rgba(0,0,0,.78);color:#fff;font:13px/1.4 Roboto,Arial,sans-serif;padding:6px 10px;border-radius:6px;";
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
  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const key = (s) => `${Math.round(s.start)}-${Math.round(s.end)}`;

  // --- skipping --------------------------------------------------------------
  setInterval(() => {
    const video = document.querySelector("video.html5-main-video") || document.querySelector("video");
    if (!video || !segments.length) return;
    if (document.getElementById("movie_player")?.classList.contains("ad-showing")) return; // YouTube's own ads
    const t = video.currentTime;
    for (const s of segments) {
      if (undone.has(key(s)) || !SKIP.has(s.category)) continue;
      if (t >= s.start && t < s.end - 0.5) {
        video.currentTime = s.end;
        toast(`Skipped ${s.category} ${fmt(s.start)}–${fmt(s.end)}`, {
          label: "Undo",
          run: () => {
            undone.add(key(s));
            video.currentTime = s.start;
          },
        });
      }
    }
  }, 250);

  // --- per video: SponsorBlock at once, then the agent's reading -------------
  async function load(id) {
    videoId = id;
    segments = [];
    undone.clear();
    try {
      const quick = await api(`/quick/${id}`);
      if (videoId !== id) return;
      segments = (quick.segments || []).filter((s) => SKIP.has(s.category));
      if (!quick.interim) return announce(quick);
      toast("SponsorSkip: Claude is reading the transcript…");
      const full = await api(`/analyze/${id}`);
      if (videoId !== id) return;
      segments = (full.segments || []).filter((s) => SKIP.has(s.category));
      announce(full);
    } catch (e) {
      if (videoId === id) toast(`SponsorSkip: ${e.message}`);
    }
  }
  function announce(result) {
    const via = result.source === "sponsorblock" ? "SponsorBlock" : "transcript read by Claude";
    const n = segments.length;
    toast(n ? `SponsorSkip: ${n} sponsor segment${n > 1 ? "s" : ""} (${via})` : `SponsorSkip: no sponsors found (${via})`);
  }

  // YouTube is a single-page app: watch the URL instead of relying on its events.
  setInterval(() => {
    const id = location.pathname === "/watch" ? new URL(location.href).searchParams.get("v") : null;
    if (id === videoId) return;
    if (!id) {
      videoId = null;
      segments = [];
      return;
    }
    load(id);
  }, 500);
})();
