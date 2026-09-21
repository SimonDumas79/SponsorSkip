const api = globalThis.browser ?? globalThis.chrome;
const DEFAULTS = { enabled: true, skipSelfpromo: true, reader: "claude" };
const $ = (id) => document.getElementById(id);
const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

let readerSetting = DEFAULTS.reader;

// The steps a reading goes through, in order, for each reader setting. Each
// gets its own colour so the strip reads at a glance, and every reading runs
// visibly to a finish -- so "found nothing" never looks like "it broke".
const STEP_ORDER = {
  claude: ["captions", "opening", "full"],
  local: ["captions", "gpu", "verify", "full"],
  marker: ["captions", "marker"],
  "marker-qwen": ["captions", "markerQwen"],
};
const STEP = {
  captions: { label: "Fetching captions", color: "#7aa2f7" },
  opening: { label: "Reading the opening", color: "#7dcfff" },
  full: { label: "Reading the whole transcript", color: "#9ece6a" },
  gpu: { label: "Reading on your GPU", color: "#bb9af7" },
  verify: { label: "Claude checking that answer", color: "#e0af68" },
  marker: { label: "Reading with the free marker", color: "#73daca" },
  markerQwen: { label: "The marker finds, your GPU checks", color: "#2ac3de" },
};

function renderSteps(reading) {
  const box = $("steps");
  if (!reading?.step) {
    box.hidden = true;
    box.replaceChildren();
    return;
  }
  const order = STEP_ORDER[readerSetting] ?? STEP_ORDER.claude;
  const keys = order.includes(reading.step) ? order : [...order, reading.step];
  const at = keys.indexOf(reading.step);
  box.hidden = false;
  box.replaceChildren(
    ...keys.map((key, i) => {
      const row = document.createElement("div");
      row.className = `step ${i < at ? "done" : i === at ? "active" : "todo"}`;
      row.style.setProperty("--c", STEP[key]?.color ?? "var(--accent)");
      const track = document.createElement("span");
      track.className = "track";
      const fill = document.createElement("span");
      fill.className = "fill";
      track.append(fill);
      const lbl = document.createElement("span");
      lbl.className = "lbl";
      lbl.textContent = (key === reading.step && reading.label) || STEP[key]?.label || key;
      row.append(track, lbl);
      return row;
    }),
  );
}

async function loadSettings() {
  const s = { ...DEFAULTS, ...(await api.storage.sync.get(DEFAULTS)) };
  readerSetting = s.reader;
  $("enabled").checked = s.enabled;
  $("skipSelfpromo").checked = s.skipSelfpromo;
  for (const r of document.querySelectorAll('input[name="reader"]')) r.checked = r.value === s.reader;
}
$("enabled").addEventListener("change", (e) => api.storage.sync.set({ enabled: e.target.checked }));
$("skipSelfpromo").addEventListener("change", (e) => api.storage.sync.set({ skipSelfpromo: e.target.checked }));
for (const r of document.querySelectorAll('input[name="reader"]'))
  r.addEventListener("change", (e) => {
    readerSetting = e.target.value;
    api.storage.sync.set({ reader: e.target.value });
  });

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

// The popup is a live view, not a snapshot: a reading usually finishes while
// it is open, and it should say so without being reopened.
//
// Only the tab's own state is polled. It is a message to the content script,
// which already holds the latest result, so it costs nothing. /health is
// deliberately NOT polled: it shells out to `claude --version` and samples the
// GPU and the CPU, so a tick of it every half second would spawn a process a
// second. The connection line stays a one-shot from when the popup opened.
const POLL_MS = 500;
let lastSig = null;
let sbBuiltFor = null; // the video the Help SponsorBlock rows were built for

// Everything that changes what the panel should say. Redrawing on every tick
// instead would throw away a half-finished SponsorBlock review: which edges
// have been previewed, and any nudges, live in the DOM of those rows.
const stateSig = (state) =>
  JSON.stringify([
    state?.videoId ?? null,
    state?.result?.error ?? null,
    state?.result?.interim ?? null,
    state?.result?.source ?? null,
    state?.result?.seconds ?? null,
    state?.result?.reason ?? null,
    state?.result?.partial ?? null,
    state?.result?.part ?? null,
    state?.reading?.step ?? null,
    (state?.result?.segments ?? []).map((s) => [s.start, s.end, s.category]),
  ]);

async function pollVideo() {
  const [tab] = await api.tabs.query({ active: true, currentWindow: true });
  let state = null;
  if (tab?.url?.startsWith("https://www.youtube.com/watch")) {
    try {
      state = await api.tabs.sendMessage(tab.id, { type: "state" });
    } catch {
      state = null; // the page hasn't loaded the script yet
    }
  }
  const sig = stateSig(state);
  if (sig === lastSig) return;
  lastSig = sig;
  if (!state?.videoId) {
    // Navigated off a watch page, or the script isn't up yet.
    $("video").hidden = true;
    $("sb").hidden = true;
    sbBuiltFor = null;
    return;
  }
  renderVideo(tab, state);
}

function renderVideo(tab, state) {
  $("video").hidden = false;
  const r = state.result;
  renderSteps(state.reading);
  // Say the answer in words. A reading that finds nothing is a RESULT, and
  // used to be indistinguishable from a reading that failed.
  const settled = r && !r.error && !r.interim && !r.partial;
  const found = (r?.segments ?? []).length;
  $("verdict").hidden = !settled;
  if (settled) {
    $("verdict").className = found ? "verdict" : "verdict none";
    $("verdict").textContent = found ? `${found} to skip in this video` : "No sponsor reads in this video";
  }
  $("video-source").textContent = !r
    ? "Reading…"
    : r.error
      ? r.error
      : r.interim
        ? "Using SponsorBlock while Claude reads…"
        : r.partial
          ? `Reading… part ${r.part} of ${r.parts}, ${(r.segments ?? []).length} found so far`
          : `${r.source}${r.seconds ? `, ${r.seconds}s` : ""}${r.reason ? ` (${r.reason})` : ""}`;
  $("segments").replaceChildren(
    ...(r?.segments ?? []).map((s) => {
      const li = document.createElement("li");
      const when = document.createElement("span");
      when.className = "when";
      when.textContent = `${s.category === "selfpromo" ? "Self-promo" : "Sponsor"} ${fmt(s.start)}–${fmt(s.end)}`;
      li.append(when);
      const q = document.createElement("span");
      q.className = "muted small";
      q.textContent = s.quote ? `“${s.quote.slice(0, 32)}…”` : "";
      li.append(q);
      return li;
    }),
  );
  // Built once per video: rebuilding would reset the previews that SponsorBlock
  // requires before a segment can be submitted. sbBuiltFor is set before the
  // await inside, so overlapping ticks can't build it twice.
  // Not while r.partial: a half-read video would offer segments for review
  // that the rest of the reading may still extend.
  if (r && !r.error && !r.interim && !r.partial && !/sponsorblock/i.test(r.source || "") && sbBuiltFor !== state.videoId) {
    sbBuiltFor = state.videoId;
    reviewForSponsorBlock(tab, state, r.segments || []);
  }
  $("reread").onclick = () => {
    api.tabs.sendMessage(tab.id, { type: "reread" });
    $("video-source").textContent = "Reading again…";
    $("segments").replaceChildren();
    $("sb").hidden = true;
    sbBuiltFor = null;
    lastSig = null; // redraw from whatever the page reports next tick
  };
}

// --- Help SponsorBlock: human-reviewed submissions only -----------------------
// SponsorBlock forbids automated submissions; AI-found timings are allowed
// only after a person previews them. So Submit stays disabled until BOTH
// edges were previewed, a nudge un-previews that edge, and each segment is
// sent on its own click.
const fmt1 = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
const overlaps = (a, b) => a.start < b.end && a.end > b.start;

async function reviewForSponsorBlock(tab, state, ours) {
  const lookup = await api.runtime.sendMessage({ type: "sbLookup", id: state.videoId });
  if (!lookup.ok) return;
  const key = `sb-submitted-${state.videoId}`;
  const submitted = new Set((await api.storage.local.get({ [key]: [] }))[key]);
  const missing = ours.filter((s) => !lookup.data.some((c) => overlaps(s, c)));
  if (!missing.length) return;
  $("sb").hidden = false;
  const todo = missing.filter((s) => !submitted.has(`${Math.round(s.start)}`)).length;
  $("sb-summary").textContent = todo
    ? `SponsorBlock doesn't have ${todo === 1 ? "1 of these" : `${todo} of these`} yet. Checking ${todo === 1 ? "it" : "them"} helps everyone who uses it.`
    : "Everything here that SponsorBlock was missing has been submitted. Thanks.";
  $("sb-toggle").onclick = () => {
    $("sb-body").hidden = !$("sb-body").hidden;
    $("sb-toggle").textContent = $("sb-body").hidden ? "Review" : "Hide";
  };
  $("sb-list").replaceChildren(...missing.map((seg) => reviewRow(tab, state, seg, submitted, key)));
  $("sb-id-change").onclick = () => {
    $("sb-id-edit").hidden = false;
    $("sb-id-help").hidden = false;
    $("sb-id-input").focus();
  };
  $("sb-id-save").onclick = async () => {
    const v = $("sb-id-input").value.trim();
    if (v.length < 30) {
      $("sb-id-input").value = "";
      $("sb-id-input").placeholder = "Must be at least 30 characters";
      return;
    }
    await api.storage.sync.set({ sbUserId: v });
    $("sb-id-edit").hidden = true;
    $("sb-id-text").textContent = "Submitting under your own SponsorBlock ID.";
  };
}

function reviewRow(tab, state, seg, submitted, key) {
  const li = document.createElement("li");
  const done = () => submitted.has(`${Math.round(seg.start)}`);
  const edit = { start: seg.start, end: seg.end };
  const seen = { start: false, end: false };
  const title = document.createElement("div");
  title.textContent = `${seg.category === "selfpromo" ? "Self-promo" : "Sponsor"}${seg.quote ? ` · “${seg.quote.slice(0, 38)}…”` : ""}`;
  const submit = Object.assign(document.createElement("button"), { className: "sb-submit", textContent: "Submit to SponsorBlock" });
  const status = Object.assign(document.createElement("span"), { className: "sb-status" });
  const refresh = () => (submit.disabled = !(seen.start && seen.end) || edit.end - edit.start < 1 || done());

  const edge = (which, label) => {
    const row = Object.assign(document.createElement("div"), { className: "edge" });
    const lbl = Object.assign(document.createElement("span"), { className: "lbl", textContent: label });
    const t = Object.assign(document.createElement("span"), { className: "t", textContent: fmt1(edit[which]) });
    const play = Object.assign(document.createElement("button"), { textContent: "▶ preview", title: "Plays 5 s starting exactly at this time" });
    const nudge = (d) => {
      edit[which] = Math.max(0, +(edit[which] + d).toFixed(1));
      t.textContent = fmt1(edit[which]);
      seen[which] = false; // a changed edge has to be watched again
      play.classList.remove("seen");
      refresh();
    };
    const minus = Object.assign(document.createElement("button"), { textContent: "−", title: "0.5 s earlier" });
    const plus = Object.assign(document.createElement("button"), { textContent: "+", title: "0.5 s later" });
    minus.onclick = () => nudge(-0.5);
    plus.onclick = () => nudge(0.5);
    play.onclick = async () => {
      const r = await api.tabs.sendMessage(tab.id, { type: "preview", at: edit[which] });
      if (r?.ok) {
        seen[which] = true;
        play.classList.add("seen");
        refresh();
      }
    };
    row.append(lbl, minus, t, plus, play);
    return row;
  };

  submit.onclick = async () => {
    submit.disabled = true;
    status.className = "sb-status";
    status.textContent = "Sending…";
    const r = await api.runtime.sendMessage({ type: "sbSubmit", id: state.videoId, start: edit.start, end: edit.end, category: seg.category, duration: state.duration });
    if (r.ok) {
      submitted.add(`${Math.round(seg.start)}`);
      await api.storage.local.set({ [key]: [...submitted] });
      status.className = "sb-status ok";
      status.textContent = "Submitted. Thank you!";
    } else {
      status.className = "sb-status bad";
      status.textContent = r.error;
      refresh();
    }
  };
  if (done()) {
    status.className = "sb-status ok";
    status.textContent = "Already submitted";
  }
  refresh();
  const actions = document.createElement("div");
  actions.append(submit, status);
  li.append(title, edge("start", "Start"), edge("end", "End"), actions);
  return li;
}

loadSettings();
connection();
pollVideo();
setInterval(pollVideo, POLL_MS);
