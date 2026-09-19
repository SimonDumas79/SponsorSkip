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
  if (r && !r.error && !r.interim && !/sponsorblock/i.test(r.source || "")) reviewForSponsorBlock(tab, state, r.segments || []);
  $("reread").onclick = () => {
    api.tabs.sendMessage(tab.id, { type: "reread" });
    $("video-source").textContent = "Reading again…";
    $("segments").replaceChildren();
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
  $("sb-id-change").onclick = () => ($("sb-id-edit").hidden = false);
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
    const play = Object.assign(document.createElement("button"), { textContent: "▶ preview", title: "Plays from 2 s before to 3 s after this edge" });
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
thisVideo();
