# SponsorSkip

**Do you hate advertisements? Me too!**

SponsorSkip is a mix of multiple small, directed models used to mark and skip sponsored content on YouTube.

It is a Chrome and Firefox extension plus a small program on your own PC. **By default the free marker reads each video's transcript** (level 1: seven small models, CPU only, no language model, nothing leaves your PC). Every level above it is opt-in, because those are the ones that cost you something: your GPU's time, or the plan of whichever AI you pick (Claude Code, Codex, Gemini, Ollama, or an API). It doesn't rely on crowd-sourced timestamps. The program listens on `127.0.0.1` only.

The level-1 model files, including the self-promotion detector (about 250 MB, CC BY-NC-SA 4.0), are not in the repository: the program downloads them from the GitHub release named in `models.json` the first time it starts, and checks them against its sha256 (`server/models.mjs`). The first level-1 reading also fetches about 1 GB of base models from Hugging Face.

**Setting it up with an AI agent:** point your AI agent (Claude Code, Codex, Gemini CLI or similar) at `SETUP_WITH_AN_AI_AGENT.md`. Licence: GPL-3.0 for the code, CC BY-NC-SA 4.0 for the models (`NOTICE.md`). Privacy: `PRIVACY.md`.

## How it works

1. **The extension** (`extension/`, one codebase, Manifest V3, Chrome and Firefox) watches YouTube. When a video opens, it asks the SponsorSkip program on this PC for the sponsor segments, then skips any the playhead enters, with an **Undo** button. It stays still during YouTube's own ads. The popup shows the connection, what was found on the current video, settings (on/off, reader, and self-promotion skipping: merch, Patreon, memberships, on by default since 0.4.1), and the review list for SponsorBlock. While a video is being read the popup shows a **labelled progress strip**, one coloured bar per step, so a reading that finds nothing still visibly runs to a finish and says **"No sponsor reads in this video"** — a result, not a failure. **The popup updates itself while it is open**, so a reading that finishes with it up fills in on its own, and a review already under way is left alone. **Settings sync with your browser account.**
2. **The program** (`server/`, Node, no npm dependencies, `127.0.0.1:4790`):
   - `GET /quick/:id` answers at once, from the cache or else from [SponsorBlock](https://sponsor.ajay.app)'s community segments marked *interim*, so a sponsor read in the first minute is covered while the reader works. The SponsorBlock lookup goes by hash prefix, so it never sees the video id.
   - `GET /analyze/:id` fetches the English captions with **yt-dlp**, and the chosen reader reads the transcript: the free marker by default, or your AI (below) at levels 3 and 4. With Claude Code that is `claude -p` with no tools, no MCP servers and no hooks. It reads **the first 7 minutes on their own first**, which comes back in about 10 s, so a sponsor read at the start is skipped straight away; then it reads the whole transcript, which is the answer. `GET /progress/:id` is what the page polls meanwhile. The segment edges are then snapped to exact caption lines using the words the AI copies for where each read starts and where the show resumes. If the resume words can't be found, a segment longer than 180 s is cut to 180 s (a runaway end, not a real read: across 494 labelled videos no read that ends a video runs past 160 s), and if that cut would leave 30 s or less of the video, the skip runs to the end instead. Results are cached in `cache/<id>.json`, so a rewatch is instant and free. A video with no English captions falls back to SponsorBlock.
   - **Only the extension can use it** (0.10.1). Every route needs an `x-sponsorskip: 1` header, which a web page cannot send without a preflight, and preflights are answered for extension origins only; `<img>` and no-cors requests cannot carry it at all. So no website can start a reading or spend your AI plan. The settings routes can never set a custom command, and a stored API key is dropped when the provider or URL changes, so it is only ever sent where you entered it. **What remains:** another extension you have installed could still start readings on your plan, and could point your readings at a different AI endpoint (your key is dropped when the URL changes, but the caption text would go to the new one; the popup's label shows which). It could not run commands or read your key.


**Your AI** (levels 3 and 4 only; the default needs none). Pick it in the popup under **Manage AI model**, then **Save** and **Test it**. The choice is kept by the program in `ai-config.json` (gitignored), not in the browser.

| your AI | what it costs you | where the caption text goes |
|---|---|---|
| Claude Code (`claude -p`), the default choice, Haiku | your Claude plan, no key | Anthropic |
| Codex CLI (`codex exec`, read-only sandbox) | your ChatGPT plan or OpenAI login | OpenAI |
| Gemini CLI (`gemini -p`) | your Google account | Google |
| Ollama (local, default qwen3:8b) | free, your GPU or CPU | nowhere: it stays on your PC |
| a custom command (prompt on stdin, answer on stdout) | whatever that command costs | wherever that command sends it |
| an OpenAI-compatible API (URL + model + your key) | billed per token by that provider | that URL |

Only Claude Haiku's accuracy has been measured; the popup says so for every other choice. Codex and Gemini are wired in but untested. One more difference: Claude Code is run with no tools at all, while the Codex and Gemini CLIs keep their own (Codex in a read-only sandbox can still read files). A transcript is text written by whoever uploaded the video, so with those two an unusual transcript could in principle steer the agent; if that bothers you, use Claude Code, Ollama or an API endpoint, which only ever answer. **A custom command can only be set by editing `ai-config.json` by hand**; the popup shows it read-only, so nothing that talks to the program can make it run a command.

**Why Claude signs in through Claude Code rather than in the browser:** Anthropic doesn't offer a Claude sign-in to third-party extensions, and reusing the claude.ai browser session would break the consumer terms. Claude Code is Anthropic's own client and runs on your subscription, so it is the "log in with Claude" here. Do it once per PC.

**Why the page doesn't fetch the transcript itself:** as of 2026-09-18, YouTube refuses `get_transcript` ("Precondition check failed") and serves empty caption files without a proof-of-origin token, from scripts and even from an automated browser. yt-dlp's maintainers keep up with that.

## Help SponsorBlock (secondary)

When Claude finds a segment SponsorBlock doesn't have, the popup offers it under **Help SponsorBlock** (collapsed by default, since skipping is the point). SponsorBlock bans automated submissions and accepts AI-found timings only after a person previews them ([their rule](https://wiki.sponsor.ajay.app/w/Automating_Submissions)), so:

- each segment has **▶ preview** buttons for both edges (they play 5 s starting exactly at the edge, with skipping paused, so what you hear is what that number claims), plus **− / +** nudges of 0.5 s;
- **Submit stays disabled until both edges have been previewed**, and nudging an edge un-previews it;
- one click submits one segment. Nothing is ever sent automatically. A submitted segment is remembered, so it can't be sent twice.

Submissions use a private SponsorBlock user ID the extension creates once and keeps in synced browser storage. Use my own in the popup lets you paste an existing SponsorBlock ID so contributions count toward it. The submission format comes from SponsorBlock's server source (`postSkipSegments.ts`): `videoID`, `userID` (at least 30 characters, private), `userAgent`, `videoDuration`, and `segments: [{segment: [start, end], category, actionType: "skip"}]`.

## Readers (popup setting)

**The popup offers four levels** (0.8.0, 2026-09-22). **Level 1 is the default since 0.9.0 (2026-09-24); levels 2-4 are opt-in**, since each costs the user GPU time or AI usage. A setting nobody chose falls to level 1; a level someone picked is kept.

1. **Free marker** (`?reader=marker-v3`, `predict.py serve --tier v3`): candidate v3, seven detectors including the community SponsorBlock model, CPU only, about 35-60 s a video. On holdout 5 (46 fresh videos from 46 channels, graded once): **62.7% of ad time at 4.7 s of show lost per video** (holdout 4: 67.2% at 5.5 s).
2. **Free marker + your GPU checks each find**: not built yet, and not shown in the popup since 0.9.1 (which also drops the level numbers from the popup; they stay here and in the code). The design (v3 at a looser threshold, with qwen removing false finds) is not decided yet.
3. **Free marker finds, Claude places each edge** (`?reader=marker-v3-cascade`, `--tier v3-cascade`): v3's regions, then Claude Haiku gets only the lines around each one and says where it starts and ends. If Claude fails, our edge heads are used. Measured on holdout 5: 61.9% of ad time at 7.5 s lost, no better than level 1 alone, because only 8 of Claude's 62 answers landed inside the window where they are accepted.
4. **Claude reads the whole video** (`claude`): the most ad time, and opt-in. It was the default until 0.9.0. Holdout 4: 84.3% of ad time at 10.7 s lost per video. Holdout 5: 73.2% at 28.7 s, most of that lost show from a few segments whose end ran on for minutes, which is what the 180 s limit on unconfirmed ends (0.10.1, above) now stops.

A saved 0.7.x setting is rewritten once: `marker`, `marker-candidate` and `marker-qwen` become level 1, `marker-cascade` becomes level 3, `local` becomes level 4. The program still accepts the old reader names. A cached reading is only reused for the reader that made it. The 0.7.x readers (the free tier and the qwen tier) are measured in `marker/README.md`; their full description is in this file's git history.

## Measured (2026-09-18, two Dwarkesh Patel episodes, 6 sponsor reads)

| Reader | Found | Edges | Time per episode | Cost |
|---|---|---|---|---|
| Claude Haiku | 6 of 6 | starts within 0–2 s of SponsorBlock; ends within 0–4 s on one episode, −6 to +10 s on the other | 30–90 s | ~$0.06–0.09 of subscription usage |
| qwen3:8b, ~30k-char parts | 2 of 6 whole | one start 44 s late | 55–91 s of GPU | free |
| qwen3:8b, ~12k-char parts | 1 of 6 whole | otherwise only the closing call-to-action line, leaving 40–60 s of each ad | 56–103 s of GPU | free |

## Measured again (2026-09-20, same 6 reads): reading a WINDOW beats reading the whole transcript

Haiku read a 62-minute episode and placed one sponsor read at **1616 s instead of 3616 s** — it quoted the read correctly and mistyped the number, which appears nowhere in the prompt. That would skip a minute of the show and play the sponsor in full. The read sat 93% of the way through an 82,000-character prompt.

So each read was handed over as a **window** instead (90 s before it, 60 s after, 43–51 caption lines), with the answer given as a **line index** rather than a timestamp — which makes an error of that kind structurally impossible. Run it with `node bench/edge-windows.mjs`.

| Reader, on a window | Found | Start error (median / worst) | Starts within 5 s | End error (median) | Per window |
|---|---|---|---|---|---|
| Claude Haiku | 6 of 6 | **0.6 s / 2.2 s** | **6 of 6** | 0.4 s | 7 s |
| qwen3:8b | 6 of 6 | 55.7 s / 69 s | 2 of 6 | 7.8 s | 11 s |

**Claude on a window beats Claude on the whole transcript** (worst start error 2.2 s, against a 33-minute miss), and costs less.

**qwen3:8b is not usable for edges, and the answer format was not the problem.** Its start errors are bimodal — `+63, +5.2, +55.7, +0.6, +69, +2.1` — so it either nails the start or lands a minute late, which is the same closing-call-to-action failure as 09-18. Handing it the ad whole and centred changed nothing, because it doesn't recognise where a read *begins*. Detection it can do; edges it cannot.

The design this points to: a cheap **marker** finds candidate regions (word patterns now, a trained classifier later), and Claude verifies each one in a window, answering with line indices. Start-line accuracy is the metric a trained detector has to beat.

The extension was tested in Firefox 156 on the first episode: it connected, loaded 3 segments, skipped 21:10 → 21:58, Undo returned to 20:55 and stayed, and 39:10 skipped to 40:23.

## Install (once per PC)

1. **Optional, for levels 3 and 4 only: your AI.** Sign in to its CLI (Claude Code, Codex, Gemini), or set up Ollama or an API key, then pick it under **Manage AI model** in the popup.
2. **yt-dlp**: `python -m pip install --user yt-dlp`. Keep it current with `--upgrade`, since YouTube keeps changing.
3. **The program**: double-click `start-hidden.vbs` (no window), or `npm start`. To start it at login, put a shortcut to `start-hidden.vbs` in `shell:startup`.
4. **The extension**:
   - **Chrome**: `chrome://extensions` → turn on **Developer mode** → **Load unpacked** → pick the `extension` folder.
   - **Firefox**: `about:debugging#/runtime/this-firefox` → **Load Temporary Add-on** → pick `extension/manifest.json`. This lasts until Firefox restarts. For a permanent install without Mozilla signing, use Firefox Developer Edition or Nightly: set `xpinstall.signatures.required` to `false` in `about:config`, zip the `extension` folder's contents, and install the zip from `about:addons` → gear → **Install Add-on From File**. (On regular Firefox a permanent install needs the add-on signed by Mozilla: `npx web-ext sign --channel=unlisted` with addons.mozilla.org API keys.)
   - Firefox may ask to allow the extension on youtube.com and 127.0.0.1. Allow both.

## Tests

`npm test`: 29 offline checks, most of them of what decides a skip (reply parsing, clamping and merging, quote snapping at both edges, the 180 s limit and the end-of-video rule, caption file choice, transcript chunking and part splitting for the local reader).
