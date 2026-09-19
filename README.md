# SponsorSkip

Skips sponsor reads in YouTube videos. **An agent reads the video's entire transcript** to find them, rather than relying on crowd-sourced timestamps.

Built 2026-09-18 for Simon. Personal and local: the backend listens on `127.0.0.1` only.

## How it works

1. A **userscript** (Violentmonkey, in Firefox) watches the YouTube page. When a video opens, it sends the video id to the backend. That's all it does in the page.
2. The **backend** (`server/server.mjs`, Node, no npm dependencies):
   - `GET /quick/:id` answers at once with the cached result, or with [SponsorBlock](https://sponsor.ajay.app)'s community segments marked *interim*, so a sponsor read in the first minute is covered while the agent works. The lookup goes by hash prefix, so SponsorBlock never sees the video id.
   - `GET /analyze/:id` fetches the English captions with **yt-dlp**, has **Claude Haiku** (`claude -p`, on the local Claude Code login: no tools, no MCP, no hooks) read the whole transcript, and returns the segments. The segment boundaries are then snapped to exact caption lines using the quotes the agent copies for where each read starts and where the show resumes. The result is cached in `cache/<id>.json`, so a rewatch is instant and free. A video with no English captions falls back to SponsorBlock.
3. The userscript skips any `sponsor` segment the playhead enters, with an **Undo** button. It stays still during YouTube's own ads.

**Why not read the transcript in the page?** As of 2026-09-18, YouTube refuses `get_transcript` ("Precondition check failed") and returns empty caption files without a proof-of-origin token, both from scripts and from an automated browser. yt-dlp's maintainers keep up with that.

## Measured (2026-09-18, Dwarkesh Patel episodes)

| Episode | Transcript | Agent time | Cost | Claude vs SponsorBlock |
|---|---|---|---|---|
| OpenAI researcher on agent swarms (1h20) | 993 lines | ~40–90 s | ~$0.06–0.09 | same 3 reads; starts within 1 s, ends within 0–4 s |
| Ajeya Cotra (2h20) | 1,564 lines | 30 s | ~$0.08 | same 3 reads; starts within 1–2 s, ends −6 s to +10 s |

## Install (once per machine)

1. `python -m pip install --user yt-dlp` (keep it current: `--upgrade`, since YouTube keeps changing).
2. The Claude Code CLI must be installed and logged in (`claude` on PATH).
3. Start the backend: `npm start`, or double-click `start-hidden.vbs` (no window). The desktop has a Startup-folder shortcut to `start-hidden.vbs`, so it runs at login.
4. In Firefox, install the **Violentmonkey** add-on, then open <http://127.0.0.1:4790/sponsorskip.user.js> and click **Install**.

## Settings

- Also skip merch and Patreon plugs: add `"selfpromo"` to `SKIP` at the top of the userscript.
- Agent model: `SPONSORSKIP_MODEL=sonnet` (default `haiku`). Port: `SPONSORSKIP_PORT` (default 4790; change the userscript's `API` too).
- Re-analyse a video: delete `cache/<id>.json`.

## Tests

`npm test`: offline checks of the parts that decide what gets skipped (reply parsing, clamping and merging, chunking, quote snapping, caption parsing).
