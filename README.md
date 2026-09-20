# SponsorSkip

A Chrome and Firefox extension that skips sponsor reads in YouTube videos. **Claude reads each video's entire transcript** to find them, on your Claude subscription through Claude Code, with no API key. It doesn't rely on crowd-sourced timestamps.

Built 2026-09-18 for Simon. Personal and local: the program the extension talks to listens on `127.0.0.1` only.

## How it works

1. **The extension** (`extension/`, one codebase, Manifest V3, Chrome and Firefox) watches YouTube. When a video opens, it asks the SponsorSkip program on this PC for the sponsor segments, then skips any the playhead enters, with an **Undo** button. It stays still during YouTube's own ads. The popup shows the connection, what was found on the current video, settings (on/off, reader, and self-promotion skipping: merch, Patreon, memberships, on by default since 0.4.1), and the review list for SponsorBlock. **The popup updates itself while it is open**, so a reading that finishes with it up fills in on its own, and a review already under way is left alone. **Settings sync with your browser account.**
2. **The program** (`server/`, Node, no npm dependencies, `127.0.0.1:4790`):
   - `GET /quick/:id` answers at once, from the cache or else from [SponsorBlock](https://sponsor.ajay.app)'s community segments marked *interim*, so a sponsor read in the first minute is covered while Claude reads. The SponsorBlock lookup goes by hash prefix, so it never sees the video id.
   - `GET /analyze/:id` fetches the English captions with **yt-dlp**, and **Claude Haiku** reads the whole transcript through the Claude Code CLI (`claude -p`, no tools, no MCP servers, no hooks). The segment edges are then snapped to exact caption lines using the words Claude copies for where each read starts and where the show resumes. Results are cached in `cache/<id>.json`, so a rewatch is instant and free. A video with no English captions falls back to SponsorBlock.
   - Only browser extensions get answers (CORS is echoed for `chrome-extension://` and `moz-extension://` origins only), so no web page can make it spend your Claude usage.

**Why Claude signs in through Claude Code rather than in the browser:** Anthropic doesn't offer a Claude sign-in to third-party extensions, and reusing the claude.ai browser session would break the consumer terms. Claude Code is Anthropic's own client and runs on your subscription, so it is the "log in with Claude" here. Do it once per PC.

**Why the page doesn't fetch the transcript itself:** as of 2026-09-18, YouTube refuses `get_transcript` ("Precondition check failed") and serves empty caption files without a proof-of-origin token, from scripts and even from an automated browser. yt-dlp's maintainers keep up with that.

## Help SponsorBlock (secondary)

When Claude finds a segment SponsorBlock doesn't have, the popup offers it under **Help SponsorBlock** (collapsed by default, since skipping is the point). SponsorBlock bans automated submissions and accepts AI-found timings only after a person previews them ([their rule](https://wiki.sponsor.ajay.app/w/Automating_Submissions)), so:

- each segment has **▶ preview** buttons for both edges (they play from 2 s before to 3 s after the edge, with skipping paused), plus **− / +** nudges of 0.5 s;
- **Submit stays disabled until both edges have been previewed**, and nudging an edge un-previews it;
- one click submits one segment. Nothing is ever sent automatically. A submitted segment is remembered, so it can't be sent twice.

Submissions use a private SponsorBlock user ID the extension creates once and keeps in synced browser storage. Use my own in the popup lets you paste an existing SponsorBlock ID so contributions count toward it. The submission format comes from SponsorBlock's server source (`postSkipSegments.ts`): `videoID`, `userID` (at least 30 characters, private), `userAgent`, `videoDuration`, and `segments: [{segment: [start, end], category, actionType: "skip"}]`.

## Readers (popup setting)

- **Claude through Claude Code: default.**
- **Local GPU first** (qwen3:8b via Ollama): the GPU reads, and Claude re-reads whenever the local answer misses a SponsorBlock segment. The model is unloaded right after each video, and it's skipped when the GPU is busy (over 2.5 GB used) or warm (78 °C or more). It isn't the default because it measured far worse (below).

## Measured (2026-09-18, two Dwarkesh Patel episodes, 6 sponsor reads)

| Reader | Found | Edges | Time per episode | Cost |
|---|---|---|---|---|
| Claude Haiku | 6 of 6 | starts within 0–2 s of SponsorBlock; ends within 0–4 s on one episode, −6 to +10 s on the other | 30–90 s | ~$0.06–0.09 of subscription usage |
| qwen3:8b, ~30k-char parts | 2 of 6 whole | one start 44 s late | 55–91 s of GPU | free |
| qwen3:8b, ~12k-char parts | 1 of 6 whole | otherwise only the closing call-to-action line, leaving 40–60 s of each ad | 56–103 s of GPU | free |

The extension was tested in Firefox 156 on the first episode: it connected, loaded 3 segments, skipped 21:10 → 21:58, Undo returned to 20:55 and stayed, and 39:10 skipped to 40:23.

## Install (once per PC)

1. **Claude Code**, installed and logged in with your Claude account (run `claude` once).
2. **yt-dlp**: `python -m pip install --user yt-dlp`. Keep it current with `--upgrade`, since YouTube keeps changing.
3. **The program**: double-click `start-hidden.vbs` (no window), or `npm start`. To start it at login, put a shortcut to `start-hidden.vbs` in `shell:startup` (the desktop already has one).
4. **The extension**:
   - **Chrome**: `chrome://extensions` → turn on **Developer mode** → **Load unpacked** → pick the `extension` folder.
   - **Firefox**: `about:debugging#/runtime/this-firefox` → **Load Temporary Add-on** → pick `extension/manifest.json`. This lasts until Firefox restarts. A permanent install needs the add-on signed by Mozilla (free, unlisted): `npx web-ext sign --channel=unlisted` with your addons.mozilla.org API keys.
   - Firefox may ask to allow the extension on youtube.com and 127.0.0.1. Allow both.

## Tests

`npm test`: 11 offline checks of what decides a skip (reply parsing, clamping and merging, chunking, quote snapping at both edges, caption parsing, part splitting for the local reader).
