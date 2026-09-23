# SponsorSkip

A Chrome and Firefox extension that skips sponsor reads in YouTube videos. **Claude reads each video's entire transcript** to find them, on your Claude subscription through Claude Code, with no API key. It doesn't rely on crowd-sourced timestamps.

Built 2026-09-18 for Simon. Personal and local: the program the extension talks to listens on `127.0.0.1` only.

## How it works

1. **The extension** (`extension/`, one codebase, Manifest V3, Chrome and Firefox) watches YouTube. When a video opens, it asks the SponsorSkip program on this PC for the sponsor segments, then skips any the playhead enters, with an **Undo** button. It stays still during YouTube's own ads. The popup shows the connection, what was found on the current video, settings (on/off, reader, and self-promotion skipping: merch, Patreon, memberships, on by default since 0.4.1), and the review list for SponsorBlock. While a video is being read the popup shows a **labelled progress strip**, one coloured bar per step, so a reading that finds nothing still visibly runs to a finish and says **"No sponsor reads in this video"** — a result, not a failure. **The popup updates itself while it is open**, so a reading that finishes with it up fills in on its own, and a review already under way is left alone. **Settings sync with your browser account.**
2. **The program** (`server/`, Node, no npm dependencies, `127.0.0.1:4790`):
   - `GET /quick/:id` answers at once, from the cache or else from [SponsorBlock](https://sponsor.ajay.app)'s community segments marked *interim*, so a sponsor read in the first minute is covered while Claude reads. The SponsorBlock lookup goes by hash prefix, so it never sees the video id.
   - `GET /analyze/:id` fetches the English captions with **yt-dlp**, and **Claude Haiku** reads the transcript through the Claude Code CLI (`claude -p`, no tools, no MCP servers, no hooks). It reads **the first 7 minutes on their own first**, which comes back in about 10 s, so a sponsor read at the start is skipped straight away; then it reads the whole transcript, which is the answer. `GET /progress/:id` is what the page polls meanwhile. The segment edges are then snapped to exact caption lines using the words Claude copies for where each read starts and where the show resumes. Results are cached in `cache/<id>.json`, so a rewatch is instant and free. A video with no English captions falls back to SponsorBlock.
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

**0.8.0: the popup offers four levels** (Simon's decision, 2026-09-22):

1. **Free marker** (`?reader=marker-v3`, `predict.py serve --tier v3`): candidate v3, seven detectors including the community SponsorBlock model, CPU only, about 35-60 s a video. v3 on holdout 4: 67.2% of ad time at 5.5 s of show lost per video.
2. **Free marker + your GPU checks each find**: shown but disabled, not built yet. The design (v3 at a looser threshold, with qwen removing false finds) waits on Simon.
3. **Free marker finds, Claude places each edge** (`?reader=marker-v3-cascade`, `--tier v3-cascade`): v3's regions, then Claude Haiku gets only the lines around each one and says where it starts and ends. If Claude fails, our edge heads are used. Not yet measured on v3's regions (the cascade's 71.2% / 4.6 s was on v2's).
4. **Claude reads the whole video** (`claude`): default.

A saved 0.7.x setting is rewritten once: `marker`, `marker-candidate` and `marker-qwen` become level 1, `marker-cascade` becomes level 3, `local` becomes level 4. The program still accepts the old reader names. A cached reading is only reused for the reader that made it. The rest of this section describes the 0.7.x readers, kept for the record.

- **Claude through Claude Code: default.**
- **Local GPU first** (qwen3:8b via Ollama): the GPU reads, and Claude re-reads whenever the local answer cannot be confirmed — when it misses a SponsorBlock segment, when SponsorBlock has nothing to check it against, or when the GPU found nothing at all. (Until 0.6.0 the check only fired on a *disagreement*, and `[].every()` is true, so on any video SponsorBlock did not already cover the local answer was accepted unverified — exactly the videos this extension exists for.) The model is unloaded right after each video, and it's skipped when the GPU is busy (over 2.5 GB used) or warm (78 °C or more). It isn't the default because it measured far worse (below).

- **Free marker on this PC** (0.7.0, opt-in): no language model at all. `marker/predict.py serve` runs the free tier trained in `marker/` (a linear marker, a context model and start/resume heads) on the CPU in about 10 s, and skips only what it is sure of. On two holdouts of channels it never saw it skipped about half of all ad time (49.6% and 49.3%) for 9.9 s and 7.5 s of real show lost per video; Claude and the GPU read more, the marker costs nothing. If it fails, SponsorBlock's answer is used. Needs the trained bundle: `python marker/predict.py export` writes `marker/data/production/free_tier.pt`.
- **Free marker + your GPU checks each find** (0.7.0, opt-in, `?reader=marker-qwen`): the same marker, with qwen3:8b confirming each region it flags and placing each confirmed read's first line and resume line (`predict.py serve --tier qwen`, about 45 s, qwen unloaded afterwards). On the two holdouts it skipped 63.4% of all ad time, for 19.9 s and 9.9 s of real show lost per video (one video per holdout lost more than a minute). When the GPU is busy or warm it runs as the free marker instead and says so. Live qwen answers were checked against the recorded ones used for grading (15/15 yes/no, 3/3 edges).

Both marker options are new in 0.7.0; they were tested through the program on a second port, but the popup options themselves were not yet tried in a browser.

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

1. **Claude Code**, installed and logged in with your Claude account (run `claude` once).
2. **yt-dlp**: `python -m pip install --user yt-dlp`. Keep it current with `--upgrade`, since YouTube keeps changing.
3. **The program**: double-click `start-hidden.vbs` (no window), or `npm start`. To start it at login, put a shortcut to `start-hidden.vbs` in `shell:startup` (the desktop already has one).
4. **The extension**:
   - **Chrome**: `chrome://extensions` → turn on **Developer mode** → **Load unpacked** → pick the `extension` folder.
   - **Firefox**: `about:debugging#/runtime/this-firefox` → **Load Temporary Add-on** → pick `extension/manifest.json`. This lasts until Firefox restarts. A permanent install needs the add-on signed by Mozilla (free, unlisted): `npx web-ext sign --channel=unlisted` with your addons.mozilla.org API keys.
   - Firefox may ask to allow the extension on youtube.com and 127.0.0.1. Allow both.

## Tests

`npm test`: 11 offline checks of what decides a skip (reply parsing, clamping and merging, chunking, quote snapping at both edges, caption parsing, part splitting for the local reader).
