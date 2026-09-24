# Setting up SponsorSkip with an AI agent

**For the person:** open your AI coding agent (Claude Code, or any agent that can run commands on your
computer) in the folder where you want SponsorSkip, and tell it:

> Read SETUP_WITH_AN_AI_AGENT.md from the SponsorSkip repository and set it up for me.

Everything below is written for the agent.

---

## For the agent: what you are setting up

SponsorSkip skips sponsor reads and other promotion in YouTube videos. It has two parts:

1. **A browser extension** (Chrome or Firefox), in `extension/`. It shows what will be skipped and does the skipping.
2. **A program that runs on this computer**, in `server/`: Node, no npm dependencies, listening on
   `http://127.0.0.1:4790` only. It fetches captions with yt-dlp and reads them at the level the person
   picks in the extension's popup:

| level | what reads the video | needs | cost |
|---|---|---|---|
| 1 | the free marker (seven small models, CPU only) (**the default**) | Python packages + the model files | free |
| 2 | the free marker + a local GPU model checking it | not built yet; not shown in the popup | — |
| 3 | the free marker finds, Claude places each read's start and end | level 1 + Claude Code | a little of the person's Claude plan |
| 4 | Claude reads the whole transcript (opt-in) | Claude Code | more of the person's Claude plan |

Claude is used through the **person's own Claude Code login** (`claude -p` on this computer). There is no
API key, and nothing is billed except the person's own Claude plan.

## Rules while you work

- **Ask before installing anything**, and say what it is and how big. Python + PyTorch for level 1 is
  several GB.
- **Never log in for the person.** Claude Code sign-in and browser steps are theirs. Tell them exactly
  what to click.
- **Never submit anything to SponsorBlock.** The extension's "Help SponsorBlock" card is for people only.
  SponsorBlock bans automated submissions.
- **Ask before a test that reads a video with Claude.** It uses their plan.
- Keep the program listening on `127.0.0.1` only. Do not expose it to the network.

## Steps

### 1. Check what is already here

```sh
node --version          # need 18 or newer (fetch is built in)
python --version        # need 3.10 or newer (only for yt-dlp, and for levels 1 and 3)
claude --version        # Claude Code, needed for levels 3 and 4
```

If `claude` is missing, point the person to https://claude.com/claude-code. Once it is installed,
they run `claude` once and log in. Wait for them to confirm before going on.

### 2. Get the code

```sh
git clone https://github.com/SimonDumas79/SponsorSkip.git
cd SponsorSkip
```

(The repository is private while it is being prepared. If the clone fails, ask the person for access or a copy.)

### 3. Install yt-dlp (captions)

```sh
python -m pip install --user --upgrade yt-dlp
python -m yt_dlp --version
```

YouTube changes often. When skipping stops working, **upgrading yt-dlp is the first fix.**

### 4. Levels 1 and 3 only: the free marker

Skip this step if the person only wants level 4.

```sh
python -m pip install --user numpy torch transformers sentence-transformers model2vec scikit-learn
```

A CPU-only PyTorch is enough; the marker never uses the GPU. **The model files are not in the
repository.** The program downloads them itself the first time it starts (about 200 MB, from the GitHub
release named in `models.json`, checked against its sha256, unpacked into `marker/data/production/`).
Level 1's first reading also fetches its base models from Hugging Face (about 1 GB more, cached in
`~/.cache/huggingface`), so the first video is slow and needs the network. Tell the person the whole
one-time cost before you start the program: several GB of Python packages, about 200 MB of SponsorSkip's
models, and about 1 GB from Hugging Face. Until it finishes,
level 1 falls back to SponsorBlock's community segments. `GET http://127.0.0.1:4790/health` shows
`models.state`: `downloading`, `ready`, or `failed` with the reason (a failed download is retried at the
next start). The files are CC BY-NC-SA 4.0 (`NOTICE.md`).

### 5. Start the program

- **Windows:** double-click `start-hidden.vbs` (it runs with no window, logging to `backend.log`). To
  start it at every login, ask the person, then put a shortcut to `start-hidden.vbs` in the folder that
  `shell:startup` opens.
- **macOS / Linux:** `npm start`, kept running in the background (a login item, launchd, or systemd
  user service, whichever the person prefers).

Check it:

```sh
curl http://127.0.0.1:4790/health
```

Expect `{"ok":true,"agents":[...]}`. The `claude-haiku (Claude Code)` entry must say `"ready":true` for
levels 3 and 4. If the port is taken, stop the other program. The extension always calls 4790.

### 6. Load the extension (the person does this, you guide them)

- **Chrome:** open `chrome://extensions`, turn on **Developer mode** (top right), click **Load unpacked**,
  and choose the `extension` folder inside SponsorSkip.
- **Firefox:** open `about:debugging#/runtime/this-firefox`, click **Load Temporary Add-on**, and choose
  `extension/manifest.json`. This lasts until Firefox restarts; a permanent install needs Mozilla signing.
  If Firefox asks for access to youtube.com and 127.0.0.1, allow both.

Most agents cannot open `chrome://` or `about:` pages, so give the person these clicks rather than trying.

### 7. Check it works

A free check that needs no Claude:

```sh
curl http://127.0.0.1:4790/quick/dQw4w9WgXcQ
```

Expect JSON with a `segments` list (SponsorBlock's community answer, possibly empty). Then ask the person
to open any YouTube video. A "SponsorSkip: ..." message should appear over the player within a few
seconds. For a full test with Claude, ask first (it uses their plan), then:

```sh
curl "http://127.0.0.1:4790/analyze/<videoId>?reader=claude"
```

### 8. Choosing a level

The person picks it in the extension's popup (click the SponsorSkip icon). Level 4 is the default and
the most accurate. Level 1 is free and never leaves the computer. Readings are cached per video in
`cache/`, and switching level reads the video again.

## When something is wrong

| symptom | first thing to try |
|---|---|
| nothing appears over the player | reload the extension; check `/health`; check that `backend.log` shows a line for the video |
| "couldn't fetch captions" | `python -m pip install --user --upgrade yt-dlp`; if the log says 429, YouTube is rate-limiting, so wait an hour |
| level 3 or 4 falls back to SponsorBlock | run `claude` in a terminal; the person may need to log in again or has hit their plan's limit |
| `/health` does not answer | the program is not running: step 5 |
| the video has no English captions | SponsorSkip uses SponsorBlock's answer for it; this is expected |

Logs: `backend.log` in the SponsorSkip folder, one line per reading and per failure.
