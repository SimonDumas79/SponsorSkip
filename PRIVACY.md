# Privacy

SponsorSkip collects nothing. There is no account, no analytics and no server of ours.

**What leaves the browser, and where it goes:**

- **The ID of the YouTube video you open** goes to the SponsorSkip program **on your own computer**
  (`127.0.0.1`, never the network).
- **SponsorBlock** gets the first 4 characters of a hash of the video ID, a lookup shared by many
  videos, so it cannot tell which one you are watching.
- **Captions** are fetched from YouTube by the program on your computer (yt-dlp), as YouTube would serve
  them to you.
- **Levels 3 and 4 only:** caption text is sent to **the AI you picked** under Manage AI model, under that
  provider's terms: Anthropic (Claude Code), OpenAI (Codex), Google (Gemini), the URL you entered (API), or
  wherever your custom command sends it. With Ollama it stays on your computer. The default level sends
  caption text nowhere.
- **Only if you choose to submit a segment to SponsorBlock** in the "Help SponsorBlock" card: that
  segment, the video ID and a private SponsorBlock user ID stored in your browser go to SponsorBlock.
  Nothing is ever submitted automatically.

**API keys:** SponsorSkip needs no API key. If you choose the API-endpoint option, the key you enter is
stored only in `ai-config.json` on your computer, is sent only to the URL you entered, and is never shown
back in the extension or uploaded anywhere else. Changing the provider or the URL deletes it.

**What is stored:** your settings in your browser's extension storage, and your AI choice in
`ai-config.json` on your computer; each video's reading in the
`cache/` folder on your computer, and a log line per reading in `backend.log`. Delete either at any time.
