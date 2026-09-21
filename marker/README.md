# marker

The cheap detector that finds *candidate* sponsor regions, so Claude only has to
read a short window around each one instead of a whole transcript.

Why the split exists: on 2026-09-20 Claude, reading an 82,000-character
transcript, placed a sponsor read at 1616 s when it starts at 3616 s — it quoted
the read correctly and mistyped the number. Given a window and asked for a *line
index*, the same model placed all six test reads within 2.2 s. So: a marker
shortlists, Claude verifies in a window, and any answer that resolves outside
its own window is rejected.

**The marker is judged on recall, not accuracy.** A false alarm costs one 7-second
Claude window. A miss costs a whole sponsor read played at full volume. Those are
not the same size of mistake, so they are not scored the same way.

The full explanation, with diagrams:
<https://claude.ai/artifact/AxjiintXBAVa4rbeyfX2aN>

## Running it, in order

```bash
pip install --user sentence-transformers scikit-learn   # torch + yt-dlp already present

python marker/sample_segments.py --prefixes 40     # SponsorBlock -> data/raw_segments.json
python marker/pick_videos.py --take 450            # filter        -> data/candidates.json
python marker/fetch_captions.py --limit 300        # yt-dlp, slow  -> data/captions/*.json
python marker/build_dataset.py                     # label         -> data/examples.jsonl
python marker/features.py                          # MiniLM        -> data/features.npz
python marker/score.py --baseline                  # the number to beat
```

Every step is resumable and skips work already done, so a stopped fetch can just
be run again. Nothing in `data/` is committed.

`train.py` is Simon's to write. It reads `data/features.npz` (`X`, `y`, and which
`split` each row belongs to) and writes one `.npy` of probabilities for the
validation rows; `score.py --scores that_file.npy` grades it. Nothing here needs
to know how the model works.

## Notes worth keeping

- **SponsorBlock's CSV dumps are switched off** (bandwidth); the hash-prefix API
  is the way in, and it never learns which video is being asked about.
- **A 429 from YouTube is usually one awkward video, not a block on us** — measured
  here at roughly one video in ten. The fetcher backs off twice, gives up on that
  video, and only stops entirely after five refusals in a row.
- **Captions are fetched exactly as `server/youtube.mjs` fetches them** (same
  `SUB_LANGS`, same json3 parsing). If the training data were cleaner than what
  the extension sees live, the measured accuracy wouldn't survive the trip into
  production.
- **The split is by channel, never at random.** SponsorBlock covers big channels
  with conventional reads best; a model trained on it inherits that bias, and
  only a channel-held-out validation set will show it.
