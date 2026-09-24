# Notices and attribution

SponsorSkip's code is released under the **GNU General Public License v3.0** (see `LICENSE`).

## Data and models it depends on

- **SponsorBlock** (https://sponsor.ajay.app), by Ajay Ramachandran and its contributors. SponsorSkip's
  training labels come from SponsorBlock's public database, which is licensed **CC BY-NC-SA 4.0**. The
  extension also looks up SponsorBlock's community segments (by a 4-character hash prefix, never the
  video ID). SponsorSkip is **not affiliated** with SponsorBlock.
- **sponsorblock-ml** by Xenova (Joshua Lochner), https://github.com/xenova/sponsorblock-ml: a model that
  finds sponsor segments in transcripts. Its code is **GPL-3.0** and its models are trained on
  SponsorBlock data (**CC BY-NC-SA 4.0**). SponsorSkip's v3 marker runs it as one of its detectors. Its
  Python source is vendored unchanged in `marker/vendor/sponsorblock_ml/` (upstream commit
  `7b6d44bf2397d6efcff5c2baff9bd33353f2cae3`, with its LICENSE); its model weights are downloaded from
  Hugging Face at first run, not redistributed here.
- **Sentence-embedding models** (MiniLM, BGE-small, potion-base-8M) from their authors on Hugging Face,
  under their own licences.
- **yt-dlp** (https://github.com/yt-dlp/yt-dlp), Unlicense, for captions.

## What that means for the models SponsorSkip ships

Every model trained here learned from SponsorBlock labels, so the **model files are shared under
CC BY-NC-SA 4.0**: attribution as above, **non-commercial use only**, and any model derived from them
must be shared under the same terms. SponsorSkip is free and carries no ads or paid tier.

SponsorSkip is not affiliated with YouTube or Google.
