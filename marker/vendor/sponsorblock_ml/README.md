# sponsorblock-ml (vendored)

Unmodified `src/` and `LICENSE` from https://github.com/xenova/sponsorblock-ml by Xenova (Joshua Lochner),
at commit `7b6d44bf2397d6efcff5c2baff9bd33353f2cae3`. GPL-3.0, the same licence as SponsorSkip.

SponsorSkip's level-1 marker (v3) runs this project's chunking, output parsing and merging code through
`marker/sponsorblock_ml.py`. Vendored so a fresh install does not depend on the upstream repository
staying available. The models it runs (Xenova/sponsorblock-small, EColi/SB_Classifier) load from
Hugging Face at first use.
