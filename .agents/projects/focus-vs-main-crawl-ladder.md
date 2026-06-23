# Focus crawl vs CC-MAIN-2024-18: equal-token grug-moe ladder

## Goal

Side-by-side data comparison: train the `experiments/grug/moe` compute-optimal
ladder (d512 / d768 / d1024 / d1280) on two raw-WET corpora sampled to the
**same token count**:

- **A — focus crawl** `CC-SUPPLEMENTAL-2026-22` ("top-10k science domains").
  WET at `…/projects/cc-open-athena-test/CC-SUPPLEMENTAL-2026-22/segments/<seg>/wet/*.warc.wet.gz`
- **B — main crawl** `CC-MAIN-2024-18` (the crawl used to build the focus seed list).
  WET manifest: `https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-18/wet.paths.gz`

Both are readable credential-free over the `data.commoncrawl.org` HTTPS gateway
(verified: focus WET returns 200; it does NOT need AWS creds despite the S3
bucket being private).

## Ingestion path (mirror the Nemotron-CC v1 path)

Reuse the exact shape of `lib/marin/src/marin/datakit/download/nemotron_v1.py`:

1. **Manifest** — fetch a `.paths`/`.paths.gz` list of WET files per crawl.
   - B: published `wet.paths.gz`.
   - A: no published manifest + listing is locked; derive the WET path list
     from the focus **index parquet** (`SELECT DISTINCT warc_segment` + the
     per-segment `wet/` listing) — or read `warc_filename` rows and map
     `warc/…​.warc.gz → wet/…​.warc.wet.gz`. (Already proven to resolve 200.)
2. **Seeded random sample** of files from each manifest, enough to clear the
   target token budget with headroom.
3. **Per-file download+parse** via a Zephyr `Dataset.from_list(...).map(...)`
   pipeline, streaming each `.warc.wet.gz` from the gateway and emitting
   dolma records `{id, text, source, format:"text", metadata}` to
   `.jsonl.zst`. New code = a small WET WARC-record reader (no warcio dep;
   split on `WARC/1.0`, keep `WARC-Type: conversion` blocks, take the body as
   `text`, `WARC-Record-ID` as `id`). ~4g RAM/worker like Nemotron.
4. **normalize** (`normalize_step`, text_field="text") → **tokenize**
   (`TokenizeConfig(train_paths=[…glob…], tokenizer=llama3)`).
5. **Equalize tokens**: tokenize ≥ target for both, then cap each cache to the
   exact same N tokens (truncate to min) so the comparison is token-matched.
6. Wrap each as `lm_data_config(training_set=…, shuffle=BlockShuffle…)` →
   feed `GrugMoeLaunchConfig.data`.

## Training

For each corpus, run the 4 compute-optimal rungs from `README.md`/`agent.md`
via `build_from_heuristic(budget, hidden_dim)` on **v5p-8**:

| Rung  | Budget   | Tokens  | v5p-8 runtime |
|-------|----------|---------|---------------|
| d512  | 2.19e17  | 8.37e8  | 0.6h |
| d768  | 1.70e18  | 2.71e9  | 2.8h |
| d1024 | 9.00e18  | 6.63e9  | 10.5h |
| d1280 | 2.83e19  | 1.24e10 | 26.8h |

→ 8 runs total, ~80 v5p-8-hours. Metric: `eval/paloma/macro_loss` per rung,
focus vs main, plus the fitted scaling-law curve per corpus.

Token budget must be ≥ d1280's 1.24e10 so the top rung never repeats data.

## Open decisions (see AskUserQuestion)

1. Token budget per crawl.
2. Run scope (full ladder now vs validate-smallest-first vs data-only).
3. Egress authorization + target GCS prefix (`MARIN_PREFIX`).

## Notes / risks

- Open-internet egress: ~target_tokens × ~5 bytes/token × 2 crawls. At 13B
  tokens that's ~120–180 GB pulled from `data.commoncrawl.org` into GCS, once.
  Flag per AGENTS.md cost rule; one-time, then training reads from GCS.
- Tokenizer: llama3 (matches Nemotron baseline + heuristic anchoring @ seq 4096).
- Issue/branch/W&B tracking per `run-research` SKILL; issue is a sub-issue of #4281.
