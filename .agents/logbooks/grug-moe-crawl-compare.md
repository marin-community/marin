# grug-moe focus vs main crawl: Research Logbook

Issue: https://github.com/marin-community/marin/issues/6570
Branch: `agent/grug-moe-crawl-compare`

## Scope
- Goal: at matched compute/tokens, does the focus crawl (CC-SUPPLEMENTAL-2026-22,
  science-steered) beat a random sample of the main crawl (CC-MAIN-2024-18) on
  the grug-MoE compute-optimal ladder?
- Primary metric: `eval/paloma/macro_loss` (final), per rung.
- Constraints: data source is the only axis. Model/optimizer/batch/steps/seq are
  the documented baselines. One v5p-8 per run.

## Baseline
- Date: 2026-06-22
- Code refs: `experiments/grug/moe/README.md` (compute-optimal table),
  `experiments/grug/moe/heuristic.py` (`build_from_heuristic`).
- Nemotron-mix reference macro: d512 3.8104 / d768 3.4339 / d1024 3.1605 / d1280 3.0065.
- Rung configs reproduced from the heuristic match the README exactly:
  d512 (6L, b32, 6387 steps, 8.37e8 tok), d768 (8L, b64, 10343, 2.71e9),
  d1024 (11L, b128, 12649, 6.63e9), d1280 (13L, b256, 11807, 1.24e10).

## Experiment Log

### 2026-06-22 — setup + data pipeline validation
- Hypothesis: focus crawl yields lower macro_loss at matched tokens.
- Built `marin.datakit.download.common_crawl_wet` (WET WARC parser → dolma JSONL)
  and `experiments/grug/moe/launch_crawl_compare.py` (STAGE=data | STAGE=train).
- Verified end-to-end ingest on one real WET file per crawl:
  focus 6834 records (0pointer.net etc.), main 29417 records — both valid dolma
  JSONL. Manifests: main 90,000 files (`wet.paths.gz`), focus 4,573 files
  (index parquet, row-partitioned so one part = full universe). Seeded sampling
  deterministic and distinct.
- Measured yield (real Llama-3.1 tokenizer): 3.59 chars/tok, ~13.8M tok per focus
  WET file → 13B ≈ 942 files. Sample sizes: focus 1200 (~16.5B), main 230 (~18.4B).
- Result: pipeline validated; import + step graph + paloma validation-set wiring
  confirmed; lint/pyrefly clean.
- Next action: STAGE=data Iris job, then 8 training jobs on v5p-8.
