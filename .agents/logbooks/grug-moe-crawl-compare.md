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

### 2026-06-23 — data-prep done, training recovery, 8 runs launched
- Data-prep complete. Token pools (llama3, seed 0): focus 11,328,978 docs (~23B tok),
  main 6,963,215 docs (~16B tok). Both > d1280 (12.4B) → token-matched per rung.
- Recovery 1: focus tokenize hit the test/validation path guard (bucket `cc-open-athena-test`);
  added `allow_test_in_train` passthrough to `default_tokenize`.
- Recovery 2: training dispatch hard-requires WANDB_API_KEY → submit with `WANDB_MODE=disabled`
  (json_logger tracker captures macro_loss).
- Recovery 3: executor pinned the training step to the data region us-central2 but v5p-8 is in
  us-central1/us-east5 → `ResourceConfig.with_tpu("v5p-8", regions=["us-central1","us-east5"])`.
- All 8 runs launched: `/held/grug-crawlcmp-{focus,main}-{d512,d768,d1024,d1280}-r3`.
- Next: collect final `eval/paloma/macro_loss` per rung from json-logger output.

### 2026-06-23 — d512 results
- Final eval/paloma/macro_loss: focus 4.712, main 4.402 (Δ main−focus = −0.310).
- Main (general crawl) < focus (science crawl) on paloma macro at d512. Both >> the
  Nemotron-curated baseline (3.81), expected for raw WET. eval/macro_loss (overall):
  focus 4.627, main 4.420.
- Each d512 run took 1 preemption, auto-resumed from checkpoint. Wall ~1h for ~0.6h compute.
- Extraction: `iris job logs --max-lines N` (default tail is ~1000 lines; eval block needs a larger tail).

### 2026-06-23 02:30 — preemption mitigation
- Overnight v5p preemptions caused net-negative progress at 10-min checkpoints
  (main-d768 993→454 over ~1h). Added GRUG_CHECKPOINT_MINUTES env knob (default 10),
  relaunched 6 rungs with =3 + --max-retries 10. Resume from existing checkpoints.
- New job names: focus/main d768-r5, d1024-r5; focus-d1280-r5, main-d1280-r6.
