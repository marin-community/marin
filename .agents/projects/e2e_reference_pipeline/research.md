# Research: Canonical end-to-end reference pipeline

Backing research for `design.md`. Issue: [#7273](https://github.com/marin-community/marin/issues/7273).
Goal: one blessed experiment that runs `raw sample → datakit → pretrain → eval → readout`
on a tiny budget, as the "does the whole path still work + what did my change do to the
numbers" harness. Three sub-pipelines already exist; the work is composition + a resume/footprint
contract, not new stage machinery.

## The execution model this rides on (post-Executor)

The old content-addressed `Executor` was replaced by lazy `ArtifactStep` handles (commit
`f7f5535c39`, #6649). This is the single most important fact for the design — resume and
bounded footprint are *properties of this model*, not new code we write.

- `ArtifactStep[T]` — `lib/marin/src/marin/execution/lazy.py:169`. Frozen dataclass: `name`,
  `version`, `artifact_type`, `run`, `build_config`, `deps`, `runtime_args`, `override_path`,
  `adopt_source`, `expected_fingerprint`.
- **Identity = `name@version`, path = `{prefix}/{name}/{version}`** (`lazy.py:225`). NOT
  content-addressed, NOT region- or prefix-dependent. Re-running the same recipe writes to the
  same path → cache hit → **resume + no duplicate artifacts come for free** (directly answers the
  issue's resumable + object-store-footprint requirements, and #5744/#6897).
- **Fingerprint** = SHA256 of `canonical_json(build_config(ctx))` (`lazy.py:211`, `fingerprint.py`).
  Literals (model, hyperparams, dep *versions*) enter the fingerprint; values pulled from
  `StepContext` (`output_path`, `artifact_path(dep)`, `runtime_arg(key)`) are placeholders at
  fingerprint time, real at run time. So **execution choices (TPU type, region, pool size) do
  not change identity** — they live in `runtime_args`.
- **Drift**: `check_drift()` warns if the recorded fingerprint differs; `expected_fingerprint`
  turns it into a hard error (`artifact.py`).
- `StepRunner().run(steps, max_concurrent=8)` — `step_runner.py`. DAG schedule, distributed
  `step_lock()` (GCS atomic write, SPMD-safe), reads/writes `.artifact.json` records with config +
  fingerprint + provenance + typed result.
- `experiment_main(build)()` — `lib/marin/src/marin/experiment/cli.py:174`. Click CLI; `--version`
  (or `dev`), prints the lowered plan, `--run` to execute.
- `remote(fn, resources=...)` — `lib/marin/src/marin/execution/remote.py` dispatches a step to
  Fray/Iris; resources stay off identity.

## Stage 1 — datakit (exists, has its own driver)

- `experiments/datakit/reference_pipeline.py` — `reference_datakit_steps()` (`:364-624`) builds
  the full per-source + cross-source DAG: tokenize → embed (luxical) → quality → cluster_assign →
  decontam → minhash → dedup → **store** (5-way join routing docs into per-`(cluster, quality)`
  Levanter caches). Returns `DatakitSteps` with `output_buckets` as the terminal output.
- **Smallest canonical config = `SMOKE_SCALE`** (`:208-215`): K_train=64, K_views=(8,16),
  cluster_view=8, pool=16 workers, dedup parallelism 64. Comment: "a true end-to-end run on a
  testbed sample". `DEFAULT_SCALE` is the production 512-worker envelope.
- **Raw sample input**: pre-built testbed samples at
  `s3://marin-us-east-02a/marin/datakit/sample_<tokens>_<hash>/`; default `SAMPLE_PREFIX` is
  `sample_0.1b_7d7d8fd7` (~0.1B tokens). Per-source normalized `NormalizedData`
  (`{source}/outputs/main/part-*.parquet`, cols `{id, text, ...}`). `sample_sources()`
  (`:647-684`) auto-discovers sources from a prefix.
- **Store output** (`experiments/datakit/store/datakit_store.py:80-120`): `ClusteredStoreData`,
  path `cluster=<C>/quality=<Q>/part-…`, Levanter `SerialCache` shards + `artifact.json`
  (bucket_edges, tokenizer, per-bucket token counts). Per-bucket mixture weights come from
  tokenize `.stats.json` (`experiments/datakit/testbed/mixture.py:27-37`).
- Current driver: a **standalone `main()` + `StepRunner().run()`** with its own CLI
  (`--mode sample --sample-prefix … --sources …`), *not* a `build()` returning `ArtifactStep`
  handles. **Integration gap**: the store output must be exposed as `ArtifactStep`(s) a training
  mixture can consume.
- Existing CI: `experiments/ferries/datakit_ferry.py` (`.github/workflows/marin-canary-datakit-tier1.yaml`,
  daily) runs a *simpler* datakit path (download→normalize→minhash→dedup→consolidate→tokenize on
  FineWeb-Edu 10BT) — not the reference pipeline.

## Stage 2 — pretrain (exists, is the compose target)

- `experiments/references/reference_training_pipeline.py` — **currently working** (600M Grug, one
  run, static-weight mixture approximating pretrain/midtrain/SFT; validation = paloma + uncheatable).
  Helw150's "totally borked post Executor death" comment predates a fix; the file builds an
  `ArtifactStep[LevanterCheckpoint]` today. But it has **no datakit stage and only PPL/validation
  evals** — no downstream task evals. Helw150: replace the many references with this one.
- Tiny-model template: `experiments/tutorials/train_tiny_model.py` — `train_lm(...)` with
  `llama_nano`/`llama_150m`, `num_train_steps=100`, device-keyed `resources`/`batch_size`
  (CPU/H100/TPU). Token budget ≈ steps × batch × seq_len (CPU ≈ 0.8M tokens).
- `train_lm()` (`lib/marin/src/marin/experiment/train.py:101`) and `mixture(ctx, weights, validation)`
  (`lib/marin/src/marin/experiment/data.py:139-278`) are the assemblers. Training output =
  `LevanterCheckpoint` (`lib/marin/src/marin/training/training.py:56-86`), `checkpoint_dir` →
  `<output>/checkpoints/`.

## Stage 3 — eval + readout (exists, reusable steps landed)

- Reusable eval steps in `experiments/evals/evals.py` (issue cites #7267; the reusable-steps API
  landed around #7253):
  - `eval_step(model, group)` (`:164-186`) — one `EvalGroup` → one `EvalResult` artifact.
  - `eval_steps(model, groups)` (`:189-193`) — list convenience.
  - `eval_report(results, name)` (`:196-235`) — merge into one `EvalReport`, writes `report.json`.
  - `EvalGroup` (`:79-105`) — tasks + id + serve spec (the composable unit).
- Menus (`experiments/evals/task_configs.py`): `core_evals()` (12 MCQ tasks, `CORE_TASKS` `:11-27`),
  `key_evals()` (generation + MCQ as two groups, `:243-300`), `base_model_evals()`.
- Handoff: `eval_step` resolves the `LevanterCheckpoint` path (`evals.py:134-148`), marin-serve
  boots it as an OpenAI endpoint, evalchemy hits it (`evalchemy/serve_and_eval.py:350-600`).
- Readout: `EvalRunRecord` → `record.json` in object store (`lib/marin/src/marin/evaluation/records.py:94-127`,
  `gs://marin-eval-metadata/runs/…` or `s3://marin-us-east-02a/marin/eval-metadata/runs/…`),
  consumed by evaldash; in-loop evals log to W&B (`evaluation_config.py:11`). `eval_report`
  additionally writes a human-readable `report.json`.

## The datakit→train seam, precisely (load-bearing)

Reading the actual code sharpened the seam well past "reshape the output". Two execution
primitives coexist and the bridge crosses them:

- **Datakit is built on `StepSpec`** (`marin.execution.step_spec`), run by `StepRunner`, with
  **content-addressed** output routing `{MARIN_PREFIX}/{name}_{hash}/` driven by `hash_attrs`.
  `reference_datakit_steps(...)` (`experiments/datakit/reference_pipeline.py:465`) returns a
  `DatakitSteps` whose `output_buckets` is the terminal store `StepSpec`
  (`reference_pipeline.py:450-462`); its own `main()` runs `StepRunner().run(datakit.all_steps)`.
- **Train + eval are authored as `ArtifactStep`** handles (`name@version`) that *lower* to
  `StepSpec` (`lazy.py:300 _lower`, `lazy.py:397 run`). `train_lm(datasets=...)` wants
  `Mapping[ArtifactStep[TokenizedCache], float]` (`experiment/train.py:106`).
- **`ClusteredStoreData.buckets` omits empty buckets** (`store/datakit_store.py:91-120`,
  `_merge_per_bucket_ledgers`), so the non-empty `(cluster, quality)` set — and thus the training
  mixture's dataset list — **is only known after datakit runs**. It cannot be enumerated at
  authoring time.
- **`ArtifactStep.adopt(name, version, source, kind=, config=)`** (`lazy.py:245`) registers a
  pre-existing path as a typed handle, but the adopted handle has **no `deps`** — its lowered `fn`
  only checks the path exists (`FileNotFoundError` otherwise, `lazy.py:336-343`). So adopting
  bucket dirs does **not** create a dependency edge that forces datakit to run first.
- **Consequence**: a single lazy `ArtifactStep[EvalReport]` graph cannot, without new machinery,
  (a) discover the dynamic bucket set or (b) order datakit before train. The pragmatic bridge is a
  **driver with two `StepRunner` passes**: run datakit, `read_artifact(store.output_path,
  ClusteredStoreData)`, then adopt the resolved non-empty buckets into `train_lm` and `run(report)`.
  This also means the reference is a bespoke `main()` (like datakit's), **not** the
  `experiment_main(build)` single-handle idiom the current `reference_training_pipeline.py` uses.
- **The manual precedent**: `experiments/grug/moe/launch_datakit_moe_mix.py` hardcodes
  `_STORE_PREFIX = "datakit/store_8ac06c74"` and a transcribed 167-row `_BUCKET_PHASE_WEIGHTS`
  table, building a Levanter `LmDataConfig` with per-bucket `UrlDatasetSourceConfig` pointing at
  `cluster=…/quality=…` dirs — it does **not** go through `train_lm`/`TokenizedCache`. So there is
  a real fork on the trainer side: adopt-buckets-as-`TokenizedCache` + `train_lm` (aligned with the
  single-edit-model goal, assumes buckets satisfy the `TokenizedCache` contract) vs a moe-style
  `LmDataConfig` over bucket dirs (matches precedent, bypasses `train_lm`).
- **Resolved by the stress-test (Phase 6): the `TokenizedCache`/`train_lm` fork is a dead end.**
  Store buckets are *flat* Levanter caches (no `train/` subdir). `TokenizedCache.as_component()`
  (`tokenize.py:118`) omits `flat_cache` (default `False`, `datasets.py:335`), so Levanter looks for
  `<bucket>/train/` (`datasets.py:924`), doesn't find it, and **silently skips the component** →
  empty mixture. Adoption is separately broken: an adopted handle's `config` is written at the
  canonical `name@version` address, but `ctx.resolved` reads the record at `adopt_source` (the
  bucket dir), which has no `.artifact.json` → `cache.tokenizer` raises. So the design commits to the
  moe shape: a directly-built `LmDataConfig` with `flat_cache=True` per-bucket components, tokenizer
  from `store.tokenizer`, bucket paths as literals (no `adopt`), fed to the Grug launch trainer (not
  `train_lm`). Also confirmed: `total_tokens` defaults to `0` if `field_counts["input_ids"]` is
  absent (`datakit_store.py:575`) and a `0`-weight component is dropped by Levanter → `store_mixture`
  asserts `total_tokens > 0`; and `num_train_epochs` is unusable for store buckets (reads a
  `<bucket>/train/.stats.json` the store never writes).

## What surprised us / load-bearing findings

1. **Resume + bounded footprint are already guaranteed by `name@version` addressing** — the design
   does not need to invent caching or TTL logic, only *use the model correctly* (stable versions,
   `runtime_args` for execution knobs, no content in paths). This reframes the issue's "resumable"
   and "object-store-friendly" requirements from features-to-build into invariants-to-preserve.
2. **The real work is the datakit→train seam.** Datakit's terminal output is N `(cluster, quality)`
   Levanter caches + a weights recipe, whereas the training mixture wants dataset handles + weights.
   Datakit's reference pipeline is driven by a standalone `main()`, not `build()`-returning
   `ArtifactStep`s, so we need an adapter that yields the store as `ArtifactStep` dep(s) for the
   trainer. This is the one genuinely new piece of glue.
3. `reference_training_pipeline.py` is not actually broken today, but it is *incomplete* for this
   purpose (no datakit, PPL-only). The issue's intent is to grow/replace it into the full-path
   reference, retiring the narrower one.

## Open / unclear (feeds Interrogate + design Open Questions)

- Canonical "small" datakit config: `SMOKE_SCALE` on `sample_0.1b_7d7d8fd7` is the obvious pick —
  confirm.
- Reference eval readout: `key_evals()` vs `core_evals()` vs a tiny custom set; cost/latency of
  serving a nano model for evalchemy on the smoke budget.
- Location + replacement: extend `experiments/references/reference_training_pipeline.py` vs a new
  file; do we retire the old reference and the scaling/HP references' overlap.
- CI: wire as a ferry/canary once green (like `datakit_ferry.py`) or leave as an on-demand harness.
- Train tier: CPU (cheapest, truest smoke) vs a small TPU/GPU (needed for evalchemy serving?).
