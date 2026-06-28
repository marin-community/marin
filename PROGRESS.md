# Lazy Artifacts Migration — Progress

Tracking the finish of the executor → lazy-artifacts migration on branch
`weaver/adopt-executor-context-for-remai`.

## Goal

Replace Marin's eager, import-time, content-addressing `Executor` with lazy typed
artifacts addressed by explicit `name@version`. End state:

- **`lib/marin`** holds the core `Artifact`/`Recipe`/`lower` machinery *plus* generic,
  high-quality helpers that abstract common workloads (tokenize, mixture, train, eval,
  sweep/select).
- **Experiments are self-describing.** An experiment declares its datasets, how they're
  tokenized, the optimizer, the model, the budget — all inline. There is **no
  `default_train`**; nothing hides the important decisions behind a wrapper.
- The `Executor` (content-addressing, `versioned`, `InputName`, `this_output_path`,
  `executor_main`) is **deleted**.

## Shipping plan — two stacked PRs

Do all the work in this one worktree (so the full migration surface is known), then split:

1. **PR1 — infrastructure + examples.** The `lib/marin` core + authoring helpers, plus a
   few exemplary self-describing experiment files. Lands first.
2. **PR2 — bulk migration + delete executor.** Stacked on PR1. Migrate every remaining
   import-time `ExecutorStep` site, flip the strict guard, delete the Executor.

## Status

🔄 **Experiment tree fully migrated + god modules deleted. Remaining: lib layer + executor deletion.**

Done (all committed, lint+pyrefly clean, touched test suites green):
- **Generic builders** in `marin.experiment.data`: `hf_download`, `derived`, `sample_count`.
  Sweep redesigned: `AnnotatedCheckpoint` + `annotate` + `select() -> Checkpoint` reads each
  trial's recorded eval metric from its output (no trial-return-payload bridge).
- **All dataset catalogs** → lazy step-functions; pins/revisions/paths preserved verbatim;
  mixture assembly moved into experiments.
- **All training experiments** → `train_lm`: tutorials (dclm_1b_1x_inline, exp1078 7B, tiny
  cpu/gpu/tpu, 125M, hello_world, both sweep tutorials), references/reference_training_pipeline,
  daily ferry.
- **grug launchers** (moe launch/launch_cw_scale/base, deleted launch_datakit_moe_mix one-off);
  **datakit testbed** (StepSpec-based, runtime token-proportional weights); **evals/\*** (7
  eval-running experiments); **posttrain** dataset-prep catalogs; **canary + multislice CI
  ferries** (build()→Checkpoint, validate_canary_metrics resolves via `build().path('mirror://')`).
- **No more `default_*`**: deleted `experiments/defaults.py` + `experiments/tokenization.py`
  (`default_train`/`default_validation_sets`/`default_tokenize`/`default_download`/`default_sft`/
  `default_dpo` — all had their consumers migrated to inline `train_lm` + `tokenized`/`hf_download`
  + inline `[*paloma_validation(), *uncheatable_validation()]`). `recipes.default_validation`
  was removed too (inlined at call sites per "don't reach for a default_*").
- Deleted dead weight: golden migration harness, grug lazy prototypes, `completed_adamh`
  (historical), the Vizier `reference_hyperparameter_sweep` + its stale `add_scaling_heuristic.md`
  recipe (last remnant of the already-deleted isoflop/scaling-ladder subsystem).

**The experiment tree no longer references any executor symbol** (only docs: `experiments/AGENTS.md`
+ two grug READMEs still mention them).

### Remaining tail

1. **Library layer (~17 lib/marin files)** — strip executor coupling so `types.py` can go:
   - Foundational (owned in main loop): `processing/tokenize/tokenize.py` (drop `InputName`/
     `VersionedValue` from path unions), `processing/tokenize/data_configs.py` (collapse
     `TokenizerStep | TokenizeConfig` → `TokenizeConfig`, drop `output_path_of`/ExecutorStep
     branches; delete unused `lm_varying_mixture_data_config`/`interpolate_mixture_weights`/
     `add_validation_sets_to_mixture`/`mixture_for_evaluation`), `utilities/executor_utils.py`
     (`ckpt_path_to_step_name` → str-only; rename file off `_utils`), `training/training.py`
     (drop the two no-op `materialize(config)` calls), `rl/placement.py` (move
     `infer_tpu_variant_regions_from_iris` here from executor.py).
   - Mechanical config-default strips (delegate): `datakit/download/*` (5 — `this_output_path`
     default_factory → plain str; delete dead npm ExecutorStep builder), `evaluation/*`
     (log_probs, save_logprobs, perplexity_gap, visualize, trace_labeled_eval — strip configs,
     delete dead ExecutorStep builders), `export/levanter_checkpoint.py`,
     `transform/conversation/conversation_to_dolma.py`, `rl/rl_experiment_utils.py`
     (`ExecutorMainConfig`).
   - Check `tests/integration_test.py` (uses `lm_data_config`) + `tests/execution/*`.
2. **Capstone — the atomic executor deletion** (fully scoped). Key finding: the lazy runtime
   (`lazy.py`/`step_runner.py`/`remote.py`/`registry.py`) is already executor-free; the ONLY
   coupling is `step_spec.py::as_executor_step()` (a transition shim — dead, delete it) and the
   region-inference chain. `step_runner.py` does NO region inference and never calls
   `resolve_executor_step` — so all of `executor.py` (Executor/executor_main/materialize/
   ExecutorMainConfig/resolve_executor_step/_step_dag_tpu_regions/_infer_gcs_regions/...) is
   executor-only and gets deleted, EXCEPT `infer_tpu_variant_regions_from_iris` +
   `_regions_for_tpu_variants_from_iris` + `_regions_for_tpu_variant_from_iris`, which MOVE to
   `rl/placement.py` (its only consumer). Steps:
   - Move the 3 TPU-region-from-iris fns into `rl/placement.py`; update its import.
   - Delete `step_spec.py::as_executor_step()` + its `types` import.
   - Lib holdouts: `data_configs.py` (delete `lm_data_config`, collapse `step_to_lm_mixture_component`/
     `_verify_tokenizers_same` to `TokenizeConfig`-only, drop `TokenizerStep`/executor imports +
     `__init__` exports); `perplexity_gap.py` (delete `model_perplexity_scores`); `rl_experiment_utils.py`
     (migrate `make_rl_step`→lazy `Checkpoint` + drop `ExecutorMainConfig`, or delete — only its test uses it).
   - Tests: delete `tests/execution/test_executor.py`; trim `test_step_runner.py` (drop the
     `resolve_executor_step`/`_step_dag_tpu_regions` tests, keep the lazy StepRunner tests);
     migrate-or-delete `tests/integration_test.py` (executor_main end-to-end) and
     `tests/rl/test_rl_experiment_utils.py`; fix `tests/processing/tokenize/test_tokenize.py` (`InputName`).
   - Delete `executor.py`, `types.py`, `context.py` (the `executor_context` strict machinery —
     `MARIN_EXECUTOR_STRICT` never existed in the lazy world, just delete). Trim `execution/__init__.py`
     + `__init__.pyi` to lazy-only exports.
   - Docs: rewrite `docs/tutorials/train-an-lm.md` to teach `train_lm`; update `experiments/AGENTS.md`
     + the two grug READMEs (drop `executor_main` references). Codex review of sample experiments;
     split into two stacked PRs.

### PR1 — DONE (commits 63a60c2c95, a11760fcdf on top of d284eb3555)

- `marin.experiment.train.train_lm` — the generic assembler (verified against
  `default_train` by the golden test; standalone test added).
- DCLM 1B/1x tutorial rewritten onto it (~165→78 lines, every decision still visible).
- Data/sweep/evals helpers already solid. Example set for PR1: `dclm_1b_1x_inline`
  (train_lm), `grug/moe/launch_lazy` (mixture/pins), `grug/moe/phases_lazy` (chain).

### PR2 — ordered plan (gated: executor deletion is LAST, needs the whole tail)

1. Make the `_lazy` catalogs self-contained (inline the PIN / `PALOMA_DATASETS_TO_DIR`
   constants they pull from executor originals). [safe, additive]
2. Migrate executor *experiments* → `train_lm` + lazy catalogs: tutorials
   (`exp1077`, `exp1078`, `train_tiny_sweep_dclm_tpu`), `references/*`, training files.
3. Migrate the big tokenize catalogs → lazy step-functions: `common_pile`
   (32), `eval_datasets` (27), `midtraining_datasets` (7), `models`, remaining
   `pretraining_datasets/*`; migrate their consumers.
4. Rename `_lazy` → canonical, delete executor catalog originals.
5. Migrate `defaults.py` / `tokenization.py` consumers; delete `default_train` & co.
6. Rewrite `materialize` to drive `StepRunner`; move `infer_tpu_variant_regions_from_iris`
   into `rl/placement.py`; fix `rl/rl_experiment_utils.py`.
7. Delete executor content-addressing from `executor.py`; flip `MARIN_EXECUTOR_STRICT`
   default-on; trim `execution/__init__`.
8. Delete the golden migration harness + executor-only tests; clean prototype tests.
9. Codex review of sample experiment files; split into two stacked PRs.

Sweep tutorial (`train_tiny_sweep_tpu.py`) migrates in step 2 — it currently leans on
the executor-era `prepare_lm_train` / `_run_training_on_worker` in `defaults.py`.

## Design decision: the training helper (resolved 2026-06-27)

lib/marin gets a **light, generic `train_lm()` assembler** — NOT a `default_train`.
The dividing line is this design's own identity-vs-execution line:

- **It DERIVES the mechanical/execution plumbing** (no experiment meaning): the marin
  data-parallel mesh + token `compute_mapping`, `per_device_parallelism=-1`,
  `allow_nondivisible_batch_size`, the rolling resumption checkpointer, the
  `remote(run_levanter_train_lm)` dispatch, `output_path=ctx.out`, the WandB
  `replicate_path` plumbing, eval-harness wiring from an `EvalSuite`. Sharding
  (`tensor_parallel_size`) is the kind of heuristic hardware help that's explicitly OK.
- **It REQUIRES every meaningful experiment choice** as an explicit argument and
  defaults NONE of them: `model`, `optimizer`, `data`, `batch_size`, `seq_len`,
  `num_train_steps`, `z_loss_weight`, `evals` (pass `None` to opt out — no silent
  default suite), `resources`. Reading the experiment file shows every decision.

`mp` (precision) is identity-bearing but a single universal marin standard, so it's a
named-default parameter (override changes numerics). No baked optimizer, no baked
mixture, no baked eval set — those were `default_train`'s sins.

## Survey results (the full migration surface)

- **128 import-time `ExecutorStep` constructions across 46 files** (116 sites / 34 files
  excluding already-migrated). Heaviest: `common_pile/tokenize_common_pile.py` (32),
  `eval_datasets.py` (27), `midtraining_datasets.py` (7), `evals/evals.py` (5),
  `defaults.py` (5).
- **Bottlenecks (high fan-in):** `defaults.py` (16 importers), `tokenization.py` (15),
  `eval_datasets.py`, `evals/evals.py`. Migrating these unlocks dependents.
- **Executor deletion: ZERO hard blockers.** ~52 symbols to delete (content-addressing),
  4 to keep (`resolve_executor_step`, `materialize`, `resolve_local_placeholders`,
  `unwrap_versioned_value`). Only real library consumers: `marin/training/training.py`
  (imports `materialize`), `marin/rl/rl_experiment_utils.py` (`ExecutorMainConfig`),
  `marin/rl/placement.py` (`infer_tpu_variant_regions_from_iris` — move to placement.py).
  `materialize` itself calls `Executor().run()` internally → needs a rewrite to drive
  `StepRunner` directly.
- **`_lazy`/`_inline` prototypes shadow originals** → rename to replace, drop suffixes:
  `dclm_lazy.py`→`dclm.py`, `simple_lazy.py`→`simple.py`, `nemotron_lazy.py`→`nemotron.py`,
  `paloma_lazy.py`→`paloma.py`, `uncheatable_lazy.py`→`uncheatable.py`.
- **`test_train_lm_golden.py` is a migration harness** (pins lazy build vs executor
  `default_train`); delete it when the executor goes.

## Checkpoints

- [x] Survey complete: full ExecutorStep inventory, Executor API consumers, example/test cleanup list
- [x] lib/marin infrastructure solid — `train_lm` assembler added (the `default_train` replacement)
- [x] Exemplary self-describing experiment files (PR1): dclm inline on `train_lm`, grug launch/phases
- [x] `train_lm` covered by a standalone test that survives executor deletion
- [x] Core pretraining catalogs cleaned + self-contained (nemotron/simple/paloma); common_pile migrated
- [x] ALL dataset catalogs → lazy (eval_datasets, midtraining, models, long_context, prebuilt_caches,
      and the ~12 small pretraining_datasets/*) — pins/revisions preserved, each imports + builds
- [x] Training tutorials → `train_lm` (exp1078 7B, tiny cpu/gpu/tpu, 125M, hello_world); exp1077 deleted
- [x] Added `hf_download`/`derived`/`sample_count`; golden harness deleted
- [x] Sweep redesigned: `AnnotatedCheckpoint`/`annotate`/`select() -> Checkpoint` (no metrics bridge); both sweep tutorials migrated
- [x] Dismantled + DELETED `defaults.py`/`tokenization.py` (no more `default_*`); `default_validation` inlined
- [x] Migrated bespoke launchers (grug, datakit/testbed, ferries incl. canary+multislice CI, references); deleted completed_adamh + Vizier sweep + stale recipe
- [x] Migrated eval-running experiments (evals/*) + posttrain dataset catalogs
- [x] **Experiment tree references zero executor symbols** (docs aside)
- [ ] Library layer (~17 lib/marin): InputName/versioned/this_output_path removal; drop materialize calls; move infer_tpu_variant_regions_from_iris
- [ ] Executor content-addressing + types.py + strict machinery deleted, all call sites updated
- [ ] Docs rewritten (train-an-lm.md teaches `train_lm`); experiments/AGENTS.md + grug READMEs updated
- [ ] Codex review of sample experiment files (self-describing criteria)
- [ ] Split into two stacked PRs

## Remaining ExecutorStep surface (import-time sites still to migrate)

Big/non-dataset: `eval_datasets.py` (27, download→convert eval pipeline — different shape),
`midtraining_datasets.py` (7), `evals/evals.py` (5), `defaults.py` (5),
`references/reference_hyperparameter_sweep.py` (4), `posttrain/{preference,instruction}_datasets.py`
(2 each), `tokenization.py` (2), `tutorials/hello_world.py` (2), `datakit/testbed/train.py` (2).
Small uniform pretraining catalogs (1 each): climblab_ja, common_corpus, diagnostic_logs, dolma,
dolmino (2), hplt, massive_function_calling, molmo2_cap, nemotron_v2, nsf_awards, svg, swe_zero_12m.
Executor catalog originals (dclm/simple/nemotron/paloma) drop in the endgame once their
executor-experiment consumers (exp1077/exp1078/references/golden) migrate.

Follow-up: common_pile lazy handles tokenize to fresh `common_pile/<name>/v1` caches (the executor
originals were content-addressed, no stable pin). Raw downloads are reused; first use re-tokenizes.
Pin existing tokenized caches if their locations are recoverable.

## Log

- **2026-06-27** — Resumed after design-doc phase (PR #6649 design merged/ready). Created this
  tracker. Starting the full migration-surface survey.
