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

🔄 **PR1 infra complete & verified.** Now executing PR2: the bulk migration.

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

- [ ] Survey complete: full ExecutorStep inventory, Executor API consumers, example/test cleanup list
- [ ] lib/marin infrastructure solid (identify + fill helper gaps: train, eval)
- [ ] Exemplary self-describing experiment files written (PR1 examples)
- [ ] Old lazy prototype experiments cleaned up
- [ ] Bulk experiment migration complete
- [ ] Executor deleted, all call sites updated, strict guard flipped
- [ ] Codex review of sample experiment files (self-describing criteria)
- [ ] Split into two stacked PRs

## Log

- **2026-06-27** — Resumed after design-doc phase (PR #6649 design merged/ready). Created this
  tracker. Starting the full migration-surface survey.
