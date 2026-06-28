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

🔄 **PR1 infra complete & verified. PR2 underway — catalog layer + tutorials done.**

Done so far on PR2 (all committed, tree imports green, lint+pyrefly clean):
- Added the last two generic builders to `marin.experiment.data`: `hf_download` (the
  HuggingFace-Hub download case) and `derived` (generic single-step transform/convert/filter
  with deps). Added `sample_count` to `tokenized()`.
- Folded the verified `_lazy` catalog prototypes into canonical filenames
  (nemotron/simple/dclm/paloma/uncheatable/common_pile); `pretraining_datasets/__init__`
  reduced to a docstring (catalogs imported by submodule).
- Migrated **all dataset catalogs** to lazy step-functions (climblab_ja, common_corpus, hplt,
  massive_function_calling, molmo2_cap, svg, dolma, dolmino, diagnostic_logs, nemotron_v2,
  swe_zero_12m, nsf_awards, starcoder2_extras, eval_datasets, midtraining_datasets, models).
  Pins/revisions/paths preserved verbatim; published reference weights kept as constants;
  mixture assembly moved to the experiment.
- Migrated the training **tutorials** onto `train_lm` (exp1078 7B, tiny cpu/gpu/tpu, 125M
  fineweb-edu, hello_world); deleted exp1077 (redundant with `dclm_1b_1x_inline`).

Also done: migrated long_context_datasets (finepdfs, longmino) + prebuilt_caches; added the
`default_validation` recipe (lazy Paloma+Uncheatable); added `sample_count` to `tokenized()`;
deleted the golden migration harness (it had served its purpose — `train_lm` proven equal to
`default_train`).

### Current tree state (expected mid-migration breakage)

The catalog migration removed the old executor catalog symbols, which cascades through the
god module `defaults.py` (slated for deletion): `defaults.py` imports
`paloma.paloma_tokenized` / `exp1600.uncheatable_eval_tokenized` (both removed), and
`exp1600_uncheatable_evals.py` imports `models.download_model_step` (renamed to
`download_model`). So `defaults.py` and everything importing it currently fail to import.
This is the expected WIP state of the PR2 bulk branch — the fix is the dismantle below, not a
patch. The migrated catalog/tutorial modules all import and build handles in isolation.

The grug-moe lazy *prototype* examples (`launch_lazy`/`phases_lazy`) and their tests were
deleted (per the user — they were design-validation scaffolding superseded by the real
migration). The `tests/experiment/` suite is green (12 passed). The canonical lazy example is
now `tutorials/dclm_1b_1x_inline.py` + the catalogs themselves.

### Remaining PR2 tail (the careful second phase)

1. **Dismantle `defaults.py`** (lynchpin; unblocks green). `default_train`/`train`/
   `prepare_lm_train`/`simulated_epoching_train` → `train_lm`. `default_sft`/`default_dpo`
   need light `sft_lm`/`dpo_lm` assemblers (same identity-vs-execution discipline as
   `train_lm`) or consumer rewrites. `default_validation_sets` → `recipes.default_validation`.
   Then delete `defaults.py` + `tokenization.py` (`default_download`/`default_tokenize`).
2. **Test-coupled bespoke launchers** (run their tests after each):
   - `grug/*` (4 old executor launchers: `launch`, `launch_cw_scale`,
     `launch_datakit_moe_mix`, `base/launch`) — `tests/test_grug_launch_checkpoint_paths`,
     `tests/test_grug_variant_contracts`. Migrate or delete (judge if superseded). The lazy
     prototypes were already deleted.
   - `datakit/testbed/*` (4) — `tests/datakit/testbed/test_*`.
   - `ferries/*` (3) — CI smoke/daily jobs; high-stakes, migrate carefully.
   - `references/*` (2), `scaling_law_sweeps/completed_adamh` (likely delete — historical).
3. **Eval-running experiments**: `evals/*` (7: evals, run_base_model_evals, run_key_evals,
   exp1600_uncheatable_evals, exp_evalchemy_eval, hf_log_probs, evalchemy_results_compiler).
   These run evaluations (not `train_lm`); may need light eval-running helpers.
4. **Post-training**: `posttrain/{instruction,preference}_datasets` — gated on the
   `sft_lm`/`dpo_lm` assembler decision.
5. **Sweep tutorials** (`train_tiny_sweep_tpu`, `train_tiny_sweep_dclm_tpu`) — blocked on the
   sweep↔train metrics bridge (below).
6. **Library layer (~25 lib/marin files)** — do AFTER experiments stop passing
   `InputName`/`versioned`: remove `THIS_OUTPUT_PATH`/`InputName`/`versioned` from config
   dataclasses + download/transform fns; drop the two `materialize` calls in `training.py`
   (lazy `build_config` already resolves configs fully, so they are no-ops); move
   `infer_tpu_variant_regions_from_iris` into `rl/placement.py`; fix `rl/rl_experiment_utils.py`
   (`ExecutorMainConfig`).
7. **Capstone**: delete executor content-addressing + `types.py`; flip
   `MARIN_EXECUTOR_STRICT` default-on; rewrite/remove `materialize`; delete executor-only
   tests; rewrite `docs/tutorials/train-an-lm.md`; codex review of sample experiments; split
   into two stacked PRs.

Open design question (blocks the two sweep tutorials): `select()` reduces over trials that
return a **metrics mapping**, but `train_lm` trials return a `Checkpoint` (no metrics payload).
Needs a sweep↔train metrics bridge (read W&B/levanter eval metrics) before sweeps over training
runs work end-to-end.

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
- [x] Added `hf_download`/`derived`/`sample_count`/`default_validation`; golden harness deleted
- [ ] Dismantle `defaults.py`/`tokenization.py` consumers; delete `default_train` & co. (lynchpin for green)
- [ ] Migrate test-coupled bespoke launchers (grug, datakit/testbed, ferries, references, scaling_law)
- [ ] Migrate eval-running experiments (evals/*) + posttrain (needs sft_lm/dpo_lm assemblers)
- [ ] Fix grug-test catalog-symbol fallout; sweep tutorials (needs sweep↔train metrics bridge)
- [ ] Library layer (~25 lib/marin): THIS_OUTPUT_PATH/InputName/versioned removal; drop materialize calls
- [ ] Executor deleted, strict guard flipped, all call sites updated
- [ ] Docs rewritten (train-an-lm.md teaches `train_lm`, not `default_train`)
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
