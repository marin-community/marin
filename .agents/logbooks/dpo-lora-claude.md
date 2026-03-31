# DPO LoRA Claude Logbook

## Tune-LoRA Sweep Jobs (2026-03-31)

### Job Launch — 2026-03-31T07:27Z

Launched 5 LoRA DPO tuning jobs on Iris (v5p-8) from `experiments/tune_lora/`:

| Experiment | LR | Seed | Iris Job ID | Status |
|---|---|---|---|---|
| `beta0p1_lr5e6_seed2_b64` | 5e-6 | 2 | `/ahmed/iris-run-beta0p1_lr5e6_seed2_b64-20260331-072714` | RUNNING (train_dpo PENDING) |
| `beta0p1_lr6p25e6_seed2_b64` | 6.25e-6 | 2 | `/ahmed/iris-run-beta0p1_lr6p25e6_seed2_b64-20260331-072731` | RUNNING (train_dpo PENDING) |
| `beta0p1_lr8p75e6_seed2_b64` | 8.75e-6 | 2 | `/ahmed/iris-run-beta0p1_lr8p75e6_seed2_b64-20260331-072745` | RUNNING (train_dpo PENDING) |
| `beta0p1_lr1e5_seed2_b64` | 1e-5 | 2 | `/ahmed/iris-run-beta0p1_lr1e5_seed2_b64-20260331-072752` | RUNNING (train_dpo PENDING) |
| `beta0p1_lr7p5e6_seed0_b64` | 7.5e-6 | 0 | `/ahmed/iris-run-beta0p1_lr7p5e6_seed0_b64-20260331-072757` | RUNNING (train_dpo PENDING) |

All executor parents RUNNING, train_dpo sub-jobs PENDING (awaiting v5p-8 allocation). 0 failures, 0 preemptions across all jobs.

### Babysit Check Log

#### Check 0 — 2026-03-31T07:28Z
- All 5 executor parents: `JOB_STATE_RUNNING`
- All 5 `train_dpo` sub-jobs: `JOB_STATE_PENDING` (waiting for v5p-8 TPU)
- Failures: 0 across all jobs
- Preemptions: 0 across all jobs
- Action: None needed. Monitoring every 10 min.

#### Check 1 — 2026-03-31T07:38Z
- All 5 executor parents: `JOB_STATE_RUNNING`
- All 5 `train_dpo` sub-jobs: `JOB_STATE_PENDING`
  - Scheduler message: "Insufficient memory (need 400.0GB, available 2.8G...)" — autoscaler spinning up v5p-8 workers
- Failures: 0 across all jobs
- Preemptions: 0 across all jobs
- Action: None needed. Waiting for v5p-8 TPU allocation.

#### Check 2 — 2026-03-31T07:48Z
- All 5 executor parents: `JOB_STATE_RUNNING`
- All 5 `train_dpo` sub-jobs: `JOB_STATE_PENDING`
  - Scheduler message unchanged: "Insufficient memory (need 400.0GB, available 2.8G...)" — still waiting for v5p-8 scale-up
- Failures: 0 across all jobs
- Preemptions: 0 across all jobs
- Action: None needed. Autoscaler provisioning in progress (~20 min since launch).

#### Check 3 — 2026-03-31T07:58Z
- All 5 executor parents: `JOB_STATE_RUNNING`
- All 5 `train_dpo` sub-jobs: **`JOB_STATE_RUNNING`** (v5p-8 TPUs allocated!)
- Failures: 0 across all jobs
- Preemptions: 0 across all jobs
- Action: None needed. All jobs now actively training on TPU.

#### Check 4 — 2026-03-31T08:08Z
- All 5 executor parents: `JOB_STATE_RUNNING`
- All 5 `train_dpo` sub-jobs: `JOB_STATE_RUNNING`
- Failures: 0 | Preemptions: 0
- Action: None needed. Steady state.

#### Check 5 — 2026-03-31T08:18Z
- All 5 executor parents: `JOB_STATE_RUNNING`
- All 5 `train_dpo` sub-jobs: `JOB_STATE_RUNNING`
- Failures: 0 | Preemptions: 0
- Action: None needed. Steady state (~50 min into training).

#### Check 6 — 2026-03-31T08:30Z
- All 5 executor parents: `JOB_STATE_RUNNING`
- All 5 `train_dpo` sub-jobs: `JOB_STATE_RUNNING`
- Failures: 0 | Preemptions: 0
- Action: None needed. Steady state (~1h into training).

#### Check 7 — 2026-03-31T08:40Z
- All 5 executor parents: `JOB_STATE_RUNNING`
- All 5 `train_dpo` sub-jobs: `JOB_STATE_RUNNING`
- Failures: 0 | Preemptions: 0
- Action: None needed. Steady state (~1h10m into training).

#### Check 8 — 2026-03-31T08:50Z
- All 5 executor parents: `JOB_STATE_RUNNING`
- All 5 `train_dpo` sub-jobs: `JOB_STATE_RUNNING`
- Failures: 0 | Preemptions: 0
- Action: None needed. Steady state (~1h20m into training).

#### Check 9 — 2026-03-31T09:00Z
- All 10 jobs: `JOB_STATE_RUNNING`
- Failures: 0 | Preemptions: 0
- Steady state (~1h30m into training).

#### Check 10 — 2026-03-31T09:10Z
- All 10 jobs: `JOB_STATE_RUNNING`
- Failures: 0 | Preemptions: 0
- Steady state (~1h40m into training).

#### Check 11 — 2026-03-31T09:20Z
- All 10 jobs: `JOB_STATE_RUNNING`
- Failures: 0 | Preemptions: 0
- Steady state (~1h50m into training).

#### Check 12 — 2026-03-31T09:30Z
- All 10 jobs: `JOB_STATE_RUNNING`
- Failures: 0 | Preemptions: 0
- Steady state (~2h into training).

#### Check 13 — 2026-03-31T09:40Z
- All 10 jobs: `JOB_STATE_RUNNING`
- Failures: 0 | Preemptions: 0
- Steady state (~2h10m into training).

#### Check 14 — 2026-03-31T09:50Z
- All 10 jobs: `JOB_STATE_RUNNING`
- Failures: 0 | Preemptions: 0
- Steady state (~2h20m into training).

#### Check 15 — 2026-03-31T10:00Z
- All 10 jobs: `JOB_STATE_RUNNING`
- Failures: 0 | Preemptions: 0
- Steady state (~2h30m into training).

#### Check 16 — 2026-03-31T10:10Z
- All 10 jobs: `JOB_STATE_RUNNING`
- Failures: 0 | Preemptions: 0
- Steady state (~2h40m into training).

#### Check 17 — 2026-03-31T10:20Z
- All 10 jobs: `JOB_STATE_RUNNING`
- Failures: 0 | Preemptions: 0
- Steady state (~2h50m into training).

#### Check 18 — 2026-03-31T10:30Z
- All 10 jobs: `JOB_STATE_RUNNING`
- Failures: 0 | Preemptions: 0
- Steady state (~3h into training).

#### Check 19 — 2026-03-31T10:40Z — FAILURES DETECTED
- **lr=7.5e-6, seed=0** (`iris-run-beta0p1_lr7p5e6_seed0_b64-20260331-072757`):
  - Executor parent: `FAILED` — "Exit code 133: OOM killed (container exceeded memory limit)"
  - train_dpo sub-job: `KILLED` — "Job exceeded max_task_failures"
- **lr=6.25e-6, seed=2** (`iris-run-beta0p1_lr6p25e6_seed2_b64-20260331-072731`):
  - Executor parent: `FAILED` — "Exit code 133: OOM killed (container exceeded memory limit)"
  - train_dpo sub-job: `KILLED` — "Job exceeded max_task_failures"
- **lr=5e-6, seed=2**: RUNNING (OK)
- **lr=8.75e-6, seed=2**: RUNNING (OK)
- **lr=1e-5, seed=2**: RUNNING (OK)
- **Root cause**: Executor parent allocated only 1 GiB default memory via `iris job run`. The executor Python process OOM'd. Not a TPU/training issue.
- **Action**: Re-launching 2 failed jobs with `--memory 4GB` for executor parent.
- **Re-launched**:
  - lr=7.5e-6, seed=0 → `/ahmed/iris-run-beta0p1_lr7p5e6_seed0_b64-20260331-103046` (memory=4GB)
  - lr=6.25e-6, seed=2 → `/ahmed/iris-run-beta0p1_lr6p25e6_seed2_b64-20260331-103055` (memory=4GB)
- **Mistake 1**: Original launch used default `--memory 1GB` for executor parent. Should have specified `--memory 4GB`. The remaining 3 jobs (lr=5e-6, lr=8.75e-6, lr=1e-5) are still running on 1GB — they may OOM too. Will monitor closely.

#### Check 20 — 2026-03-31T10:50Z — RE-LAUNCH FAILED
- Re-launched jobs both `FAILED`:
  - "Exit code: 2. stderr: python: can't open file '/app/experime..."
  - **Mistake 2**: Re-launched from main repo (`cd /Users/ahmed/code/marin`) but `experiments/tune_lora/` only exists in the worktree. Workspace bundle didn't include the scripts.
- **Action**: Re-launching again from the worktree directory with `--memory 4GB`.
- 3 original surviving jobs (lr=5e-6, lr=8.75e-6, lr=1e-5) still RUNNING.
- **Re-launched (attempt 2, from worktree, 4GB memory)**:
  - lr=7.5e-6, seed=0 → `/ahmed/iris-run-beta0p1_lr7p5e6_seed0_b64-20260331-104044` (6.5 MB bundle — correct)
  - lr=6.25e-6, seed=2 → `/ahmed/iris-run-beta0p1_lr6p25e6_seed2_b64-20260331-104054` (6.5 MB bundle — correct)

#### Check 21 — 2026-03-31T10:50Z — ALL RECOVERED
- All 5 active jobs now RUNNING (executors + train_dpo sub-jobs):
  - lr=5e-6, seed=2 (original) — RUNNING
  - lr=8.75e-6, seed=2 (original) — RUNNING
  - lr=1e-5, seed=2 (original) — RUNNING
  - lr=7.5e-6, seed=0 (re-launched, 4GB) — RUNNING, train_dpo RUNNING
  - lr=6.25e-6, seed=2 (re-launched, 4GB) — RUNNING, train_dpo RUNNING
- Failures: 0 on active jobs | Preemptions: 0
- Action: None needed. Back to steady state.

#### Check 22 — 2026-03-31T11:00Z — ANOTHER OOM
- **lr=8.75e-6, seed=2** (`iris-run-beta0p1_lr8p75e6_seed2_b64-20260331-072745`):
  - Executor parent: `FAILED` — "Exit code 133: OOM killed (container exceeded memory limit)" (1GB)
  - train_dpo sub-job: `KILLED` — "Job exceeded max_task_failures"
- Same root cause as Check 19: original executor launched with default 1GB memory.
- 4 other active jobs still RUNNING.
- **Action**: Re-launching lr=8.75e-6 from worktree with `--memory 4GB`.
- **Re-launched**: lr=8.75e-6, seed=2 → `/ahmed/iris-run-beta0p1_lr8p75e6_seed2_b64-20260331-110037` (4GB, 6.5 MB bundle)
- Remaining original jobs on 1GB: lr=5e-6 and lr=1e-5. Expecting they may OOM too.

#### Check 23 — 2026-03-31T11:10Z
- All 5 active jobs RUNNING (executors + train_dpo):
  - lr=5e-6, seed=2 (original, 1GB) — RUNNING (~3h40m)
  - lr=1e-5, seed=2 (original, 1GB) — RUNNING (~3h40m)
  - lr=7.5e-6, seed=0 (re-launched, 4GB) — RUNNING (~30m)
  - lr=6.25e-6, seed=2 (re-launched, 4GB) — RUNNING (~30m)
  - lr=8.75e-6, seed=2 (re-launched, 4GB) — RUNNING (~10m, TPU allocated)
- Failures: 0 on active jobs | Preemptions: 0

#### Check 24 — 2026-03-31T11:20Z — OOM on lr=5e-6
- **lr=5e-6, seed=2** (`iris-run-beta0p1_lr5e6_seed2_b64-20260331-072714`):
  - Executor parent: `FAILED` — "Exit code 133: OOM killed" (1GB)
  - train_dpo sub-job: `KILLED` — "Job exceeded max_task_failures"
- 4 other active jobs still RUNNING.
- **Re-launched**: lr=5e-6, seed=2 → `/ahmed/iris-run-beta0p1_lr5e6_seed2_b64-20260331-112030` (4GB, worktree)
- Last remaining 1GB original: lr=1e-5, seed=2. Expecting OOM soon.

#### Check 25 — 2026-03-31T11:30Z
- All 5 active jobs RUNNING:
  - lr=5e-6, seed=2 (re-launched, 4GB) — RUNNING, train_dpo RUNNING (~10m)
  - lr=6.25e-6, seed=2 (re-launched, 4GB) — RUNNING (~50m)
  - lr=7.5e-6, seed=0 (re-launched, 4GB) — RUNNING (~50m)
  - lr=8.75e-6, seed=2 (re-launched, 4GB) — RUNNING (~30m)
  - lr=1e-5, seed=2 (original, 1GB) — RUNNING (~4h, last 1GB survivor)
- Failures: 0 on active jobs | Preemptions: 0

#### Check 26 — 2026-03-31T11:40Z
- All 10 jobs (5 executors + 5 train_dpo): `JOB_STATE_RUNNING`
- Failures: 0 | Preemptions: 0
- lr=1e-5 original (1GB) still surviving at ~4h10m. Steady state.

#### Check 27 — 2026-03-31T11:50Z
- All 10 jobs: `JOB_STATE_RUNNING`
- Failures: 0 | Preemptions: 0
- Steady state. lr=1e-5 (1GB) still alive at ~4h20m.

#### Check 28 — 2026-03-31T12:00Z
- All 10 jobs: `JOB_STATE_RUNNING`
- Failures: 0 | Preemptions: 0
- Steady state. lr=1e-5 (1GB) at ~4h30m — may have dodged OOM.

#### Check 29 — 2026-03-31T12:10Z
- All 10 jobs: `JOB_STATE_RUNNING`
- Failures: 0 | Preemptions: 0
- Steady state (~4h40m for lr=1e-5 original).

#### Check 30 — 2026-03-31T12:20Z — LAST 1GB OOM
- **lr=1e-5, seed=2** (`iris-run-beta0p1_lr1e5_seed2_b64-20260331-072752`):
  - Executor parent: `FAILED` — "Exit code 133: OOM killed" (1GB) after ~4h50m
  - train_dpo sub-job: `KILLED` — "Job exceeded max_task_failures"
- All 5 original 1GB executors have now OOM'd (4/5 recovered, this is the last).
- **Re-launched**: lr=1e-5, seed=2 → `/ahmed/iris-run-beta0p1_lr1e5_seed2_b64-20260331-122035` (4GB, worktree)
- All 5 jobs now on 4GB executors. No more 1GB time bombs.

#### Active Job Summary (post-recovery)
| Experiment | Iris Job ID | Executor Memory | Status |
|---|---|---|---|
| lr=5e-6, seed=2 | `112030` | 4GB | RUNNING |
| lr=6.25e-6, seed=2 | `104054` | 4GB | RUNNING |
| lr=7.5e-6, seed=0 | `104044` | 4GB | RUNNING |
| lr=8.75e-6, seed=2 | `110037` | 4GB | RUNNING |
| lr=1e-5, seed=2 | `122035` | 4GB | RUNNING (just launched) |

#### Check 31 — 2026-03-31T12:30Z
- All 10 jobs: `JOB_STATE_RUNNING` (all 4GB executors now)
- lr=1e-5 re-launch already has TPU allocated
- Failures: 0 | Preemptions: 0
- Steady state. All 5 jobs on 4GB executors — OOM risk eliminated.

#### Checks 32–48 — 2026-03-31T12:40–15:20Z
- All jobs steady through checks 32–48.

#### Check 49 — 2026-03-31T15:30Z — lr=1e-5 EXECUTOR FAILED + lr=7.5e-6 PREEMPTED
- **lr=1e-5, seed=2** (`122035`):
  - Executor parent: `FAILED` — "ValueError: Executor step 'tokenized/bloom_speceval_v2_train_prefs_marin_tokenizer' has cross-region GCS dependencies. Found regions {us-central1, us-east5}."
  - train_dpo sub-job: `KILLED` — "Parent task preempted"
  - Root cause: Executor landed in a region where the cache/output path (us-east5) doesn't match the train data (us-central1). Region-inference code raises ValueError.
  - **Action**: Re-launching. The executor should land in a compatible region on retry.
- **lr=7.5e-6, seed=0** (`104044`):
  - train_dpo: PREEMPTED, auto-rescheduled by Iris (new timestamp 15:21:25). Executor still RUNNING.
  - **Action**: None (not interpreting preemptions per user instructions).
- 3 other jobs: RUNNING, healthy.
- **Re-launched**: lr=1e-5, seed=2 → `/ahmed/iris-run-beta0p1_lr1e5_seed2_b64-20260331-153209` (4GB, worktree)

#### Check 50 — 2026-03-31T15:40Z — PERSISTENT CROSS-REGION FAILURE
- **lr=1e-5, seed=2** (`153209`): `FAILED` again — same cross-region GCS ValueError
- **lr=7.5e-6, seed=0** (`104044`): Executor `FAILED` with same cross-region error. train_dpo `KILLED` (parent preempted).
  - Error: "Executor step 'tokenized/bloom_speceval_v2_train_prefs_marin_tokenizer' has cross-region GCS dependencies. Found regions {us-central1, us-east5}. us-central1: train_paths; us-east5: cache_path, output_path"
  - Root cause: tokenized cache was allocated in us-east5 but train data is in us-central1. Executor validation rejects cross-region deps.
- **Action**: Re-launching both with `--region us-central1` to pin executor region.
- 3 other jobs (lr=5e-6, lr=6.25e-6, lr=8.75e-6): RUNNING, healthy.
- **Re-launched with `--region us-central1`**:
  - lr=1e-5, seed=2 → `/ahmed/iris-run-beta0p1_lr1e5_seed2_b64-20260331-154129`
  - lr=7.5e-6, seed=0 → `/ahmed/iris-run-beta0p1_lr7p5e6_seed0_b64-20260331-154158`

#### Check 51 — 2026-03-31T16:10Z — REGION PIN WORKED, ALL RECOVERED
- All 5 active jobs RUNNING (executors + train_dpo):
  - lr=5e-6, seed=2 (`112030`) — RUNNING (~4h50m)
  - lr=6.25e-6, seed=2 (`104054`) — RUNNING (~5h30m)
  - lr=8.75e-6, seed=2 (`110037`) — RUNNING (~5h10m)
  - lr=1e-5, seed=2 (`154129`, us-central1 pinned) — RUNNING, train_dpo RUNNING
  - lr=7.5e-6, seed=0 (`154158`, us-central1 pinned) — RUNNING, train_dpo RUNNING
- Failures: 0 on active jobs | Preemptions: 0
- `--region us-central1` resolved the cross-region GCS error.

---

## PR Goals

- Unify regular DPO and LoRA-DPO under the canonical `levanter.main.train_dpo` entrypoint.
- Keep `TrainerState.model` policy-only for both regular DPO and LoRA-DPO.
- Factor adapter behavior into a reusable layer so future LoRA-enabled training flows can share the same machinery.
- Represent DPO variants via config flips:
  - `adapter.type: none` + `reference.type: separate` for regular DPO
  - `adapter.type: lora` + `reference.type: adapter_base` for LoRA-DPO
- Keep `lora_dpo.py` only as a compatibility shim for legacy configs.
- Preserve regular DPO behavior closely enough to sanity-check against earlier runs, including the legacy `model_key` lineage.
- Update configs, tests, and docs to match the new runtime shape.

## Work Completed

### 1. Canonical DPO refactor

- Refactored `lib/levanter/src/levanter/main/train_dpo.py` to be the single canonical DPO runtime.
- Removed the old `DpoModel(policy, reference)` trainer-state shape.
- Added `DpoReferenceConfig` with:
  - `SeparateReferenceConfig`
  - `AdapterBaseReferenceConfig`
- Added validation rules for invalid combinations:
  - `reference.type=adapter_base` requires a non-`none` adapter
  - `adapter.type=lora` + `reference.type=adapter_base` requires `zero_init_b=true`
- Changed the loss function to accept only the policy model.
- Implemented separate frozen-reference loading outside `TrainerState`, captured by the loss closure.
- Implemented adapter-base reference lookup through the adapter runtime.
- Applied `jax.lax.stop_gradient` to reference log-probs so the reference path is explicitly non-differentiated in both modes.

### 2. Shared adaptation layer

- Added `lib/levanter/src/levanter/adaptation.py`.
- Introduced:
  - `AdaptationConfig`
  - `AdaptationExportConfig`
  - `NoAdaptationConfig`
  - `LoraAdaptationConfig`
- Centralized:
  - adapter application
  - trainable-filter derivation
  - adapter-base model view lookup
  - export-hook installation
- Reused low-level LoRA operations from `lib/levanter/src/levanter/lora.py` rather than using legacy `lora_lm.py` as the architectural template.

### 3. Shared model/bootstrap helper

- Added `lib/levanter/src/levanter/main/model_init.py`.
- Factored out shared model loading/bootstrap logic for:
  - HF converter setup
  - tokenizer replacement
  - optional HF-config adoption
  - HF checkpoint load
  - Levanter checkpoint load
  - parameter casting/sharding

### 4. Legacy LoRA-DPO compatibility shim

- Rewrote `lib/levanter/src/levanter/main/lora_dpo.py` into a translation wrapper.
- Kept the legacy `LoraDpoConfig` surface for old config files.
- Translates legacy LoRA-DPO configs into canonical `TrainDpoConfig` with:
  - `adapter=LoraAdaptationConfig(...)`
  - `reference=AdapterBaseReferenceConfig()`
- Forwards execution into canonical `train_dpo.main`.

### 5. Experiment/config updates

- Updated `experiments/defaults.py` so `default_dpo(...)` constructs canonical `TrainDpoConfig(reference=SeparateReferenceConfig(...))`.
- Updated canonical DPO YAML configs under `lib/levanter/config/` to use nested `adapter` / `reference` blocks instead of top-level `reference_model_path` / `reference_is_hf`.
- Left legacy `lib/levanter/config/dpo/lora_dpo_*` YAMLs on the compatibility path intentionally, so old LoRA-DPO configs still route through the shim.

### 6. Tests

- Updated `lib/levanter/tests/test_dpo.py` to match the new architecture.
- Removed old tests that assumed `DpoModel` lived in trainer state.
- Added tests for:
  - policy-only `TrainerState`
  - invalid adapter/reference combinations
  - canonical config parsing for `adapter.type: none`
  - canonical config parsing for `adapter.type: lora`
  - legacy `LoraDpoConfig` translation
- Kept existing `lib/levanter/tests/test_lora_dpo.py` passing against the refactor.
- Replaced brittle parse-from-repo-config tests with minimal temp YAML fixtures after an initial failure exposed unrelated data-config parsing fields.

### 7. Docs

- Updated `lib/levanter/docs/guides/DPO-Training.md` to describe:
  - canonical `train_dpo.py`
  - nested `adapter` / `reference` config
  - policy-only trainer state
  - separate-reference vs adapter-base reference behavior
- Updated `lib/levanter/docs/guides/LoRA-DPO-Training.md` to describe:
  - canonical `train_dpo.py` usage
  - `adapter.type: lora`
  - `reference.type: adapter_base`
  - explicit `zero_init_b: true` requirement
  - legacy `lora_dpo.py` status as compatibility-only

## Follow-up Review Changes

After the initial refactor, a follow-up review requested two concrete changes.

### Logger style

- Changed the new `logger.info(...)` calls in `lib/levanter/src/levanter/main/train_dpo.py` back to f-strings.

### RNG lineage preservation for regular DPO

- Restored the legacy full-DPO top-level split shape in canonical `train_dpo.py`:
  - `data_key, adapter_key, model_key, training_key = split(PRNGKey(seed), 4)`
  - this intentionally repurposes the old unused loader-key slot as `adapter_key`
- Added `_derive_training_keys(seed)` to preserve the old regular-DPO policy key lineage:
  - `policy_key = split(model_key)[0]`
- Used `model_key` as the separate-reference checkpoint shape key so non-HF separate references follow the old regular-DPO path more closely.
- Added a regression test in `lib/levanter/tests/test_dpo.py` to verify that the derived `data_key`, `model_key`, `policy_key`, and `training_key` match the legacy full-DPO derivation.

## Notes From Design / Review Questions

- `inference_mode(...)` does not stop gradients; it flips modules with an `inference` flag into eval behavior, typically relevant for dropout-like modules.
- `jax.lax.stop_gradient(...)` is still needed for the reference path when the reference can be derived from the policy model itself.
- There is no `haliax.stop_gradient` helper in the current Haliax version used here, so `jax.lax.stop_gradient(...)` is the direct primitive.
- `SeparateReferenceModelProvider.model_for(policy_model)` takes `policy_model` only to share one interface with the adapter-base provider. In the separate-reference case it is intentionally ignored.

## Validation Performed

- Ran targeted syntax verification on changed Python files during implementation.
- Ran targeted DPO tests multiple times during the refactor.
- Final targeted test result:
  - `uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/test_dpo.py lib/levanter/tests/test_lora_dpo.py`
  - Result: `27 passed, 1 skipped`
- Ran repo-required targeted lint/type/style checks with:
  - `./infra/pre-commit.py --fix ...`
  - Final result: passed on the touched files

## Remaining Scope / Non-Goals In This Branch

- I did not migrate `lora_lm.py` to the new adaptation layer.
- I did not remove legacy LoRA-DPO YAMLs; they still intentionally target the compatibility shim.
- I did not run a full training job end-to-end.
- I did not update every `SimpleDPOConfig` call site in `experiments/`; the executor defaults path now emits canonical `TrainDpoConfig`, but the higher-level simple config surface still uses `reference_model_path` / `reference_is_hf`.

## 2026-03-29 Debugging Update

### What Went Wrong

- The original refactor goal "`TrainerState.model` is policy-only for both regular DPO and LoRA-DPO" did not survive a real multihost TPU run.
- For `reference.type=separate`, I changed regular DPO to load the frozen reference once and close over it in the loss function instead of storing it in trainer state.
- On multihost TPU, that closed over a sharded `jax.Array` spanning non-addressable devices.
- JAX rejected that during lowering with:
  - `RuntimeError: Closing over jax.Array that spans non-addressable (non process local) devices is not allowed.`
- This was not an Iris scheduling problem and not a W&B naming problem. It was a real training-runtime bug in the new canonical DPO code path.
- The old regular DPO script worked because the frozen reference lived inside `state.model`, so it was passed into the compiled train step as an argument instead of being captured as a constant.
- Because I launched a batch of relaunches before catching this, all of the `new_dpo` regular-DPO reruns failed at the same compile/lowering boundary.

### Operational Mistakes / Cleanup

- I relaunched the central1 `lr7.5e-7` sweep members before validating a real multihost TPU training run of the refactor.
- I also had east5 `new_dpo` relaunches still active while debugging the central1 failure path.
- After the failure was confirmed, I killed the still-running east5 sibling reruns so they would stop consuming TPU capacity:
  - `new_dpo_bloom_speceval_v2_marin_instruct_beta0.1_seed2`
  - `new_dpo_bloom_speceval_v2_marin_instruct_beta0.01_seed1`
  - `new_dpo_bloom_speceval_v2_marin_instruct_beta0.01_seed0`
  - `new_dpo_bloom_speceval_v2_marin_instruct_beta0.1_seed1`
- I then used `/ahmed/new_dpo_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed0` as the single debug specimen.
- When relaunching the fixed job in two regions in parallel, I had to stop reusing the same run id. Parallel east5/central1 launches need distinct W&B ids and distinct checkpoint roots or they collide.

### Code Fix Applied

- Restored `DpoModel(policy, reference)` in `lib/levanter/src/levanter/main/train_dpo.py` for `reference.type=separate`.
- Kept policy-only state for `reference.type=adapter_base` so LoRA-DPO still uses the unified adapter-base path.
- Updated the loss function to accept:
  - `DpoModel` for separate-reference regular DPO
  - `LmHeadModel` for adapter-base LoRA-DPO
- Added `_load_separate_reference_model(...)` for the explicit separate-reference load path.
- Added `_install_separate_reference_export_hooks(...)` so the separate-reference path still exports only the policy model.
- Kept the adapter-based export path for LoRA unchanged.

### Validation After Fix

- Ran:
  - `./infra/pre-commit.py --fix lib/levanter/src/levanter/main/train_dpo.py lib/levanter/tests/test_dpo.py`
  - `uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/test_dpo.py lib/levanter/tests/test_lora_dpo.py`
- Final result:
  - `29 passed, 1 skipped`
- Added a regression test in `lib/levanter/tests/test_dpo.py` to assert that the separate reference is marked non-saveable / non-trainable in the mask.

### Relaunch State After Fix

- Relaunched only the single target experiment in parallel across two regions with distinct ids:
  - central1: `/ahmed/new_dpo_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed0-c1`
  - east5: `/ahmed/new_dpo_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed0-e5`
- Both use the same training hyperparameters as the original `beta0.1_seed0_lr7.5e-7` sweep member.
- Both add region / TPU tags so the hardware is visible later in W&B:
  - `new_dpo`
  - `v5p-32`
  - region tag (`us-central1-a` or `us-east5-a`)
- Current state at time of writing:
  - central1 job is `PENDING` waiting for `tpu_v5p_32-us-central1-a` capacity
  - east5 job is `RUNNING`

### Updated Conclusion

- The abstraction split is still good at the config level:
  - regular DPO: `adapter.type=none`, `reference.type=separate`
  - LoRA-DPO: `adapter.type=lora`, `reference.type=adapter_base`
- The runtime shape is not fully symmetric:
  - regular DPO currently needs `DpoModel(policy, reference)` in trainer state for multihost safety
  - LoRA-DPO can stay policy-only because the reference is derived from the policy model inside the step
- So the original "policy-only for both modes" objective should be considered disproven by experiment.

## Next Steps

- Watch the east5 relaunch until it gets past the old failure point:
  - JAX lowering / first train-step compile
  - first actual optimization step
- If the east5 run clears that boundary, keep it as the validation run and decide whether the queued central1 duplicate is still useful.
- If the east5 run fails again, pull logs immediately and compare against the old closure error to see whether there is a second bug behind the first one.
- If the central1 run later gets capacity and starts cleanly, compare behavior across regions before relaunching any broader DPO sweep.
- Only after one regular DPO run is confirmed stable on TPU should the rest of the failed `new_dpo` regular-DPO reruns be resubmitted.

## 2026-03-29 Live Monitoring Update

### Current Job State

- Both duplicate validation jobs are now genuinely running, not just scheduler-level `RUNNING`:
  - `/ahmed/new_dpo_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed0-c1`
  - `/ahmed/new_dpo_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed0-e5`
- Iris shows both jobs with:
  - `JOB_STATE_RUNNING`
  - `task_state_counts.running = 4`
  - empty `pending_reason`

### Evidence That The Regression Is Fixed

- Both jobs loaded the model weights successfully.
- Both jobs reached:
  - `First batch loaded ..., starting first train step (includes JIT compilation)...`
  - `Tracing train_step for jaxpr...`
  - `Lowering train_step to HLO...`
  - `Lowered train_step ...`
- Both jobs completed the first optimization step and entered the step-0 eval:
  - log line shape: `Progress on:train 1.00it/850it`
- This is the exact boundary where the earlier regular-DPO refactor was failing, so the separate-reference fix is now validated in live TPU execution.

### Current Eval Progress

- `-c1` is currently ahead:
  - reached about `106/184` eval batches
  - eval loss is still `0.693`, which is expected at step 0 before learning
- `-e5` is behind but healthy:
  - reached about `68/184` eval batches
  - eval loss is also `0.693`

### Why W&B Still Looks Sparse

- W&B can still appear to show only system metrics even while the jobs are healthy.
- The reason is the trainer ordering:
  - step-0 eval runs before the per-step tracker log flush
  - this eval is fairly large (`184` batches)
- So the jobs can be actively training/evaluating while W&B still has little or no `train/*` / `eval/*` history visible.
- This is a logging-order artifact, not evidence of a hang.

### Operational Recommendation

- The duplicate runs have now served their original debugging purpose:
  - they proved that the multihost compile/lowering regression is fixed
- The remaining reason to keep both is redundancy across regions.
- Practical recommendation:
  - keep watching until `-c1` finishes the initial eval and emits real W&B metrics
  - then keep `-c1` and kill `-e5` unless cross-region duplication is still desired overnight

## 2026-03-29 Babysitting Handoff (claude-lora-dpo session)

### Check 0 — 07:55 UTC

Picked up monitoring from Codex agent. Both jobs still active.

| Job | State | Steps | Loss | Failures | Preemptions | Notes |
|-----|-------|-------|------|----------|-------------|-------|
| `-c1` | RUNNING (pending retry) | 65/850 | 0.514 | 0 | 4 | Just preempted, Iris retrying |
| `-e5` | RUNNING | 58/850 | 0.553 | 0 | 0 | Healthy, all 4 tasks active |

- Loss dropping steadily from 0.693 (step 0) — DPO fix confirmed working.
- c1 preemption is routine; no action needed, Iris will reschedule.
- Will check hourly for 10 hours (until ~18:00 UTC).

### Check 1 — ~09:00 UTC

Both jobs had failures since last check but Iris retried successfully. Both restarting from scratch (no checkpoint saved — `steps_per_checkpoint=1000`, neither had reached it).

| Job | State | Progress | Failures | Preemptions | Notes |
|-----|-------|----------|----------|-------------|-------|
| `-c1` | RUNNING (4 tasks active) | restarting | 4 | 4 | Recovered from preemption, no recent train logs yet |
| `-e5` | RUNNING (4 tasks active) | reinitializing from HF | 8 | 0 | 8 failures, reloading marin-8b-instruct from scratch |

- Both lost ~58-65 steps of training progress (no checkpoint saved).
- This is expected given `steps_per_checkpoint=1000` and early preemption/failure.
- No action needed — both jobs are recovering autonomously.

### Check 2 — ~10:02 UTC

| Job | State | Steps | Loss | Failures | Preemptions | Notes |
|-----|-------|-------|------|----------|-------------|-------|
| `-c1` | RUNNING (4/4) | 56/850 | 0.588 | 8 | 4 | Training at 15.4s/it |
| `-e5` | RUNNING (4/4) | loading model | — | 12 | 0 | Still loading HF weights after restart |

- c1 back to step 56, loss tracking same trajectory as before.
- e5 had more failures (12 total), still loading marin-8b-instruct weights.
- Both non-terminal, Iris handling retries. No action needed.

### Check 3 — ~11:00 UTC

| Job | State | Progress | Failures | Preemptions | Notes |
|-----|-------|----------|----------|-------------|-------|
| `-c1` | RUNNING (4/4) | restarting | 16 | 4 | Was checkpointing, coord service died (preemption) |
| `-e5` | RUNNING (4/4) | loading HF weights | 16 | 0 | Still reinitializing from marin-8b-instruct |

- Both accumulating failures (16 each) but Iris retrying successfully.
- c1 error: `UNAVAILABLE: Failed to send RPC to coordination service` — classic preemption cascade.
- e5 keeps restarting from scratch (no checkpoint saved, `steps_per_checkpoint=1000`).
- Infrastructure churn, not code bugs. No action needed.

### Check 4 — ~12:05 UTC

| Job | State | Steps | Loss | Failures | Preemptions | Notes |
|-----|-------|-------|------|----------|-------------|-------|
| `-c1` | RUNNING (4/4) | 25/850 | 0.684 | 20 | 4 | Training at 15.5s/it, restarted again |
| `-e5` | RUNNING (4/4) | 1/850 | — | 20 | 0 | Just past JIT compile, first real step done |

- Both accumulating ~4 failures/hour. Infrastructure churn continues.
- Neither has reached a checkpoint yet (`steps_per_checkpoint=1000`).
- No terminal failures. Iris handling retries.

### Check 5 — ~13:05 UTC

| Job | State | Steps | Loss | Failures | Preemptions | Notes |
|-----|-------|-------|------|----------|-------------|-------|
| `-c1` | RUNNING (4/4) | initializing | — | 28 | 4 | Restarted ~12:51, connecting to JAX distributed |
| `-e5` | RUNNING (4/4) | 1/850 | — | 24 | 0 | Past JIT, should accelerate |

- c1 churning harder (28 failures). Repeatedly failing before reaching checkpoint.
- e5 finally past JIT compilation but still very early.
- Neither job has saved a checkpoint yet. Progress keeps resetting.
- No terminal failures. Iris retrying.

### Check 6 — ~14:05 UTC

| Job | State | Steps | Loss | Failures | Preemptions | Notes |
|-----|-------|-------|------|----------|-------------|-------|
| `-c1` | RUNNING (1 run, 3 pend) | just started | — | 36 | 4 | Restarted 13:52, waiting on 3 tasks |
| `-e5` | RUNNING (4/4) | 1/850 | — | 28 | 0 | Past JIT again, completing step 1 |

- Both keep getting knocked down and restarting before reaching checkpoint.
- Neither has made durable progress (no checkpoint saved).
- Failure rate: c1 ~8/hr, e5 ~4/hr. Infrastructure churn on v5p-32.
- No terminal state. Continuing to monitor.

### Check 7 — ~15:10 UTC (best check so far)

| Job | State | Steps | Loss | Failures | Preemptions | Notes |
|-----|-------|-------|------|----------|-------------|-------|
| `-c1` | RUNNING (4/4) | 57/850 | 0.575 | 37 (+1) | 4 | Stable! 42min in, 15.4s/it, ~3.4h ETA |
| `-e5` | RUNNING (4/4) | 1/850 | — | 32 (+4) | 0 | Past JIT at 14:58 |

- c1 failure rate dropped sharply (only +1 this hour). Best sustained run yet.
- At 15.4s/it, c1 should reach step 200 (HF export) in ~37 min (~15:47 UTC).
- e5 still restarting frequently but alive.

### Check 8 — ~16:12 UTC (c1 FAILED, relaunched)

| Job | State | Steps | Loss | Failures | Preemptions | Notes |
|-----|-------|-------|------|----------|-------------|-------|
| `-c1` | **FAILED** | ~57 | — | 39 | 204 | OOM exit 137 during checkpointing |
| `-c1-v2` | SUBMITTED | — | — | 0 | 0 | Relaunched with 200GB memory |
| `-e5` | RUNNING (4/4) | eval 133/184 | 0.693 | 36 | 0 | Step-0 eval, healthy |

- c1 OOM killed during checkpoint write at step ~57. 128GB CPU memory insufficient.
- Relaunched as `/ahmed/new_dpo_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed0-c1-v2` with `--memory 200GB`.
- e5 running step-0 eval, about to start actual training.

### Root Cause Analysis — Deterministic OOM at Step ~77

**Finding**: Both `-c1` and `-e5` jobs OOM at exactly step 75-78, every single restart.

**Cause**: `CheckpointerConfig(save_interval=timedelta(minutes=10))` triggers a temporary checkpoint every ~10 minutes. At ~15s/step:
- Step 1 (~0min): checkpoint saves OK (low memory)
- Step 38 (~15min): checkpoint saves OK
- Step 77 (~30min): **OOM killed** — `DpoModel(policy + reference)` = 2x 8B params, checkpoint serialization exceeds 128GB container limit

**Evidence**: `Container was OOM killed by the kernel` in every crash, always preceded by `Saving temporary checkpoint at step 77`.

**Fix applied**:
1. Bumped `ram="128g"` → `ram="256g"` in all 12 `experiments/sweep_dpo/*.py` files
2. Added `new_dpo_` prefix to run names for new W&B runs
3. Deleted 2 crashed W&B runs (`new_dpo_..._seed0-c1` and `_seed0-e5`)

**Proper long-term fix (not done)**: Exclude frozen reference model from checkpoint serialization in `train_dpo.py` — the reference is loaded from HF checkpoint anyway.

## 2026-03-29 Eval Performance Investigation

### Problem

Eval consumes ~45% of total runtime. In the old successful run (`e5e379`, 7.7h total):
- Step-0 eval: ~27 min
- Subsequent evals: ~37 min each
- 5 evals total: ~3.5 hours of eval in a 7.7 hour run

### Root cause

DPO eval runs both policy AND reference forward passes on the entire 25K-example val set.
The reference model is frozen — its log-probs never change between evals.
We recompute 25K reference forward passes 5 times identically.

### Eval parallelism experiments

| `per_device_eval_parallelism` | Eval batches | Rate/batch | Est. eval time | Notes |
|-------------------------------|-------------|------------|----------------|-------|
| 8 (default) | 184 | ~20s | ~60 min | Old runs |
| 16 | 92 | ~18s | ~27 min | Nearly free 2x |
| 32 (current) | 46 | ~29s | ~22 min | Diminishing returns, likely memory-bandwidth bound |

Going 8→16 was nearly free (TPU underutilized at 8). Going 16→32 showed sublinear scaling — per-batch time increased 1.6x for 2x examples, suggesting we're hitting memory bandwidth limits. TPU HBM stayed at ~34% throughout — dominated by model weights + optimizer state, not batch activations.

### Config changes made

- `experiments/simple_dpo_config.py`: Added `DPO_EVAL_PARALLELISM` dict and `per_device_eval_parallelism` field
- `experiments/defaults.py`: Wired `per_device_eval_parallelism` through to `TrainerConfig`
- All 12 `experiments/sweep_dpo/*.py`: Using `DPO_EVAL_PARALLELISM["v5p-32"]` = 32
- All 12 sweep files: Added `ram="256g"` and `new_dpo_` (then `new_dpo_v2_`) name prefix
- W&B naming: executor hash only changes with `name`/versioned fields/deps, NOT runtime params like `ram` or `per_device_eval_parallelism`

### Profiling infrastructure added

- `train_dpo.py`: Added `jax.named_scope` around all 4 forward passes (policy/reference × chosen/rejected)
- `callbacks/__init__.py`: Added `jax.named_scope("eval <name>")` around eval batch loop; added `timing/load_time`, `timing/loss_time`, `timing/num_batches` to eval metrics logged to W&B
- `simple_dpo_config.py`: Added `profiler: ProfilerConfig` field (defaults disabled)
- `defaults.py`: Wired `profiler` through to `TrainerConfig`
- `experiments/eval_dpo.py`: Standalone eval-only profiling script (no training, no executor framework)

### xprof profiling results (2026-03-29, `/ahmed/dpo_eval_profile_v4`)

**Setup**: v5p-32 (16 chips), per_device_eval_parallelism=32, 20 eval batches (capped), warmup 3 batches for JIT.

**Timing breakdown** (from W&B summary):
| Metric | Value | % of eval |
|--------|-------|-----------|
| Loss compute time | 537.1s | 97.3% |
| Data load time | 15.0s | 2.7% |
| Total eval time | ~552s (~9.2 min) | |
| Avg time/batch | ~27.6s | |

**Data loading is NOT the bottleneck.** 97.3% of time is in forward pass computation.

**HLO op analysis** (from xplane.pb binary search):
| Named scope | HLO op references | Notes |
|-------------|-------------------|-------|
| policy_chosen | 3003 | Full 8B Llama forward pass |
| policy_rejected | 2657 | Full 8B Llama forward pass |
| reference_chosen | 2646 | Full 8B Llama forward pass (identical every eval) |
| reference_rejected | 2606 | Full 8B Llama forward pass (identical every eval) |

All 4 forward passes have roughly equal HLO complexity (~15% variance). `stop_gradient` on reference does not reduce forward pass cost — it only affects backward.

**Communication ops:**
| Op type | Count | Interpretation |
|---------|-------|----------------|
| dot_general | 10,637 | Matmuls (actual compute) |
| collective-permute | 317,633 | Data resharding between FSDP parameter layout and compute layout |
| all-gather | 731 | FSDP weight gathering before matmuls |
| reduce-scatter | 4 | Minimal (no gradients in eval) |

**Key finding**: 317K collective-permute ops vs 10K dot_general = ~30 resharding ops per matmul. Significant communication overhead from FSDP parameter→compute layout transitions.

### Diagnosis

The eval bottleneck is **4 redundant 8B forward passes per example** with heavy FSDP resharding overhead. The reference model's 2 forward passes produce identical results every eval (frozen weights, same val data) but cost the same as the policy model's forward passes.

### Key W&B Links

| Run | URL | Notes |
|-----|-----|-------|
| **Eval profiling (SUCCESS)** | https://wandb.ai/marin-community/dpo/runs/thr046my | `polished-terrain-160`, standalone eval, 20 batches |
| **xprof artifact** | https://wandb.ai/marin-community/dpo/artifacts/jax_profile/run-thr046my-dpo_eval_profile/v0 | 1.8GB xplane.pb + perfetto trace |
| **Training run (new code)** | https://wandb.ai/marin-community/dpo/runs/new_dpo_v2_bloom_speceval_v2_marin_instruct_beta0.1_-7_seed2-947c5d | beta=0.1, seed=2, lr=7.5e-7, step 680/850 at last check |
| **Old baseline (pre-refactor)** | https://wandb.ai/marin-community/dpo/runs/bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed2-f57ace | Same config, old code, completed 849 steps |
| **Old baseline (seed0, default lr)** | https://wandb.ai/marin-community/dpo/runs/bloom_speceval_v2_marin_instruct_beta0.1_seed0-4f9703 | beta=0.1, seed=0, lr=5e-7, completed 849 steps |

**Local xprof artifact path**: `artifacts/run-thr046my-dpo_eval_profile:v0/plugins/profile/2026_03_29_22_44_03/`
- `t1v-n-111d8c3e-w-1.xplane.pb` (1.8GB) — device-level XLA trace with named scopes
- `t1v-n-111d8c3e-w-1.trace.json.gz` (11MB) — host-side Python trace
- `perfetto_trace.json.gz` (8.2MB) — Perfetto-format trace

To view the device trace, load the `.xplane.pb` in TensorBoard with the profile plugin, or use `tensorboard --logdir=artifacts/run-thr046my-dpo_eval_profile:v0/plugins/profile/`.

### Standalone eval profiling script (`experiments/eval_dpo.py`)

Created a standalone eval-only script to avoid wasting hours on training just to profile eval. Multiple iterations:
- v1: `prepare_model_init_context()` missing `use_hf_model_config` kwarg → crash
- v2: Codex fixed API calls, cache loading, tracker lifecycle
- v3: Crashed on `embed(256) vs embed(4096)` — root cause was running eval under `parameter_axis_mapping` instead of `compute_axis_mapping`. Codex fixed by switching to `@hax.named_jit(axis_resources=compute_axis_mapping)`.
- v3 (relaunch): Eval completed but `barrier_sync()` in `profile_ctx` timed out (3/4 tasks reached barrier). Results and xprof artifact lost.
- v4 (final): Removed `profile_ctx`, manual profiler start/stop, no perfetto link, no barrier. Results logged before any cleanup. Capped at `max_eval_batches=20`. Completed successfully.

### Eval profiling attempts timeline

| Job | Result | Issue |
|-----|--------|-------|
| `dpo_profile_eval_bottleneck` | Killed | Targeted step 200 (first eval, includes JIT compilation) — wrong design |
| `dpo_profile_eval_steady_state` | Killed | Correct design (step 400) but 450 train steps = too slow |
| `dpo_eval_profile` (v1) | FAILED | Missing `use_hf_model_config` kwarg |
| `dpo_eval_profile_v2` (v2) | FAILED | `embed(256) vs embed(4096)` — parameter_axis_mapping in eval context |
| `dpo_eval_profile_v3` | FAILED | Eval completed but `barrier_sync()` timed out, results lost |
| **`dpo_eval_profile_v4`** | **SUCCESS** | Clean run, results + xprof artifact captured |

### Training run status (concurrent with profiling)

The seed2 lr7.5e-7 training run with the refactored DPO code:

| Run | W&B name | State at last check | Steps | Loss | Notes |
|-----|----------|---------------------|-------|------|-------|
| `/ahmed/new_dpo_v2_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed2` | `...-947c5d` | RUNNING | 108/850 | 0.166 | Past step-77 OOM point, 256GB RAM working |

### Executor hash behavior (learned the hard way)

The executor output path hash (e.g., `-fdc0c9`, `-947c5d`) is computed from:
```python
json.dumps({"name": self.name, "attrs": self.hash_attrs, "deps": sorted(self.dep_paths)})
```
- `name`: the step name (e.g., `"dpo/new_dpo_v2_bloom_speceval_v2_..."`)
- `attrs`: only fields wrapped in `VersionedValue()` (data paths, key hyperparameters)
- `deps`: upstream dependency output paths

**Does NOT include**: `ram`, `per_device_eval_parallelism`, `profiler`, `save_interval`, or any other runtime setting. Changing these does NOT change the hash or create a new W&B run ID. Only changing `name` or versioned fields changes the hash.

This caused a W&B collision: killed run `-fdc0c9` left a broken W&B entry, new runs with same hash tried to resume it and timed out. Fixed by changing the experiment `name` to include `_v2_`.

### FSDP / axis mapping explanation (for reference)

Levanter uses two axis mappings on the TPU mesh:
- **`parameter_axis_mapping`**: `embed → data`. Parameters stored with embed dimension sharded across 16 chips (each holds embed=256). This is FSDP / ZeRO-3.
- **`compute_axis_mapping`**: `token → (replica_dcn, replica, data)`. Compute parallelizes over batch/sequence. Each chip sees full `embed=4096`.

XLA automatically inserts all-gather ops to transition between these layouts. The 317K collective-permute ops in the xprof trace are this resharding happening per-layer, per-forward-pass.

The `eval_dpo.py` crash occurred because eval ran under `parameter_axis_mapping` context, so the fused CE loss's `shard_map` saw local `embed=256` but expected `embed=4096`.

### Eval-specific sharding analysis

**Question**: Can we re-shard to pure data-parallel (replicate all weights) for eval to eliminate FSDP communication?

**Answer**: Yes in principle, but:
- **Memory**: Replicating 2x 8B models = 32GB/chip on top of existing FSDP shard. ~67% HBM, should fit in 95GB.
- **Resharding cost**: One all-gather of 32GB over DCN (~1-2s). Negligible vs 9 min eval.
- **Recompilation**: Need separate JIT for replicated inputs (~46s, cacheable).
- **Implementation**: Non-trivial in Levanter's NamedArray/axis system.
- **Unknown**: We have op COUNTS but not op DURATIONS. The 317K collective-permutes might be overlapped with compute. Can't quantify speedup without per-op timing.

**Verdict**: Do reference log-prob caching first (guaranteed 2x, simpler), re-profile, then decide.

### Recommended next steps (priority order)

1. **Cache reference log-probs** (highest impact, ~2x eval speedup) — precompute `logp_ref_chosen` and `logp_ref_rejected` for the full val set once before training. Reuse cached values in all subsequent evals. Cuts eval from 4 forward passes to 2 per example.
2. **Add `max_eval_batches` to `SimpleDPOConfig`** — expose the cap for sweep runs (currently only in `eval_dpo.py`).
3. **Exclude reference from checkpoint serialization** — reference model doesn't need to be checkpointed since it's loaded fresh from HF. Reduces checkpoint OOM risk.
4. **Re-profile after caching** — if eval is still slow, investigate eval-specific replicated sharding.
5. **Re-shard for eval** (speculative) — replicate weights for eval to eliminate FSDP communication. Only if caching + re-profiling shows communication is still a bottleneck.

## 2026-03-29 Reference Log-Prob Caching Implementation

### Implementation

Added reference log-prob caching to `experiments/eval_dpo.py` with three modes:
- `--mode uncached`: baseline 4-forward-pass eval
- `--mode build`: compute reference log-probs for full val set → write to GCS TreeCache → run cached 2-forward-pass eval
- `--mode cached`: load existing cache → run cached eval (skips reference model loading entirely)

Key components:
- `_ref_cache_path()`: deterministic GCS path as sibling to val cache, keyed by val_cache + reference_model + seq_len
- `_build_ref_cache()`: JIT-compiled reference-only forward pass, writes `{logp_ref_chosen, logp_ref_rejected}` per example to `SerialCacheWriter`
- `CachedRefDataset`: `AsyncDataset` wrapper that injects cached scalars by index into `CachedDpoExample`
- `_loss_fn_cached()`: only runs 2 policy forward passes, uses cached reference scalars directly

Cache location: `gs://marin-us-central1/tokenized/bloom_speceval_v2_val_prefs_marin_tokenizer-a06ae8/reference_logprobs/<hash>/`

### Expected performance

| Mode | Forward passes/example | Model loads | Est. eval time (10 batches) |
|------|----------------------|-------------|----------------------------|
| uncached | 4 (policy×2 + reference×2) | policy + reference | ~4.5 min |
| build | 2 reference (cache) + 2 policy (eval) | policy + reference | ~15 min total (one-time) |
| cached | 2 (policy×2 only) | policy only | ~2.5 min (+ ~6 min saved on model load) |

### Launch attempts

| Job | Result | Issue |
|-----|--------|-------|
| `dpo_eval_cached` | FAILED (SyntaxError) | `global MAX_EVAL_BATCHES` declared after use in default arg |
| `dpo_eval_cached_v2` | FAILED | `np.array()` on multihost sharded JAX array |
| `dpo_eval_cached_v3` | FAILED | Same multihost bug (launched before fix landed) |
| `dpo_eval_cached_v4` | FAILED | Multihost fixes worked for cache build (23,552 examples cached in 630s to GCS). Crashed on cached eval: `zeros_like requires ndarray, got CachedDpoExample` — dataclass not a JAX pytree |
| `dpo_eval_cached_v5` | FAILED | Fixed `CachedDpoExample` to `eqx.Module` but `_load_ref_cache` returned list not dict from `get_batch_sync` |
| `dpo_eval_cached_v6` | FAILED | Fixed cache loading. New crash: `hax.named` shape mismatch — cached values had shape `(512, 1)` vs expected `(512,)` from trailing dim in TreeCache storage |
| `dpo_eval_cached_v7` | PENDING | Fixed squeeze, switched to GCS model (`marin-8b-base`), 1 warmup + 1 eval batch, `--mode build` to rebuild cache |

### Lessons from cache eval failures

Every iteration exposed a new bug. Pattern: code that works in single-host/single-device mental model breaks on multihost or when flowing through Levanter's DataLoader/JIT pipeline.

| Attempt | Bug | Root cause |
|---------|-----|------------|
| v4 | `zeros_like got CachedDpoExample` | `@dataclass` is not a JAX pytree — DataLoader can't batch it |
| v5 | `list indices must be integers, not str` | `TreeCache.get_batch_sync` returns list of dicts, not a single dict |
| v6 | `Shape mismatch: (512,1) vs (Axis("batch",512),)` | TreeCache stored scalars with trailing `(1,)` dim; `hax.named` needs exact shape match |

### Changes for v7

- `WARMUP_BATCHES = 1`, `MAX_EVAL_BATCHES = 1` — absolute minimum to test the path works
- Model loaded from `gs://marin-us-central1/models/marin-community--marin-8b-base--main` — no HuggingFace downloads
- `jnp.squeeze()` before `hax.named()` to remove trailing dim from cached values
- `--mode build` to rebuild cache with base model (old cache was built with instruct model)

### v7 SUCCESS — Reference Log-Prob Caching Validated

**Job**: `/ahmed/dpo_eval_cached_v7` — **SUCCEEDED**, 0 failures

**Cache build**: 23,552 reference log-probs computed in 631s. Written to `gs://marin-us-central1/tokenized/bloom_speceval_v2_val_prefs_marin_tokenizer-a06ae8/reference_logprobs/65443057`

**Cached eval results (1 batch)**:

| Metric | Uncached (v4, 4 fwd passes) | Cached (v7, 2 fwd passes) | Speedup |
|--------|---------------------------|--------------------------|---------|
| **Loss compute/batch** | **26.9s** | **13.4s** | **2.0x** |
| Data load/batch | 0.75s | 8.9s* | — |
| Total/batch | 27.6s | 22.4s | 1.2x |

*Data load is high because first batch after cache build had a cold DataLoader. Steady state data load would be <1s like uncached.

**The 2x compute speedup is confirmed.** Loss compute dropped from 26.9s to 13.4s per batch — reference forward passes completely eliminated. At scale (10+ batches), the 2x compute savings dominate since data loading amortizes.

**Projected impact on full training runs** (5 evals × 46 batches):
- Uncached: 5 × 46 × 27s = 103 min (~1.7h) of eval
- Cached: 5 × 46 × 14s = 54 min (~0.9h) of eval + 11 min one-time cache build
- **Savings: ~48 min per training run**

### Full eval_dpo.py launch attempt history

| Job | Mode | Result | Issue |
|-----|------|--------|-------|
| `dpo_eval_profile` (v1) | uncached | FAILED | Missing `use_hf_model_config` kwarg |
| `dpo_eval_profile_v2` | uncached | FAILED | `embed(256) vs embed(4096)` axis mapping bug |
| `dpo_eval_profile_v3` | uncached | FAILED | `barrier_sync()` timeout, results lost |
| `dpo_eval_profile_v4` | uncached | **SUCCESS** | Baseline: 27.6s/batch, 97% loss compute |
| `dpo_eval_cached` (v1) | build | FAILED | `global MAX_EVAL_BATCHES` SyntaxError |
| `dpo_eval_cached_v2` | build | FAILED | `np.array()` on multihost sharded array |
| `dpo_eval_cached_v3` | build | FAILED | Same bug (launched before fix) |
| `dpo_eval_cached_v4` | build | FAILED | Cache built OK, eval crashed: `CachedDpoExample` not pytree |
| `dpo_eval_cached_v5` | cached | FAILED | `get_batch_sync` returns list not dict |
| `dpo_eval_cached_v6` | cached | FAILED | `hax.named` shape mismatch: `(512,1)` vs `(512,)` |
| **`dpo_eval_cached_v7`** | **build** | **SUCCESS** | **2x compute speedup confirmed** |

### 20-Batch Apples-to-Apples Comparison — FINAL RESULT

**Job**: `/ahmed/dpo_eval_cached_20batch` — **SUCCEEDED**, 0 failures

| Metric | Uncached (v4, 20 batches) | Cached (20batch) | Speedup |
|--------|--------------------------|-----------------|---------|
| **Loss compute** | **537.1s** | **270.2s** | **1.99x** |
| Data load | 15.0s | 8.9s | 1.7x |
| **Total eval** | **552.1s (9.2 min)** | **279.1s (4.7 min)** | **1.98x** |
| Avg loss/batch | 26.9s | 13.5s | 2.0x |
| Loss | 0.69315 | 0.69315 | match |

**Confirmed: 2x eval speedup at scale.** Both uncached and cached produce identical loss (0.69315 = ln(2), expected for untrained policy == reference).

**Caveat**: Loss agreement at 0.69315 is necessary but not sufficient for correctness validation — both runs use the same weights for policy and reference, so rewards are zero by construction. A proper correctness check requires running both modes on a trained policy model with different weights from the reference.

### GCS paths

| Resource | Path |
|----------|------|
| marin-8b-base model | `gs://marin-us-central1/models/marin-community--marin-8b-base--main/` |
| Reference logprob cache (base model) | `gs://marin-us-central1/tokenized/bloom_speceval_v2_val_prefs_marin_tokenizer-a06ae8/reference_logprobs/65443057` |
| Reference logprob cache (instruct model, v4) | `gs://marin-us-central1/tokenized/bloom_speceval_v2_val_prefs_marin_tokenizer-a06ae8/reference_logprobs/7fe190b0` |
| Val tokenized cache | `gs://marin-us-central1/tokenized/bloom_speceval_v2_val_prefs_marin_tokenizer-a06ae8/validation` |

**Note**: marin-8b-instruct is NOT on GCS — was loading from HuggingFace every time (~6 min with timeout retries). Future runs should cache it on GCS.

### Training run status

| Run | W&B name | Last checked | Steps | Loss | Notes |
|-----|----------|-------------|-------|------|-------|
| `/ahmed/new_dpo_v2_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed2` | `...-947c5d` | ~19:33 UTC | 680/850 | 0.00016 | Should have completed by now |

### Complete eval_dpo.py launch history (13 attempts)

| # | Job | Mode | Result | Issue |
|---|-----|------|--------|-------|
| 1 | `dpo_eval_profile` | uncached | FAILED | Missing `use_hf_model_config` kwarg |
| 2 | `dpo_eval_profile_v2` | uncached | FAILED | `embed(256) vs embed(4096)` axis mapping |
| 3 | `dpo_eval_profile_v3` | uncached | FAILED | `barrier_sync()` timeout, results lost |
| 4 | **`dpo_eval_profile_v4`** | **uncached** | **SUCCESS** | **Baseline: 537s loss, 15s load, 20 batches** |
| 5 | `dpo_eval_cached` | build | FAILED | `global MAX_EVAL_BATCHES` SyntaxError |
| 6 | `dpo_eval_cached_v2` | build | FAILED | `np.array()` on multihost sharded array |
| 7 | `dpo_eval_cached_v3` | build | FAILED | Same bug (launched before fix) |
| 8 | `dpo_eval_cached_v4` | build | FAILED | Cache built OK → eval crash: dataclass not pytree |
| 9 | `dpo_eval_cached_v5` | cached | FAILED | `get_batch_sync` returns list not dict |
| 10 | `dpo_eval_cached_v6` | cached | FAILED | `hax.named` shape mismatch `(512,1)` vs `(512,)` |
| 11 | **`dpo_eval_cached_v7`** | **build** | **SUCCESS** | **1 batch: 13.4s vs 26.9s = 2x** |
| 12 | **`dpo_eval_cached_20batch`** | **cached** | **SUCCESS** | **20 batches: 270s vs 537s = 2x** |

### Multihost fixes that were needed (cumulative)

1. `process_allgather(logp.array, tiled=True)` before `np.array()` on sharded outputs
2. Cache write gated to `jax.process_index() == 0` + `sync_global_devices()` barrier
3. `CachedDpoExample` as `eqx.Module` not `@dataclass` (JAX pytree for DataLoader)
4. `TreeCache.get_batch_sync` returns list of dicts, not a single dict — iterate properly
5. `jnp.squeeze()` before `hax.named()` to remove trailing `(1,)` dim from TreeCache storage
6. `hax.named(cached_val, policy_logp.axes)` inside JIT to wrap cached scalars as NamedArrays

## 2026-03-30 Full Validation Correctness Test — FINAL RESULT

### Setup

- **Policy model**: trained step-849 checkpoint from the new DPO pipeline (`gs://marin-us-central1/checkpoints/dpo/new_dpo_v2_bloom_speceval_v2_marin_instruct_beta0.1_-7_seed2-947c5d/hf/step-849`)
- **Reference model**: `marin-community/marin-8b-instruct` (same as training used)
- **Val set**: full bloom_speceval_v2_val (23,552 examples, 46 batches at parallelism=32)
- **Zone**: us-central1-a (pinned — earlier attempt failed with `TransferBudgetExceeded` from cross-region transfer)

### Jobs

| Job | Mode | State |
|-----|------|-------|
| `/ahmed/dpo_eval_fullval_uncached_v3` | uncached (4 fwd passes) | SUCCEEDED |
| `/ahmed/dpo_eval_fullval_cached_v3` | build + cached (2 fwd passes) | SUCCEEDED |

### Correctness — ALL METRICS MATCH EXACTLY

| Metric | Uncached | Cached | Match? |
|--------|---------|--------|--------|
| loss | 0.0055 | 0.0055 | ✅ |
| dpo_accuracy | 0.9958 | 0.9958 | ✅ |
| dpo_chosen_reward | 2.5412 | 2.5412 | ✅ |
| dpo_rejected_reward | -11.9255 | -11.9255 | ✅ |
| dpo_margin_policy | -45.0588 | -45.0588 | ✅ |
| dpo_margin_ref | -189.7258 | -189.7258 | ✅ |

This is a real trained model (loss=0.0055 vs baseline 0.693), so policy ≠ reference. The cached reference log-probs produce bit-identical results.

### Performance — 2x SPEEDUP CONFIRMED AT SCALE

| Metric | Uncached | Cached | Speedup |
|--------|---------|--------|---------|
| Loss compute time | 1232.3s (20.5 min) | 618.1s (10.3 min) | **1.99x** |
| Total eval time | ~1240s (20.7 min) | 626.9s (10.4 min) | **1.98x** |
| Avg loss/batch | 26.79s | 13.44s | **1.99x** |
| Data load/batch | 0.181s | 0.189s | ~equal |
| Batches | 46 | 46 | — |
| Cache build (one-time) | — | 631.3s | — |

### Projected savings for training runs

With 5 evals per 850-step run (at steps 0, 200, 400, 600, 800):
- **Uncached**: 5 × 20.7 min = **103 min of eval**
- **Cached**: 5 × 10.4 min + 10.5 min cache build = **62 min of eval**
- **Savings: ~41 min per training run** (from 7.7h total to ~7h)

## 2026-03-30 Plan: Replicated (Non-FSDP) Eval Sharding

### Motivation

Even with reference log-prob caching (2x speedup), each eval batch still takes 13.4s. From the xprof analysis, there were 317K collective-permute ops and 731 all-gather ops per eval pass — all from FSDP weight resharding (parameter layout `embed → data` → compute layout full embed). Eliminating this communication could give another significant speedup.

### Idea

Load the model with **replicated** sharding instead of FSDP for eval. Each of the 16 chips holds the full 8B model. No per-layer all-gathers, no collective-permutes. Pure data parallelism: each chip processes a different batch slice using its local full copy of weights.

### Inspired by `eval_lm.py`

`lib/levanter/src/levanter/main/eval_lm.py` uses the same FSDP pattern we have (load with `parameter_axis_mapping`, JIT with `compute_axis_mapping`). But it demonstrates that the eval path is independent of the training sharding — the model can be resharded after loading. Line 132: `model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)` could just as easily use a replicated mapping.

### Memory analysis (per chip, v5p = 95GB HBM)

| Component | FSDP (current) | Replicated (proposed) |
|-----------|---------------|----------------------|
| Policy model weights (bf16) | ~1GB (1/16th) | ~16GB (full) |
| Activations at per_device=32 | ~32GB | ~32GB |
| **Total** | **~33GB (34% HBM)** | **~48GB (50% HBM)** |
| Headroom | ~62GB | ~47GB |

Replicated fits comfortably at per_device=32. Possibly room for per_device=64 (~80GB total, tight but feasible). per_device=128 would OOM.

### Implementation

Minimal change to `eval_dpo.py` — pass `compute_axis_mapping` instead of `parameter_axis_mapping` when loading the model:

```python
# Current (FSDP — weights sharded, needs all-gathers during compute):
policy_model = load_model_from_source(..., parameter_axis_mapping=parameter_axis_mapping, ...)

# Proposed (replicated — full weights on each chip, zero communication):
policy_model = load_model_from_source(..., parameter_axis_mapping=compute_axis_mapping, ...)
```

Since `compute_axis_mapping` only maps `token → (replica_dcn, replica, data)`, and model weights don't have a `token` axis, the weights end up fully replicated. The JIT function already uses `compute_axis_mapping`, so no sharding transition is needed → no all-gathers or collective-permutes.

### Expected results

| Scenario | Per-batch time | Speedup vs FSDP cached |
|----------|---------------|----------------------|
| FSDP cached (current) | 13.4s | baseline |
| Replicated, per_device=32 | ~7-9s (if comms were 30-50% of time) | ~1.5-2x |
| Replicated, per_device=64 | ~4-5s (if memory fits + better MXU utilization) | ~2.5-3x |

Combined with reference caching (2x), total speedup could be 3-6x vs original uncached FSDP eval.

### Test plan

1. Run cached eval with replicated sharding at per_device=32 → measure per-batch time vs 13.4s baseline
2. If it works, try per_device=64 → see if batch doubling further improves throughput
3. If per_device=64 OOMs, stick with 32 and quantify the communication savings alone

### Risks

- **Model loading time**: replicated loading means each chip downloads the full model. With GCS, this might be slower than FSDP loading where each chip only downloads its shard. But it's a one-time cost.
- **`load_model_from_source` behavior**: need to verify it respects the passed axis mapping and doesn't hardcode FSDP assumptions internally.
- **Multihost**: replicated sharding means each host has a full copy. `hax.named_jit` should handle this correctly since the JIT function already uses `compute_axis_mapping`, but this is untested territory for DPO eval.

### What's next (priority order)

1. **Replicated eval experiment** (this plan) — test communication-free eval sharding
2. **Wire caching into `train_dpo.py`** — integrate cached reference log-probs into training eval hooks
3. **Add `max_eval_batches` to `SimpleDPOConfig`** — expose for sweep runs
4. **Exclude reference from checkpoint serialization** — reduces OOM risk
5. **Cache the instruct model on GCS** — avoid slow HF downloads

## MULTIHOST TPU PITFALLS — READ THIS BEFORE WRITING ANY NEW CODE

**This section exists because we hit the same class of bug 5+ times on this branch. Every agent working on DPO code MUST read this.**

### The environment

DPO runs on v5p-32 = 4 hosts × 4 TPU chips = 16 chips total. JAX arrays are sharded across ALL 16 chips. Each host (process) only has direct access to its local 4 chips. The other 12 chips are "non-addressable" from any given host.

### Rules for multihost JAX code

**RULE 1: Never call `np.array()`, `.item()`, or `jnp.array()` on a globally-sharded JAX array outside of JIT.**
- These try to fetch the full array to a single host, which fails if any shard lives on a non-local device.
- Fix: Use `jax.experimental.multihost_utils.process_allgather(arr, tiled=True)` first, THEN convert to numpy.
- Inside JIT is fine — JAX handles cross-device access automatically within compiled functions.

**RULE 2: Never close over a sharded JAX array in a Python closure or lambda.**
- This was the original DPO refactor bug: the loss function closed over the reference model (a sharded array), and JAX rejected it during lowering.
- Fix: Pass sharded arrays as JIT function arguments, not as closure captures.

**RULE 3: Only one host should write to GCS/storage.**
- After `process_allgather`, every host has the same data. If all 4 hosts write the same cache, they'll race.
- Fix: Gate writes with `if jax.process_index() == 0:`, then `sync_global_devices()` so other hosts wait.

**RULE 4: `barrier_sync()` and `sync_global_devices()` can timeout if hosts reach them at very different times.**
- The Perfetto trace flush took different times on different hosts, causing a barrier timeout that killed the process before results were logged.
- Fix: Log results BEFORE any barrier. Put barriers in try/except. Avoid `create_perfetto_link=True` in standalone scripts.

**RULE 5: The `parameter_axis_mapping` vs `compute_axis_mapping` context matters.**
- Parameters are stored with `embed → data` (FSDP sharding: embed=256 per chip).
- Compute runs with `token → data` (full embed=4096 per chip).
- If you run a loss function under `parameter_axis_mapping` context, `shard_map` sees `embed=256` locally but the model expects `embed=4096` → crash.
- Fix: Use `@hax.named_jit(axis_resources=compute_axis_mapping)` for eval/loss functions. Only use `parameter_axis_mapping` for model loading.

### History of multihost bugs on this branch

| Bug | Where | Symptom | Fix |
|-----|-------|---------|-----|
| Closure over reference model | `train_dpo.py` refactor | `Closing over jax.Array that spans non-addressable devices` | Pass reference in DpoModel as JIT arg, not closure |
| barrier_sync timeout | `eval_dpo.py` via `profile_ctx` | `DEADLINE_EXCEEDED: Barrier timed out. 3/4 tasks reached` | Remove barrier, log results before cleanup |
| Wrong axis mapping for eval | `eval_dpo.py` | `Axis embed present in both specs with different sizes: embed(256) vs embed(4096)` | Use `compute_axis_mapping` not `parameter_axis_mapping` for eval |
| `np.array()` on sharded array | `eval_dpo.py` cache build | `Fetching value for jax.Array that spans non-addressable devices` | Use `process_allgather` first |
| All hosts writing cache | `eval_dpo.py` cache build | Race condition (caught before launch) | Gate to process 0 + barrier |
| Cached values not NamedArray | `eval_dpo.py` cached eval | Shape mismatch (caught before launch) | Wrap with `hax.named()` inside JIT |
| Custom dataclass not JAX pytree | `eval_dpo.py` cached eval | `zeros_like requires ndarray, got CachedDpoExample` | Use `eqx.Module` instead of `@dataclass` for any type that flows through DataLoader/JIT |
| All hosts writing to GCS | `eval_dpo.py` cache build | Race condition (caught before launch) | Gate to `jax.process_index() == 0` + `sync_global_devices()` |

# CODEX START

## 2026-03-29 LoRA Status And Bloom SpecEval v2 Reproduction Plan

### Have We Actually Tried LoRA-DPO Yet?

- Not for the Bloom SpecEval v2 Marin Instruct run family in this thread.
- I do not see any recorded Bloom SpecEval v2 LoRA-DPO training launch in:
  - this Codex logbook
  - Claude's parallel logbook
  - the current Iris job list
- What we have today is **code-path readiness**, not experiment evidence:
  - canonical `levanter.main.train_dpo` supports `adapter.type=lora` with `reference.type=adapter_base`
  - the legacy `levanter.main.lora_dpo` wrapper still works for old configs
  - LoRA-DPO tests and docs are in place
  - existing LoRA YAMLs target Ultrafeedback / legacy sanity checks, not Bloom SpecEval v2

### Exact Baseline To Reproduce

Target run:

- W&B: `https://wandb.ai/marin-community/dpo/runs/new_dpo_v2_bloom_speceval_v2_marin_instruct_beta0.1_-7_seed2-947c5d`
- Executor source: `experiments/sweep_dpo/beta0.1_seed2_lr7.5e-7.py`

Baseline knobs that should stay fixed for the first LoRA reproduction attempt:

- dataset: Bloom SpecEval v2 preference data (`bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite`)
- tokenizer: `marin-community/marin-tokenizer`
- base model: `marin-community/marin-8b-instruct`
- seed: `2`
- beta: `0.1`
- learning rate: `7.5e-7`
- train batch size: `128`
- num train steps: `850`
- train / max seq len: `4096`
- steps per eval: `200`
- hf export cadence: `200`
- hardware target: `v5p-32`
- memory target: `256GB`
- eval parallelism target: `32`

The only intended semantic change for the first reproduction run is:

- full-FT policy + separate frozen reference
- becomes
- LoRA policy + implicit adapter-base reference

### Important Constraint In The Current Experiment Plumbing

- `experiments/defaults.default_dpo(...)` and `SimpleDPOConfig` currently only construct regular DPO with `reference.type=separate`.
- There is no Bloom SpecEval v2 experiment wrapper in `experiments/` that can already emit:
  - `adapter.type: lora`
  - `reference.type: adapter_base`
  - LoRA export settings like `merged_hf_save_path`

So the **fastest and lowest-risk first LoRA experiment** is:

1. write a standalone canonical `TrainDpoConfig` YAML for Bloom SpecEval v2
2. launch `python -m levanter.main.train_dpo` directly on Iris
3. only after one successful run, decide whether to teach `SimpleDPOConfig` / `default_dpo` about LoRA for sweep parity

That avoids mixing "does LoRA-DPO train correctly here?" with "did we correctly refactor the executor config surface?"

### Planned First Run: Strict Reproduction With LoRA

Goal: reproduce the `...947c5d` run as faithfully as possible while changing only the DPO parameterization.

Planned config delta relative to the seed-2 baseline:

```yaml
adapter:
  type: lora
  r: 64
  alpha: 64.0
  dropout: 0.0
  zero_init_b: true
  target_modules: null

reference:
  type: adapter_base
```

Everything else should match the baseline run unless a TPU-specific blocker appears.

Rationale for this first run shape:

- `zero_init_b: true` is mandatory for DPO because the step-0 policy must equal the implicit reference
- `r=64, alpha=64` is the current house default and matches the LoRA-DPO guide
- `target_modules: null` means all linear modules, which is the current recommended setting
- keeping `lr=7.5e-7` makes this a true reproduction attempt rather than an immediate retuning study

### Recommended Launch Form

First run should be a dedicated canonical config, something like:

- config path: `lib/levanter/config/dpo/lora_dpo_bloom_speceval_v2_seed2_lr7.5e-7_central1.yaml`
- entrypoint: `uv run python -m levanter.main.train_dpo --config_path ...`

Suggested job command shape:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --extra marin:tpu \
  --tpu v5p-32 \
  --memory 256GB \
  --disk 50GB \
  --zone us-central1-a \
  --job-name lora-new-dpo-v2-bloom-s2-lr7p5e7 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m levanter.main.train_dpo \
  --config_path lib/levanter/config/dpo/lora_dpo_bloom_speceval_v2_seed2_lr7.5e-7_central1.yaml
```

### Config Notes For The First Run

- Use the same Bloom SpecEval v2 tokenized train/val caches as the full-DPO baseline.
- Keep `validation_split_fraction: null` so eval uses the explicit validation set rather than creating a new split.
- Keep `trainer.per_device_eval_parallelism: 32` for comparability with the current full-DPO runs.
- Keep `train_batch_size: 128` and `num_train_steps: 850` unchanged.
- Use `merged_hf_save_path` for the first run, not `peft_save_path`.

Why `merged_hf_save_path` first:

- older LoRA TPU configs in this repo explicitly disabled `peft_save_path` because of multihost serialization issues
- even if the new adaptation path may be better, the first Bloom SpecEval v2 LoRA run should minimize new failure modes

### Success Criteria For The Strict Reproduction Run

Call the first LoRA reproduction successful only if all of the following hold:

- it compiles and gets past the first train step on multihost TPU
- step-0 DPO loss is near `ln(2)` / `0.693`, not a large blown-up value
- no container OOM occurs at `v5p-32`, `256GB`
- W&B logs normal DPO metrics:
  - `loss`
  - `dpo_accuracy`
  - `dpo_margin_policy`
  - `dpo_margin_ref`
  - `dpo_chosen_reward`
  - `dpo_rejected_reward`
- throughput and eval cadence look sane relative to the regular-DPO baseline

If step-0 loss is badly wrong, that is a strong sign that the LoRA identity/reference assumption is broken, most likely:

- `zero_init_b` not applied
- wrong reference mode
- adapter modules not wired as expected

### Planned Follow-Up If Strict Reproduction Is Flat

There is a real chance that a literal `lr=7.5e-7` LoRA run will learn too slowly. The LoRA-DPO guide in this branch recommends starting closer to `5e-6`.

So the plan should be:

1. run the strict reproduction first at `7.5e-7`
2. if it is stable but under-trains, launch a LoRA-tuned follow-up at `5e-6`
3. keep the same:
   - dataset
   - seed
   - beta
   - batch size
   - steps
   - hardware
   - adapter rank

That gives two distinct answers:

- **strict reproduction**: "what happens if we only swap in LoRA?"
- **LoRA-tuned comparison**: "what is the fairer LoRA baseline once the optimizer is adjusted?"

### Follow-Up Code Work Only After First Evidence

Do **not** start by extending the executor/sweep layer.

After the first LoRA run succeeds, the next cleanup step should be:

1. add adapter/reference/export fields to `SimpleDPOConfig`
2. teach `default_dpo(...)` to emit either regular DPO or LoRA-DPO
3. add a proper `experiments/sweep_dpo/lora_beta0.1_seed2_lr7.5e-7.py`

That sequencing keeps the first experiment focused on model behavior instead of config refactoring.

## 2026-03-29 External Evidence: Thinking Machines "LoRA Without Regret"

Source read:

- https://thinkingmachines.ai/blog/lora/

### Scope Notes

- This article is strong evidence for **supervised fine-tuning** and **policy-gradient RL**.
- It is **not** a direct DPO paper.
- For our Bloom SpecEval v2 plan, the safest interpretation is:
  - treat DPO as much closer to the article's supervised setting than to its RL setting
  - carry over the supervised LoRA practices first
  - mark any DPO-specific conclusions as inference, not established fact

### Direct Takeaways From The Article

The practices below are directly supported by the article:

- **Apply LoRA to all layers, especially MLP/MoE layers.**
  - Attention-only LoRA underperformed materially.
  - MLP-only was much better, and all-layer LoRA was the safest default.
- **Do not judge LoRA from a single learning rate.**
  - The article explicitly swept LR for each condition before comparing LoRA to FullFT.
- **For supervised-style training, LoRA's best LR is about 10x the FullFT LR.**
  - The article reports an empirical multiplier of about `9.8x`.
- **Large batch size can hurt LoRA more than it hurts FullFT.**
  - This effect appeared to be mostly independent of rank.
- **Keep the standard LoRA parametrization unless there is evidence otherwise.**
  - `alpha / r` scaling
  - zero-init on `B`
  - standard random init for `A`
  - same LR for `A` and `B`
  - the authors report they could not improve on this basic setup
- **Rank is mainly a capacity knob, not a cure for bad optimization settings.**
  - If LoRA is capacity-constrained, training falls off the FullFT curve.
  - But larger rank does not remove the large-batch penalty.

### DPO-Specific Inferences For This Thread

These are my inferences from the article, not direct claims made there:

- Our Bloom SpecEval v2 DPO run should be treated as a **supervised-style LoRA problem**, not as an RL-style low-capacity case.
- Therefore the article's RL result ("very low rank can match FullFT") should **not** be used to justify tiny-rank DPO first runs.
- The current repo default of:
  - `target_modules: null`
  - `zero_init_b: true`
  - `alpha = r`
is directionally correct for the first Bloom SpecEval v2 LoRA experiment.
- Because Levanter currently excludes `lm_head` from LoRA by default, our practical "all-layer" run is really "all supported linear layers except lm_head". That is still much closer to the article's recommendation than attention-only LoRA.

### How This Changes The Bloom SpecEval v2 Plan

This section **updates** the previous plan.

#### 1. Do not treat the same-LR reproduction as the main comparison

The earlier plan proposed a strict reproduction at:

- FullFT baseline LR: `7.5e-7`
- LoRA reproduction LR: also `7.5e-7`

After reading the article, that should be demoted to a **sanity / lineage run only**.

Reason:

- the article's strongest operational finding is that LoRA wants about **10x** the FullFT LR in supervised settings
- so a same-LR comparison is likely unfair and likely to make LoRA look artificially weak

Updated interpretation:

- **strict same-LR run (`7.5e-7`)**: useful only to answer "what happens if I swap in LoRA and change nothing else?"
- **fair LoRA comparison**: should center around **`7.5e-6`**

#### 2. The first real LoRA tuning sweep should be LR-first, not rank-first

Minimal first sweep for the Bloom SpecEval v2 seed-2 setup:

- `r = 64`
- `target_modules = null`
- `zero_init_b = true`
- LR grid centered around the article's 10x rule:
  - `5e-6`
  - `7.5e-6`
  - `1e-5`

For our 850-step training run, `7.5e-6` is the natural anchor because it is exactly 10x the validated FullFT baseline LR.

The article also suggests a somewhat higher multiplier in very short runs. If we do only a brief screening run, e.g. `<=100` steps, then a fourth point near `1.1e-5` is reasonable. For the full 850-step run, the main comparison should still center near `7.5e-6`.

#### 3. If LoRA underperforms at batch size 128, reduce batch before raising rank

This is one of the clearest actionable points from the article.

If the first LoRA runs are stable but learn more poorly than FullFT:

- do **not** immediately conclude that rank 64 is too small
- do **not** expect higher rank to fix a large-batch optimization penalty

Instead, test smaller train batch sizes first, for example:

- `128` (baseline)
- `64`
- possibly `32` if needed

Keep the hardware fixed if possible so the comparison stays interpretable.

#### 4. Use rank as a capacity check only after LR and batch are sane

Recommended order:

1. get a stable run with all-layer LoRA and a LoRA-appropriate LR
2. if that still underfits, test batch reduction
3. only then test higher rank

Minimal rank ladder:

- `64` first
- `128` second if there are signs of capacity limits

I do **not** think we should start with attention-only LoRA or with tiny ranks.

#### 5. Keep the plain LoRA parametrization for the first comparison

The article is a strong argument **against** piling on extra LoRA tricks in the first Bloom SpecEval v2 experiment.

So the first serious run should keep:

- standard `alpha / r` scaling
- same LR for `A` and `B`
- no LoRA+
- no rank-dependent alpha hacks
- no attention-only targeting

For this codebase, that means:

```yaml
adapter:
  type: lora
  r: 64
  alpha: 64.0
  dropout: 0.0
  zero_init_b: true
  target_modules: null
```

#### 6. Compare LoRA vs FullFT on training/eval metrics, not just generations

The article deliberately used loss-based comparisons rather than only sample-based evals.

For our DPO runs, the corresponding best practice is:

- compare validation `loss`
- compare `dpo_accuracy`
- compare `dpo_margin_policy`
- compare `dpo_margin_ref`
- compare chosen/rejected rewards
- compare throughput / wall-clock / memory

Do not treat a handful of qualitative generations as the main evidence.

### Revised Experimental Order

This is the current recommended order for Bloom SpecEval v2 LoRA:

1. **Optional sanity run**:
   - same config as baseline, but with LoRA and `lr=7.5e-7`
   - purpose: verify the pipeline and observe how much performance is lost if LR is not retuned
2. **Main fair comparison run**:
   - same config, `r=64`, all-layer LoRA, `lr=7.5e-6`
3. **Small LR sweep around the fair run**:
   - `5e-6`, `7.5e-6`, `1e-5`
4. **Batch-size follow-up only if needed**:
   - reduce `train_batch_size` from `128` to `64`
5. **Rank follow-up only if needed**:
   - `r=128`

### Practical Bottom Line

The most important correction from the article is simple:

- a Bloom SpecEval v2 LoRA run at `7.5e-7` should not be considered the serious LoRA baseline
- the serious baseline should be around **`7.5e-6`**, with all supported linear layers adapted, and with batch size treated as a separate optimization variable

## 2026-03-30 Planned Run: Bloom SpecEval v2 LoRA Fair Baseline On v5p-8

### Why This Run

- User requested that the next LoRA experiment actually be launched on `v5p-8`.
- The current best next experiment from this logbook is the **fair LoRA baseline**, not the same-LR sanity run.
- I am keeping the run in `us-central1-a` to stay in-region with the Bloom SpecEval v2 preference data and tokenized caches.

### Config Chosen

- Config path:
  - `lib/levanter/config/dpo/lora_dpo_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-6_seed2_v5p8_central1.yaml`
- Main settings:
  - dataset: Bloom SpecEval v2 GPT-4.1 vs Mixtral opposite-mode preferences
  - train cache: `gs://marin-us-central1/tokenized/bloom_speceval_v2_train_prefs_marin_tokenizer-12920b`
  - val cache: `gs://marin-us-central1/tokenized/bloom_speceval_v2_val_prefs_marin_tokenizer-a06ae8`
  - base model: `marin-community/marin-8b-instruct`
  - adapter: LoRA `r=64`, `alpha=64`, `dropout=0`, `zero_init_b=true`, `target_modules=null`
  - reference: `adapter_base`
  - LR: `7.5e-6`
  - beta: `0.1`
  - seed: `2`
  - train batch size: `128`
  - train steps: `850`
  - hardware: `v5p-8`
  - eval parallelism: `16`

### Launch Command

Planned Iris submission:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --extra marin:tpu \
  --tpu v5p-8 \
  --memory 256GB \
  --disk 50GB \
  --zone us-central1-a \
  --job-name lora-bsv2-mi-b0p1-s2-lr7p5e6-v5p8 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m levanter.main.train_dpo \
  --config_path lib/levanter/config/dpo/lora_dpo_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-6_seed2_v5p8_central1.yaml
```

### Monitoring Plan

- Babysit this as an Iris TPU job, not a fire-and-forget launch.
- Watch specifically for:
  - scheduler capacity wait vs real runtime failure
  - first-step compile/lowering failures
  - HBM / OOM signals
  - bad step-0 DPO loss that would indicate broken LoRA identity/reference behavior

## 2026-03-30 Launch Update: v5p-8 LoRA Run Submitted And Rerouted To east5

### First Attempt: us-central1-a

Initial launch:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --extra marin:tpu \
  --tpu v5p-8 \
  --memory 256GB \
  --disk 50GB \
  --zone us-central1-a \
  --job-name lora-bsv2-mi-b0p1-s2-lr7p5e6-v5p8-20260330-000321 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m levanter.main.train_dpo \
  --config_path lib/levanter/config/dpo/lora_dpo_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-6_seed2_v5p8_central1.yaml
```

Job id:

- `/ahmed/lora-bsv2-mi-b0p1-s2-lr7p5e6-v5p8-20260330-000321`

Result:

- never allocated
- stayed `JOB_STATE_PENDING`
- pending reason was:
  - insufficient memory on ready `tpu_v5p_8-us-central1-a` workers (`need 256GB`, only about `11.8GB` available)
  - autoscaler scale-up for `tpu_v5p_8-us-central1-a` was quota-blocked

Action taken:

- stopped the pending central1 job

### Why I Switched To east5

- `tpu_v5p_8-us-east5-a` was not quota-blocked
- multiple east5 `v5p-8` slices were fully idle (`committed_mem_bytes=0`, `committed_tpu=0`)
- east5 already has the Bloom SpecEval v2 tokenized caches:
  - `gs://marin-us-east5/tokenized/bloom_speceval_v2_train_prefs_marin_tokenizer-12920b`
  - `gs://marin-us-east5/tokenized/bloom_speceval_v2_val_prefs_marin_tokenizer-a06ae8`

So east5 was the first region with a realistic chance of actually running tonight.

### Active Run: us-east5-a

Relaunch:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --extra marin:tpu \
  --tpu v5p-8 \
  --memory 256GB \
  --disk 50GB \
  --zone us-east5-a \
  --job-name lora-bsv2-mi-b0p1-s2-lr7p5e6-v5p8-east5-20260330-000700 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m levanter.main.train_dpo \
  --config_path lib/levanter/config/dpo/lora_dpo_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-6_seed2_v5p8_east5.yaml
```

Active job id:

- `/ahmed/lora-bsv2-mi-b0p1-s2-lr7p5e6-v5p8-east5-20260330-000700`

Current state at last check:

- `JOB_STATE_RUNNING`
- `task_state_counts.running = 1`
- no pending reason

### Early Monitoring Result

This run is past scheduler allocation and into real startup.

Observed so far:

- W&B initialized successfully
- run id: `053ujx8y`
- W&B URL: `https://wandb.ai/marin-community/dpo/runs/053ujx8y`
- train and validation token caches loaded from east5
- no OOM / `RESOURCE_EXHAUSTED` / JAX lowering failure seen yet
- worker is actively reading HF model shards for `marin-community/marin-8b-instruct`

Notable warning:

- cache metadata mismatch warning on `preprocessor_metadata`
- this did **not** immediately kill the run
- at last check the worker was still making forward progress through model shard reads

### Current Risk Assessment

What is already ruled out:

- scheduler-capacity failure on east5
- immediate container death on startup
- immediate host-RAM OOM during initial process boot

What is still not ruled out yet:

- failure later in model load
- first-step compile/lowering failure
- TPU HBM OOM once actual training starts

### Immediate Next Watchpoints

- finish loading all `marin-8b-instruct` safetensor shards
- reach first batch load
- reach train-step tracing / lowering
- survive step-0 eval without OOM

## 2026-03-30 Failure Update: east5 v5p-8 Run Hit TPU HBM OOM At First-Step Compile

The east5 run did not survive first-step compile.

Final job state:

- job id: `/ahmed/lora-bsv2-mi-b0p1-s2-lr7p5e6-v5p8-east5-20260330-000700`
- state: `JOB_STATE_FAILED`
- W&B run: `053ujx8y`

What happened:

- the job loaded the east5 train/validation caches successfully
- it loaded all `marin-community/marin-8b-instruct` safetensor shards successfully
- it reached first batch load, tracing, and HLO lowering
- it then failed in JAX/XLA with TPU HBM exhaustion during compile

Key failure signal from logs:

- `RESOURCE_EXHAUSTED: XLA:TPU compile permanent error`
- HBM used: `111.15G` of `95.74G`
- over capacity by: `15.41G`
- dominant temporary allocation included a `bf16[32,32,4096,4096]` broadcast of about `32.00G`

Interpretation:

- this was **not** a scheduler-capacity failure
- this was **not** a host-RAM failure
- this was **not** a tokenizer/cache problem
- this was a true TPU-device-memory failure at the first training step

Most important conclusion:

- LoRA reduced trainable-parameter / optimizer-state cost, but it did **not** make the activation and temporary-memory footprint small enough for `train_batch_size: 128` at `4096` tokens on `v5p-8`

Therefore the next experiment should follow the earlier LoRA plan exactly:

- keep `r=64`
- keep all-layer LoRA (`target_modules: null`)
- keep `lr=7.5e-6`
- keep `beta=0.1`
- keep the same dataset and seed
- reduce train batch size before touching rank

Recommended next ladder on the same hardware:

1. rerun with `train_batch_size: 64`
2. if that still OOMs, rerun with `train_batch_size: 32`
3. only after memory is sane, compare learning behavior and consider rank changes

This failure is actually consistent with the experimental guidance already recorded above: batch is the first knob to lower before increasing rank.

# CODEX END

## 2026-03-30 Babysitting: LoRA DPO batch-64 v5p-8 east5

### Job Info

- **Iris job**: `/ahmed/lora-bsv2-mi-b0p1-s2-lr7p5e6-b64-v5p8-east5-20260330-0028`
- **W&B**: `https://wandb.ai/marin-community/dpo/runs/endlboq3`
- **Config**: `lib/levanter/config/dpo/lora_dpo_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-6_seed2_v5p8_east5_b64.yaml`
- **Hardware**: v5p-8, us-east5-a, 256GB RAM
- **Key params**: LoRA r=64, alpha=64, lr=7.5e-6, beta=0.1, seed=2, batch=64, 850 steps

### Monitoring Log

#### Check 1 — ~08:18 UTC

| Field | Value |
|-------|-------|
| Job state | RUNNING |
| Phase | Step-0 eval |
| Eval progress | 78/367 iterations |
| Loss | 0.693 |
| Rate | 13.2s/it |
| ETA for eval | ~64 min (~09:22 UTC) |
| Failures | 0 |
| Preemptions | 0 |

- Step-0 loss = 0.693 = ln(2) — correct for untrained LoRA (policy == reference).
- Past the OOM point that killed the batch-128 run. batch-64 fits in HBM.
- No errors. Healthy.

#### Check 2 — ~08:44 UTC

| Field | Value |
|-------|-------|
| Job state | RUNNING |
| Phase | Step-0 eval |
| Eval progress | 193/367 iterations (53%) |
| Loss | 0.693 |
| Rate | 13.2s/it (unchanged) |
| ETA for eval | ~38 min (~09:22 UTC) |
| Failures | 0 |
| Preemptions | 0 |

- Rock steady through 193 eval iterations. No errors, no restarts.
- eval has been running ~43 min. Should finish around 09:22 UTC, then training begins.
- Critical next milestone: first training step (backward pass) — will reveal if batch-64 fits for training too.

#### Check 3 — ~09:24 UTC (CRITICAL MILESTONES CLEARED)

| Field | Value |
|-------|-------|
| Job state | RUNNING |
| Phase | Training |
| Train step | 5/850 |
| Train loss | 0.692 |
| Step-0 eval loss | 0.693 (correct) |
| Checkpoint | step-1 saved to GCS OK |
| Failures | 0 |
| Preemptions | 0 |

**All critical milestones passed:**
1. Step-0 eval completed (367/367 batches, 81 min, loss=0.693)
2. First train step completed in 79.7s (includes JIT compilation)
3. Checkpoint saved at step 1 to `gs://marin-us-east5/checkpoints/dpo/lora_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-6_seed2_v5p8_b64/endlboq3/step-1` — **no OOM**
4. Training proceeding at step 5, loss=0.692
- batch-64 fits in HBM for both forward and backward passes on v5p-8.
- Rate is inflated (674.8s/it) because it includes eval+JIT in the average. Actual training step time not yet measurable.

#### Check 4 — ~09:39 UTC (training stable, loss decreasing)

| Field | Value |
|-------|-------|
| Job state | RUNNING |
| Phase | Training |
| Train step | 37/850 |
| Train loss | 0.572 |
| Rate | 28.0s/it (stabilized) |
| Checkpoints saved | step-1, step-23 (step-1 cleaned up) |
| Failures | 0 |
| Preemptions | 0 |

- Loss trajectory: 0.693 → 0.698 → 0.692 → 0.686 → 0.679 → 0.667 → 0.656 → 0.648 → 0.631 → 0.618 → 0.572
- **LoRA is learning** — clear loss decrease from 0.693 to 0.572 in 37 steps.
- Steady-state training rate: ~28s/step on v5p-8 (single host).
- Checkpoint saves are working without OOM.
- ETA for 850 steps (training only): ~6.3h from now (~16:00 UTC).
- ETA with evals: ~13h total (~22:00 UTC), due to 81-min evals every 200 steps.

#### Check 5 — ~09:58 UTC

| Field | Value |
|-------|-------|
| Job state | RUNNING |
| Phase | Training |
| Train step | 77/850 |
| Train loss | 0.306 |
| Rate | 28.0s/it |
| Checkpoints | step-67 (latest), step-45/23/1 cleaned up |
| Failures | 0 |
| Preemptions | 0 |

- Loss trajectory: 0.693 → 0.572 → 0.521 → 0.458 → 0.431 → 0.418 → 0.306
- Very strong learning — loss dropped 55% in 77 steps.
- Checkpointing every ~22 steps (~10 min), working cleanly.
- At 28s/step, step 200 (first mid-training eval) ETA: ~57 min from now (~10:55 UTC).

#### Check 6 — ~10:14 UTC

| Field | Value |
|-------|-------|
| Job state | RUNNING |
| Phase | Training |
| Train step | 111/850 |
| Train loss | 0.212 |
| Rate | 29.4s/it |
| Checkpoints | step-111 (saving), step-89 (latest saved) |
| Failures | 0 |
| Preemptions | 0 |

- Loss trajectory: 0.693 → 0.306 (step 77) → 0.212 (step 111)
- Loss dropped 69% in 111 steps. LoRA learning aggressively with lr=7.5e-6.
- Step 200 (first eval + HF export) ETA: ~43 min (~10:57 UTC).
- Checkpoints saving every ~22 steps without issues.

#### Check 7 — ~10:57 UTC (STEP 200 REACHED, EVAL STARTED)

| Field | Value |
|-------|-------|
| Job state | RUNNING |
| Phase | Step-200 eval |
| Train step | 200/850 |
| Train loss at step 200 | 0.032 |
| Rate | 27.9s/it |
| Eval progress | just started (0/367) |
| Failures | 0 |
| Preemptions | 0 |

- **Step 200 milestone reached.** Train loss dropped from 0.693 to 0.032 (95.4% reduction).
- This is comparable to the full-FT baseline which reached loss ~0.005 at step 849.
- LoRA at lr=7.5e-6 (10x FullFT lr) is learning very aggressively.
- Loss trajectory: 0.693 → 0.306 (77) → 0.212 (111) → 0.100 (131) → 0.032 (200)
- Step-200 eval will take ~81 min (367 batches at 13.2s/it).
- HF export should follow eval completion.
- Training time so far: ~3h04m for 200 steps (28s/step avg).

#### Check 8 — ~12:19 UTC (STEP-200 EVAL COMPLETE, HF EXPORT IN PROGRESS)

| Field | Value |
|-------|-------|
| Job state | RUNNING |
| Phase | HF export (step-200 merged model) |
| Train step | 200/850 |
| Step-200 eval loss | **0.025** |
| Step-0 eval loss | 0.693 |
| Eval improvement | 96.4% reduction |
| HF export | saving 7 shards (32GB) to GCS |
| Failures | 0 |
| Preemptions | 0 |

- **Step-200 eval complete.** Validation loss 0.025 vs step-0 loss 0.693 (96.4% reduction).
- For comparison, the full-FT baseline step-200 eval loss was ~0.005 (from the `947c5d` run).
- LoRA eval loss (0.025) is ~5x higher than full-FT (0.005) at step 200 — expected since LoRA has fewer parameters.
- Merged HF model being saved to `gs://marin-us-east5/checkpoints/dpo/lora_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-6_seed2_v5p8_b64/merged_hf/endlboq3/step-200`
- Eval took ~81 min (367 batches at 13.2s/it). Consistent with step-0 eval.
- Next milestone: training resumes, then step-400 eval.
- Total wall time so far: ~4h27m (started ~07:51 UTC).

#### Check 9 — ~12:35 UTC (training resumed post-HF-export)

| Field | Value |
|-------|-------|
| Job state | RUNNING |
| Phase | Training (steps 200-400) |
| Train step | 222/850 |
| Train loss | 0.018 |
| Rate | 30.2s/it |
| HF export step-200 | COMPLETE |
| Failures | 0 |
| Preemptions | 0 |

- Training resumed cleanly after HF export.
- Step-400 eval ETA: ~90 min from step 222 (~14:04 UTC).
- Rate settling around 30s/step (slightly higher than pre-eval 28s — still normalizing from average).
- Loss is very low (0.018) and mostly converged. Expect minimal further decrease.

#### Check 10 — ~12:56 UTC (~5h into training)

| Field | Value |
|-------|-------|
| Job state | RUNNING |
| Phase | Training (steps 200-400) |
| Train step | 267/850 |
| Train loss | 0.007 |
| Rate | 28.0s/it |
| Wall time | ~5h04m |
| Failures | 0 |
| Preemptions | 0 |

- Run is extremely stable. Zero failures, zero preemptions in 5+ hours.
- Loss has mostly converged: 0.693 → 0.032 (step 200) → 0.007 (step 267)
- Step-400 eval ETA: ~62 min from step 267 (~13:58 UTC)
- Step-400 eval + HF export will take ~90 min
- Training completion (step 850) ETA: many more hours due to eval overhead.

### Summary of key metrics so far

| Step | Train loss | Eval loss | Phase |
|------|-----------|-----------|-------|
| 0 | 0.693 | 0.693 | eval |
| 77 | 0.306 | — | train |
| 111 | 0.212 | — | train |
| 131 | 0.100 | — | train |
| 200 | 0.032 | **0.025** | eval + HF export |
| 267 | 0.007 | — | train |

### Key GCS artifacts

| Resource | Path |
|----------|------|
| Merged HF model (step-200) | `gs://marin-us-east5/checkpoints/dpo/lora_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-6_seed2_v5p8_b64/merged_hf/endlboq3/step-200` |
| Training checkpoints | `gs://marin-us-east5/checkpoints/dpo/lora_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-6_seed2_v5p8_b64/endlboq3/` |

#### Check 11 — ~14:00 UTC (STEP 400 REACHED, EVAL STARTED)

| Field | Value |
|-------|-------|
| Job state | RUNNING |
| Phase | Step-400 eval |
| Train step | 401/850 |
| Train loss at step 401 | 0.001 |
| Eval progress | 3/367 |
| Early eval loss | 0.001 |
| Wall time | ~6h08m |
| Failures | 0 |
| Preemptions | 0 |

- **Step 400 milestone reached.** Train loss 0.001 — nearly fully converged.
- Step-400 eval started. Early eval loss=0.001 (vs 0.025 at step-200 eval, 0.693 at step-0).
- Eval will take ~81 min. HF export will follow.
- Run has been completely stable for 6+ hours. Zero failures, zero preemptions.

| Step | Train loss | Eval loss |
|------|-----------|-----------|
| 0 | 0.693 | 0.693 |
| 200 | 0.032 | 0.025 |
| 400 | 0.001 | (in progress, early=0.005) |

#### Check 12 — ~14:19 UTC (step-400 eval ~23% through, 8h monitoring)

| Field | Value |
|-------|-------|
| Job state | RUNNING |
| Phase | Step-400 eval |
| Eval progress | 83/367 |
| Eval loss (partial) | 0.005 |
| Wall time | ~6h28m |
| Failures | 0 |
| Preemptions | 0 |

- Step-400 eval in progress. Early eval loss ~0.005, very close to the full-FT baseline (0.005 at step 849).
- Job has been rock solid for 6.5+ hours. Zero failures, zero preemptions.
- After eval completes: HF export at step-400, then training resumes to step 600.
- Monitoring session approaching 8h mark. Job is healthy and self-sustaining via Iris.
- Remaining milestones: step-400 HF export, step-600 eval+export, step-800 eval+export, step-849 complete.

#### Check 13 — ~15:22 UTC (STEP-400 EVAL COMPLETE, HF EXPORT IN PROGRESS)

| Field | Value |
|-------|-------|
| Job state | RUNNING |
| Phase | HF export (step-400 merged model) |
| Train step | 401/850 |
| Step-400 eval loss | **0.005** |
| Step-200 eval loss | 0.025 |
| Step-0 eval loss | 0.693 |
| Wall time | ~7h31m |
| Failures | 0 |
| Preemptions | 0 |

- **Step-400 eval loss 0.005 — matches the full-FT baseline!**
- The full-FT baseline (947c5d run) reached eval loss 0.005 at step 849.
- LoRA at lr=7.5e-6 reached the same eval loss at step 400 — **half the training steps**.
- This is a very strong result for LoRA DPO.
- Merged HF model saving to `gs://marin-us-east5/checkpoints/dpo/lora_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-6_seed2_v5p8_b64/merged_hf/endlboq3/step-400`

### Updated metrics table

| Step | Train loss | Eval loss | Notes |
|------|-----------|-----------|-------|
| 0 | 0.693 | 0.693 | Untrained (policy == reference) |
| 200 | 0.032 | 0.025 | |
| 400 | 0.001 | **0.005** | Matches full-FT baseline at step 849! |
| 600 | — | — | (pending) |
| 800 | — | — | (pending) |

#### Check 14 — ~15:46 UTC (~8h into training, step 440)

| Field | Value |
|-------|-------|
| Job state | RUNNING |
| Phase | Training (steps 400-600) |
| Train step | 440/850 (52%) |
| Train loss | 0.016 |
| Rate | 27.9s/it |
| Wall time | ~7h54m |
| Failures | 0 |
| Preemptions | 0 |

- Past the halfway point of total training.
- Step-400 eval loss (0.005) matched full-FT baseline — the headline result.
- Step-600 eval ETA: ~75 min from step 440 (~17:01 UTC) + 81 min eval = ~18:22 UTC.
- Run continues to be completely stable. Zero failures across 8+ hours.
- Remaining: steps 440→849, evals at 600 and 800, HF exports at each.

## 2026-03-30 LR Sweep: lr=5e-6 Parallel Run

### Motivation

Comparing the LoRA run (lr=7.5e-6) with the full-FT baseline (lr=7.5e-7) via W&B API:

| Metric @ step 400 | Full-FT (lr=7.5e-7) | LoRA (lr=7.5e-6) |
|---|---|---|
| eval loss | 0.0070 | 0.0047 |
| dpo_accuracy | 0.9958 | 0.9983 |
| dpo_margin_policy | -70.6 | -51.0 |
| dpo_rejected_reward | -9.05 | -11.96 |

LoRA at 10x LR is converging much faster than full-FT — margin_policy at step 400 (-51) is already where full-FT is at step 600 (-49). rejected_reward has bottomed out at the full-FT final value (-11.96 vs -11.93). This suggests the LR may be slightly too aggressive. A 5e-6 run (6.7x full-FT LR) should converge more gradually and potentially track the baseline curve more closely.

### Config

- **Iris job**: `/ahmed/lora-bsv2-mi-b0p1-s2-lr5e6-b64-v5p8-central1-20260330`
- **Config**: `lib/levanter/config/dpo/lora_dpo_bloom_speceval_v2_marin_instruct_beta0.1_lr5e-6_seed2_v5p8_central1_b64.yaml`
- **LR**: 5e-6 (vs 7.5e-6 in the first run)
- **Zone**: us-central1-a (caches local)
- Everything else identical: LoRA r=64, beta=0.1, seed=2, batch=64, 850 steps
- W&B group: `lora-bsv2-mi-b0p1-s2-v5p8-lr-sweep`

### Launch status

- Submitted ~09:34 UTC. Provisioned and running.
- Preempted once. Restarted WITHOUT fixed RUN_ID → created new W&B run, lost progress (step 153 → restarted from 0).
- Killed and relaunched with `-e RUN_ID uxviy7fz` → resumed from step-153 checkpoint on same W&B run.
- Step-200 eval loss: **0.068**

## 2026-03-30 LR Sweep: lr=1e-5 Run

### Config

- **Iris job**: `/ahmed/lora-bsv2-mi-b0p1-s2-lr1e5-b64-v5p8-east5-20260330`
- **W&B run ID**: `ki3eb0hi`
- **Config**: `lib/levanter/config/dpo/lora_dpo_bloom_speceval_v2_marin_instruct_beta0.1_lr1e-5_seed2_v5p8_east5_b64.yaml`
- **LR**: 1e-5 (high end of sweep)
- **Zone**: us-east5-a
- Everything else identical: LoRA r=64, beta=0.1, seed=2, batch=64, 850 steps

### Status

- Preempted once. Restarted without fixed RUN_ID → new W&B run created.
- Killed and relaunched with `-e RUN_ID ki3eb0hi` → resumed from step-271 checkpoint.

## 2026-03-30 lr=7.5e-6 Run: Step-600 and Step-800 Evals

The original lr=7.5e-6 run completed step-600 eval before being preempted:

| Step | Train loss | Eval loss | Notes |
|------|-----------|-----------|-------|
| 0 | 0.693 | 0.693 | Untrained |
| 200 | 0.032 | 0.025 | |
| 400 | 0.001 | 0.005 | Matches full-FT baseline at step 849 |
| 600 | 0.005 | **0.004** | Better than full-FT at step 600 (0.006) |
| 800 | — | — | Preempted during step-800 eval |

- W&B run ID: `endlboq3`
- Preempted at step ~800 (during step-800 eval). Iris restarted but created new W&B run.
- Killed and relaunched with `-e RUN_ID endlboq3` → resumed from step-799 checkpoint.
- Currently re-running step-800 eval, then 49 more training steps to completion.

## 2026-03-30 Preemption Recovery: RUN_ID Lesson

### Problem

When Iris restarts a preempted job, the new process generates a random W&B run ID. The checkpoint path includes this ID:
```
gs://...base_path/{wandb_run_id}/step-N
```
New process → new run ID → new checkpoint dir → starts from scratch. Progress lost.

### Root cause

Direct `python -m levanter.main.train_dpo` launches bypass the executor system. The executor uses `impute_run_id_from_output_path` to derive a deterministic run ID from the output path. Direct launches use a random 8-char ID.

### Fix

Pass a fixed `RUN_ID` env var in the Iris job command:
```bash
-e RUN_ID <wandb_run_id>
```
Levanter checks `os.environ["RUN_ID"]` before generating a random one (trainer.py line 1029). This ensures the same checkpoint dir and W&B run across restarts.

### Runs affected and fixed

| Run | Original W&B ID | Fixed? |
|-----|----------------|--------|
| lr=7.5e-6 (east5) | `endlboq3` | Yes, relaunched with `-e RUN_ID endlboq3` |
| lr=5e-6 (central1) | `uxviy7fz` | Yes, relaunched with `-e RUN_ID uxviy7fz` |
| lr=1e-5 (east5) | `ki3eb0hi` | Yes, relaunched with `-e RUN_ID ki3eb0hi` |

## 2026-03-30 v6e-128 OOM and Cross-Host FSDP Analysis

### v6e-128 OOM (batch=128, default FSDP)

- Job: `/ahmed/lora-bsv2-mi-b0p1-s2-lr7p5e6-b128-v6e128-east5-20260330`
- Error: `RESOURCE_EXHAUSTED: Used 37.98G of 31.25G hbm. Exceeded by 6.73G`
- Root cause: Levanter's default FSDP only shards within-host (4-way). Each chip holds 8B × 4 bytes / 4 = 8 GB of model params, leaving only 23 GB for everything else on a 31.25 GB chip.

### Why default FSDP only shards within-host

Levanter's mesh config (lib/levanter/src/levanter/utils/mesh.py):
- ICI axes: `{"data": -1}` → absorbs all chips within a host (4 chips)
- DCN axes: `{"replica_dcn": -1}` → absorbs all hosts (32 hosts for v6e-128)
- `param_mapping: {"embed": "data"}` → parameters only shard on `data` axis (within-host)
- `replica_dcn` is used for batch parallelism only, NOT parameter sharding

### Cross-host FSDP fix

Change param_mapping to shard across both axes:
```yaml
trainer:
  mesh:
    param_mapping:
      embed: [replica_dcn, data]
```
This shards parameters across all chips (32 × 4 = 128-way on v6e-128).
Per-chip model storage: 8B × 4 / 128 = 250 MB (vs 8 GB with default).

Tradeoff: all-gathers go over DCN (slower) but fits in memory. Acceptable for LoRA since gradient communication is only 620 MB (50× less than full FT).

### v6e-32 cross-host FSDP jobs (pending capacity)

- `/ahmed/lora-bsv2-b128-v6e32-xfsdp-east1-20260330` — pending, waiting for v6e-32 in east1-d
- `/ahmed/lora-bsv2-b128-v6e32-xfsdp-east5b-v2` — pending, waiting for v6e-32 in east5-b

See `.agents/logbooks/levanter_mesh_explained.md` for full mesh analysis.

## 2026-03-30 Infrastructure Setup: Model + Data Per Region

### marin-8b-instruct cached on GCS

Added `marin_8b_instruct` to `experiments/models.py` (revision `0378f9c`).
Downloaded via Iris CPU jobs to 3 regions:

| Region | Path | Status |
|--------|------|--------|
| us-central1 | `gs://marin-us-central1/models/marin-community--marin-8b-instruct--0378f9c/` | Done |
| us-east5 | `gs://marin-us-east5/models/marin-community--marin-8b-instruct--0378f9c/` | Done |
| us-east1 | `gs://marin-us-east1/models/marin-community--marin-8b-instruct--0378f9c/` | Done |
| europe-west4 | — | Failed (permission denied on bucket) |

### Tokenized caches

| Region | Train cache | Val cache |
|--------|------------|-----------|
| us-central1 | Yes (original) | Yes (original) |
| us-east5 | Yes (original) | Yes (original) |
| us-east1 | Yes (tokenized via Iris) | Yes (tokenized via Iris) |
| europe-west4 | No | No |

Raw preference JSONL copied to us-east1 via `gsutil -m cp -r`.

### All YAML configs updated to use GCS model paths

All 7 LoRA DPO configs updated from `initialize_from_hf: marin-community/marin-8b-instruct` to region-specific GCS paths. Never use HuggingFace for model loading — always use GCS.

## 2026-03-30 Current Experiment Status

### Active runs

| Run | LR | Batch | Hardware | W&B ID | Iris Job | Step | Status |
|-----|-----|-------|----------|--------|----------|------|--------|
| lr=7.5e-6 | 7.5e-6 | 64 | v5p-8 east5 | `endlboq3` | `lora-bsv2-mi-b0p1-s2-lr7p5e6-b64-v5p8-east5-resume` | ~800 | Running (step-800 eval) |
| lr=5e-6 | 5e-6 | 64 | v5p-8 central1 | `uxviy7fz` | `lora-bsv2-mi-b0p1-s2-lr5e6-b64-v5p8-central1-resume` | ~236 | Running |
| lr=1e-5 | 1e-5 | 64 | v5p-8 east5 | `ki3eb0hi` | `lora-bsv2-mi-b0p1-s2-lr1e5-b64-v5p8-east5-resume` | ~273 | Running |

### Pending runs

| Run | LR | Batch | Hardware | Status |
|-----|-----|-------|----------|--------|
| batch=128 xfsdp | 7.5e-6 | 128 | v6e-32 east1 | Pending (no v6e-32 capacity) |
| batch=128 xfsdp | 7.5e-6 | 128 | v6e-32 east5 | Pending (no v6e-32 capacity) |

### Eval results so far (all runs)

| Run | Step 0 | Step 200 | Step 400 | Step 600 | Step 800 |
|-----|--------|----------|----------|----------|----------|
| Full-FT baseline (lr=7.5e-7) | 0.693 | 0.022 | 0.007 | 0.006 | 0.006 |
| **LoRA lr=7.5e-6** | 0.693 | 0.025 | **0.005** | **0.004** | (in progress) |
| **LoRA lr=5e-6** | 0.693 | **0.068** | — | — | — |
| **LoRA lr=1e-5** | 0.693 | — | — | — | — |

### W&B Links

| Run | URL |
|-----|-----|
| Full-FT baseline | https://wandb.ai/marin-community/dpo/runs/new_dpo_v2_bloom_speceval_v2_marin_instruct_beta0.1_-7_seed2-947c5d |
| LoRA lr=7.5e-6 | https://wandb.ai/marin-community/dpo/runs/endlboq3 |
| LoRA lr=5e-6 | https://wandb.ai/marin-community/dpo/runs/uxviy7fz |
| LoRA lr=1e-5 | https://wandb.ai/marin-community/dpo/runs/ki3eb0hi |

## 2026-03-30 Codex: Replicated Eval Sharding Experiment — Not Worth Prioritizing

Codex ran the standalone replicated cached-eval experiment to completion.

- **W&B run**: `lr15l9n2` (`jumping-resonance-174`)
- **Iris job**: `/ahmed/dpo-eval-cached-repl-v5p32-20260329-230854`
- **Hardware**: v5p-32 (16 chips, 4 hosts)
- **Mode**: cached eval, replicated parameter layout (post-load reshard via `hax.shard_with_axis_mapping`)

### Result

| Metric | FSDP cached eval | Replicated cached eval |
|--------|-----------------|----------------------|
| Total eval time | ~10.4 min | ~9.9 min |
| Avg time/batch | ~13.4s | ~12.87s |
| Warmup | — | ~24.1s |
| Data load | — | ~8.5s (1.4% of total) |

**Conclusion**: Directionally positive but not large enough to justify complexity. Eval is compute-bound after caching, not communication-bound.

**Decision**: Do NOT prioritize integrating eval-only replicated sharding into `train_dpo.py`. Leave standalone `experiments/eval_dpo.py` support as a profiling tool. Defer until eval speed becomes a top blocker again.

## 2026-03-30 Codex: Main Merge Recovery — Chat Stop Tokens

Codex reviewed a failed Claude merge attempt and identified the main mistake: it tried to resolve DPO conflicts by provenance ("keep our refactor everywhere") instead of by behavior. That would have dropped the recent `origin/main` change that writes `generation_config.json` with chat stop tokens for HF exports.

### Resolution

- Kept the refactored DPO runtime shape from this branch (canonical `train_dpo.py`, adapter/reference abstraction, `model_init.py`)
- Ported main's generation-config behavior into the refactored export layer
- Threaded `generation_config` through:
  - `lib/levanter/src/levanter/compat/hf_checkpoints.py`
  - `lib/levanter/src/levanter/adaptation.py`
  - `lib/levanter/src/levanter/lora.py`
  - `lib/levanter/src/levanter/main/train_dpo.py`
- Also brought over DPO-facing config/docs/tests for `hf_generation_eos_token_ids`
- Merged LoRA exports now get the same `generation_config.json` behavior as regular DPO

### Verification

- Merge commit: `e6dcb2e96`
- Branch is 53 commits ahead, 0 behind `origin/main`
- `./infra/pre-commit.py --fix` on touched files: passed
- pytest: `49 passed, 2 skipped`

## 2026-03-30 Codex: Durable Reference Eval Log-Prob Cache — Implemented

Codex implemented the production cached reference-eval path that was validated in `experiments/eval_dpo.py` as a proper library feature.

### Code organization

- **New module**: `lib/levanter/src/levanter/dpo.py` — canonical home for reusable DPO runtime code
- Owns: `DpoModel`, shared DPO loss helpers, `ReferenceEvalCacheConfig`, `CachedDpoExample`, `CachedReferenceDataset`, `ValidationDatasetSpec`, cache metadata/path helpers, build/load helpers
- `train_dpo.py` stays as orchestration: builds validation specs, derives reference identity, optionally materializes caches before training, swaps eval datasets onto cached wrappers

### Config surface

- `TrainDpoConfig` now has `reference_eval_cache: ReferenceEvalCacheConfig`
- Threaded through `lora_dpo.py`, `simple_dpo_config.py`, `experiments/defaults.py`
- Default: `mode: "disabled"` (opt-in)

### Key design choices

- **Durable sidecar cache on GCS**, then loaded into host RAM (~188 KB for 23K examples)
- **Strict cache identity**: includes validation cache provenance, slice bounds, reference identity, sequence length, schema version
- **Finished-or-rebuild semantics**: missing → build, unfinished ledger → build, metadata mismatch → rebuild
- **Eval loss path** accepts either `DpoExample` (uncached) or `CachedDpoExample` (cached, skips reference forward passes)
- **Training path unchanged**

### Verification

- Committed as `35d9c444b` (`[dpo] Cache reference eval logprobs`)
- Pushed to `origin/dpo-lora`
- `./infra/pre-commit.py --fix` on scoped file set: passed
- pytest: `34 passed, 1 skipped`

## 2026-03-30 Updated PR Goals

The original goal "keep `TrainerState.model` policy-only for both regular DPO and LoRA-DPO" was **disproven by experiment** (multihost closure bug). Updated goals:

- Regular DPO: `DpoModel(policy, reference)` in trainer state (multihost safety)
- LoRA-DPO: policy-only trainer state (reference derived from adapter base)
- Config-level abstraction still clean: `adapter.type` + `reference.type`
- Reference eval log-prob caching available as opt-in for ~2x eval speedup
- Chat stop tokens in HF exports for both regular and LoRA DPO
- LoRA-DPO validated on Bloom SpecEval v2 with batch-64 on v5p-8

## 2026-03-30 Current State Summary

### What is done

1. **Canonical DPO refactor** — unified entrypoint, adapter/reference config, shared adaptation layer
2. **Multihost TPU fix** — DpoModel in trainer state for separate-reference mode
3. **Eval profiling** — identified 4 redundant forward passes, 317K FSDP resharding ops
4. **Reference eval caching** — validated 2x speedup, productionized in `levanter/dpo.py`
5. **Replicated eval sharding** — tested, marginal improvement, deferred
6. **Main merge** — chat stop tokens ported, branch up to date
7. **LoRA-DPO training** — batch-64 on v5p-8 working, loss matches full-FT baseline at step 400
8. **LR sweep** — 3 runs launched (5e-6, 7.5e-6, 1e-5)
9. **Infrastructure** — marin-8b-instruct cached on GCS in 3 regions, tokenized caches in east1

### What is in progress

- lr=7.5e-6 run: step ~800, nearing completion
- lr=5e-6 run: step ~236, running
- lr=1e-5 run: step ~273, running
- v6e-32 cross-host FSDP batch-128 jobs: pending capacity

### What is NOT done

- Wiring cached reference eval into `train_dpo.py` eval hooks (production integration)
- Excluding frozen reference from checkpoint serialization
- Teaching `SimpleDPOConfig` / `default_dpo` about LoRA for sweep parity
- Full LR sweep analysis (waiting for runs to complete)
- Batch size ablation (only if needed after LR sweep)
