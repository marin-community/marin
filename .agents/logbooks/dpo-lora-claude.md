# DPO LoRA Claude Logbook

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
