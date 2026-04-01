# DPO LoRA Codex Logbook

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

- Refactored [`lib/levanter/src/levanter/main/train_dpo.py`](/Users/ahmed/.codex/worktrees/6b74/marin/lib/levanter/src/levanter/main/train_dpo.py) to be the single canonical DPO runtime.
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

- Added [`lib/levanter/src/levanter/adaptation.py`](/Users/ahmed/.codex/worktrees/6b74/marin/lib/levanter/src/levanter/adaptation.py).
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
- Reused low-level LoRA operations from [`lib/levanter/src/levanter/lora.py`](/Users/ahmed/.codex/worktrees/6b74/marin/lib/levanter/src/levanter/lora.py) rather than using legacy `lora_lm.py` as the architectural template.

### 3. Shared model/bootstrap helper

- Added [`lib/levanter/src/levanter/main/model_init.py`](/Users/ahmed/.codex/worktrees/6b74/marin/lib/levanter/src/levanter/main/model_init.py).
- Factored out shared model loading/bootstrap logic for:
  - HF converter setup
  - tokenizer replacement
  - optional HF-config adoption
  - HF checkpoint load
  - Levanter checkpoint load
  - parameter casting/sharding

### 4. Legacy LoRA-DPO compatibility shim

- Rewrote [`lib/levanter/src/levanter/main/lora_dpo.py`](/Users/ahmed/.codex/worktrees/6b74/marin/lib/levanter/src/levanter/main/lora_dpo.py) into a translation wrapper.
- Kept the legacy `LoraDpoConfig` surface for old config files.
- Translates legacy LoRA-DPO configs into canonical `TrainDpoConfig` with:
  - `adapter=LoraAdaptationConfig(...)`
  - `reference=AdapterBaseReferenceConfig()`
- Forwards execution into canonical `train_dpo.main`.

### 5. Experiment/config updates

- Updated [`experiments/defaults.py`](/Users/ahmed/.codex/worktrees/6b74/marin/experiments/defaults.py) so `default_dpo(...)` constructs canonical `TrainDpoConfig(reference=SeparateReferenceConfig(...))`.
- Updated canonical DPO YAML configs under [`lib/levanter/config/`](/Users/ahmed/.codex/worktrees/6b74/marin/lib/levanter/config) to use nested `adapter` / `reference` blocks instead of top-level `reference_model_path` / `reference_is_hf`.
- Left legacy `lib/levanter/config/dpo/lora_dpo_*` YAMLs on the compatibility path intentionally, so old LoRA-DPO configs still route through the shim.

### 6. Tests

- Updated [`lib/levanter/tests/test_dpo.py`](/Users/ahmed/.codex/worktrees/6b74/marin/lib/levanter/tests/test_dpo.py) to match the new architecture.
- Removed old tests that assumed `DpoModel` lived in trainer state.
- Added tests for:
  - policy-only `TrainerState`
  - invalid adapter/reference combinations
  - canonical config parsing for `adapter.type: none`
  - canonical config parsing for `adapter.type: lora`
  - legacy `LoraDpoConfig` translation
- Kept existing [`lib/levanter/tests/test_lora_dpo.py`](/Users/ahmed/.codex/worktrees/6b74/marin/lib/levanter/tests/test_lora_dpo.py) passing against the refactor.
- Replaced brittle parse-from-repo-config tests with minimal temp YAML fixtures after an initial failure exposed unrelated data-config parsing fields.

### 7. Docs

- Updated [`lib/levanter/docs/guides/DPO-Training.md`](/Users/ahmed/.codex/worktrees/6b74/marin/lib/levanter/docs/guides/DPO-Training.md) to describe:
  - canonical `train_dpo.py`
  - nested `adapter` / `reference` config
  - policy-only trainer state
  - separate-reference vs adapter-base reference behavior
- Updated [`lib/levanter/docs/guides/LoRA-DPO-Training.md`](/Users/ahmed/.codex/worktrees/6b74/marin/lib/levanter/docs/guides/LoRA-DPO-Training.md) to describe:
  - canonical `train_dpo.py` usage
  - `adapter.type: lora`
  - `reference.type: adapter_base`
  - explicit `zero_init_b: true` requirement
  - legacy `lora_dpo.py` status as compatibility-only

## Follow-up Review Changes

After the initial refactor, a follow-up review requested two concrete changes.

### Logger style

- Changed the new `logger.info(...)` calls in [`lib/levanter/src/levanter/main/train_dpo.py`](/Users/ahmed/.codex/worktrees/6b74/marin/lib/levanter/src/levanter/main/train_dpo.py) back to f-strings.

### RNG lineage preservation for regular DPO

- Restored the legacy full-DPO top-level split shape in canonical `train_dpo.py`:
  - `data_key, adapter_key, model_key, training_key = split(PRNGKey(seed), 4)`
  - this intentionally repurposes the old unused loader-key slot as `adapter_key`
- Added `_derive_training_keys(seed)` to preserve the old regular-DPO policy key lineage:
  - `policy_key = split(model_key)[0]`
- Used `model_key` as the separate-reference checkpoint shape key so non-HF separate references follow the old regular-DPO path more closely.
- Added a regression test in [`lib/levanter/tests/test_dpo.py`](/Users/ahmed/.codex/worktrees/6b74/marin/lib/levanter/tests/test_dpo.py) to verify that the derived `data_key`, `model_key`, `policy_key`, and `training_key` match the legacy full-DPO derivation.

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

- Restored `DpoModel(policy, reference)` in [`lib/levanter/src/levanter/main/train_dpo.py`](/Users/ahmed/.codex/worktrees/6b74/marin/lib/levanter/src/levanter/main/train_dpo.py) for `reference.type=separate`.
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
- Added a regression test in [`lib/levanter/tests/test_dpo.py`](/Users/ahmed/.codex/worktrees/6b74/marin/lib/levanter/tests/test_dpo.py) to assert that the separate reference is marked non-saveable / non-trainable in the mask.

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

## 2026-03-29 Codex Standalone Eval Profiling Thread

### Goal

- Get `experiments/eval_dpo.py` into a runnable state so DPO eval can be profiled directly without burning 400+ training steps just to trigger an eval hook.

### Initial Findings

- The checked-in `experiments/eval_dpo.py` did not match current Levanter APIs.
- It had broken model-init calls, broken cache loading, and an incomplete profiler / tracker lifecycle.
- It also diverged from the standard standalone eval pattern enough that even a superficially fixed script would not necessarily profile the real Levanter eval path.

### Fixes Applied

- Updated `experiments/eval_dpo.py` to use current model-init helpers:
  - `prepare_model_init_context(...)`
  - `load_model_from_source(...)`
- Switched validation-cache loading to the current `TreeCache.load(... exemplar=..., options=...)` pattern.
- Reused existing Bloom SpecEval v2 experiment config, tokenizer, and model definitions instead of keeping parallel hardcoded copies.
- Switched profiling lifecycle to `profile_ctx(...)`.
- Explicitly finish the tracker on exit for remote worker safety.

### Remote Failure Investigation

Observed remote failure during warmup:

- fused next-token loss crashed with
  - `Axis embed present in both specs with different sizes`
  - local shard looked like `embed(256)` while the model contract axis was `embed(4096)`

### Root Cause

- This was not a dataset-shape bug and not an HF-config mismatch.
- The standalone script had left `parameter_axis_mapping` active across the eval path.
- In this mesh config, `parameter_axis_mapping` includes `embed -> data`.
- That is correct for parameter layout, but wrong for executing the eval kernel.
- The fused CE path uses the current thread-local mapping when it enters `shard_map(...)`, so eval wound up executing under parameter sharding instead of compute sharding.

### Sharding Fix Applied

- Removed the outer `hax.axis_mapping(parameter_axis_mapping)` context from `experiments/eval_dpo.py`.
- Changed the eval kernel from plain `eqx.filter_jit` to:
  - `@hax.named_jit(axis_resources=compute_axis_mapping)`
- Kept model loading under `parameter_axis_mapping`.

Resulting model:

- parameters are still loaded/sharded the same way as training expects
- eval batches now execute under compute sharding, which should avoid the `embed(256)` vs `embed(4096)` fused-loss conflict

### Local Verification

- `uv run python -m py_compile experiments/eval_dpo.py` passed
- `uv run python - <<'PY' import experiments.eval_dpo as m; print(...) PY` passed

### Current Position

- `experiments/eval_dpo.py` should now be much closer to the real standalone eval shape used elsewhere in Levanter.
- The next material check is to relaunch the Iris job and confirm the warmup batches clear the previous fused-loss crash.

### Current Technical View

What looks high-confidence:

- Caching frozen-reference log-probs is the cleanest eval optimization.
- It removes 2 of the 4 forward passes in DPO eval without requiring sharding redesign.

What looks plausible but not yet proven:

- An eval-specific replicated/data-parallel model layout may reduce communication overhead.
- That is not a simple `axis_mapping` flip. It would require a second eval layout for the model itself, not just a different thread-local mapping.
- The right order is still:
  1. cache reference log-probs
  2. re-profile
  3. only then decide whether an eval-specific layout is worth building

## 2026-03-29 Codex Reference Logprob Cache Plan

### Goal

- Remove the two frozen-reference forward passes from every DPO eval by precomputing:
  - `logp_ref_chosen`
  - `logp_ref_rejected`
- Keep the cache reusable across runs.
- Avoid introducing meaningful runtime I/O overhead during eval.

### Core Design

- Persist the cache to GCS once.
- Load the full cache into host RAM at job startup.
- Join cached values to eval examples by dataset index.

This is the right split because the cached payload is tiny but expensive to compute.

For the Bloom SpecEval v2 validation set:

- the cache only needs two scalar floats per example
- DPO already reduces over position via `_logp_sum(...)`
- `PreferenceChatLmDatasetFormat` defaults to `pack=False`, so the validation set is naturally aligned 1:1 with cache rows

That means the runtime working set is on the order of a few hundred KB, not GB.

### Cache Payload

Per validation example, store:

```python
{
    "logp_ref_chosen": np.float32(...),
    "logp_ref_rejected": np.float32(...),
}
```

Do not cache per-token reference log-probs for this optimization. They are unnecessary for DPO eval and would only complicate storage and shape handling.

### Storage Format

Use a small sidecar `TreeCache`, not an ad hoc JSON/NPY blob.

Reasons:

- it reuses the repo’s existing cache machinery
- it naturally lives on GCS
- it gives us a ledger and metadata validation
- it is easy to load back as an `AsyncDataset`

Implementation sketch:

```python
exemplar = {
    "logp_ref_chosen": np.zeros((), dtype=np.float32),
    "logp_ref_rejected": np.zeros((), dtype=np.float32),
}

with SerialCacheWriter(cache_dir, exemplar, metadata=CacheMetadata(...)) as writer:
    writer.write_batch(batch_dict)
```

### Cache Key / Metadata

The cache must be invalidated whenever the frozen reference semantics could change.

Minimum key inputs:

- validation cache path
- reference model path / HF ref
- immutable reference revision if available
- tokenizer id
- sequence length
- preference-format fields that affect tokenization/loss semantics:
  - `chosen_field`
  - `rejected_field`
  - `mask_user_turns`
  - `slice_strategy`

Suggested metadata shape:

```python
CacheMetadata(
    preprocessor_metadata={
        "kind": "dpo_reference_logprobs_v1",
        "val_cache": "...",
        "reference_model": "...",
        "reference_revision": "...",
        "tokenizer": "...",
        "seq_len": 4096,
        "mask_user_turns": True,
        "slice_strategy": "raise",
    }
)
```

### Cache Location

Use a sibling GCS path near the existing validation token cache, not a W&B artifact-only flow.

Preferred pattern:

```text
<validation-cache-root>/reference_logprobs/<stable-cache-key>/
```

For the current experiment, that means something under the same `marin-us-central1` storage hierarchy as:

- `gs://marin-us-central1/tokenized/bloom_speceval_v2_val_prefs_marin_tokenizer-a06ae8/validation`

This keeps the data close to the source cache and avoids inventing a second distribution mechanism for something that is fundamentally dataset-derived.

### Runtime Integration

Do not read the sidecar cache remotely batch-by-batch during eval.

Instead:

1. load the sidecar cache once at startup
2. materialize the two float arrays in CPU memory
3. map validation examples to include cached reference values by index
4. during eval, compute only:
   - `logp_pi_chosen`
   - `logp_pi_rejected`
5. combine with cached reference scalars to produce the normal DPO metrics

The index-based join is straightforward because `MappedAsyncDataset` receives the item index.

### Code Shape Plan

1. Add a small cache-builder helper, probably in a DPO-focused module rather than inside the one-off experiment script.
2. Add a lightweight cached-example type for eval-only use.
3. Add a helper that:
   - tries to load the reference-logprob sidecar cache
   - builds it if missing
   - returns host arrays or a mapped dataset carrying cached scalars
4. Teach DPO eval to use cached reference scalars when present.
5. Leave training loss unchanged for now. Only the eval hook path should use the cache in phase 1.

### Suggested Rollout

#### Phase 1

- Implement the cache only for evaluation.
- Keep the existing policy path unchanged.
- Use the one-off `experiments/eval_dpo.py` script to validate correctness and speedup.

#### Phase 2

- Wire cached reference values into the main DPO eval hook path in `train_dpo.py`.
- Re-profile a training run with eval enabled.

#### Phase 3

- Decide whether more aggressive eval-layout changes are still worth doing after the 2x reference-pass reduction.

### Validation Plan

Correctness checks:

- compare uncached vs cached eval loss/metrics on a bounded number of batches
- assert numerical agreement within a tight tolerance
- verify cache length exactly matches validation dataset length

Performance checks:

- re-run the standalone eval profile
- verify that:
  - `reference_chosen`
  - `reference_rejected`
  no longer dominate the trace
- compare total eval wall time before/after

### Non-Goals For The First Pass

- Do not cache per-token log-probs.
- Do not try to make the cache work for packed preference datasets yet.
- Do not combine this with eval-specific replicated model layout in the same patch.
- Do not redesign checkpointing in the same change.

### Expected Outcome

- DPO eval should drop from 4 forward passes per example to 2.
- The cache should be reusable across runs and cheap to load.
- After this lands, we will have a much cleaner answer to whether remaining eval time is still dominated by communication or by local model compute / HBM bandwidth.

## 2026-03-29/30 Transfer Note: Claude Work Summary And Revised Eval-Sharding Plan

### TL;DR

- Claude finished the reference log-prob caching path for standalone DPO eval and validated it on real full-validation runs.
- The cached path appears correct and gives a reproducible ~2x eval speedup by removing the two frozen-reference forward passes.
- Claude's follow-up idea of replicated non-FSDP eval sharding is plausible, but the proposed implementation seam was too optimistic.
- The better next experiment is: keep model loading unchanged, run cached eval only, explicitly reshard the loaded policy model once into an eval-only replicated layout, then measure steady-state batches and communication.

### What Claude Actually Completed

Claude's most important completed work was on standalone DPO evaluation, not the training-path eval hooks.

Implemented in `experiments/eval_dpo.py`:

- three eval modes:
  - `--mode uncached`
  - `--mode build`
  - `--mode cached`
- a deterministic reference-logprob sidecar cache path
- cache build logic for:
  - `logp_ref_chosen`
  - `logp_ref_rejected`
- a cached eval path that avoids loading or running the reference model in cached mode
- a cached-example wrapper compatible with Levanter/JAX pytree expectations

The core design was sound:

- build reference log-probs once for the full validation set
- persist them as a tiny sidecar cache
- join by dataset index at eval time
- reduce DPO eval from 4 forward passes to 2

### Claude's Validation Results

#### 1-batch validation

- Cached loss compute dropped from `26.9s` to `13.4s`.
- This established the expected 2x compute reduction.

#### 20-batch apples-to-apples validation

- Uncached total eval: `552.1s`
- Cached total eval: `279.1s`
- Measured speedup: `1.98x`

#### Full validation correctness test on a trained checkpoint

Policy:

- trained step-849 checkpoint from the new DPO pipeline

Reference:

- `marin-community/marin-8b-instruct`

Validation set:

- full `bloom_speceval_v2_val`
- `23,552` examples
- `46` batches at eval parallelism `32`

Reported metric agreement:

- `loss`: exact match
- `dpo_accuracy`: exact match
- `dpo_chosen_reward`: exact match
- `dpo_rejected_reward`: exact match
- `dpo_margin_policy`: exact match
- `dpo_margin_ref`: exact match

Reported performance at full scale:

- uncached total eval: about `20.7 min`
- cached total eval: about `10.4 min`
- measured speedup: `1.98x`

Projected per-run savings from Claude's numbers:

- roughly `41 min` to `48 min` saved per full training run depending on which baseline is used

### Claude's Important Debugging Fixes

These are worth preserving because they explain the multihost failure modes we already hit:

- use `process_allgather(..., tiled=True)` before converting sharded outputs to numpy
- only one host should write the sidecar cache
- gate cache writes with `jax.process_index() == 0`
- synchronize after cache writes so non-writers do not race ahead
- use an `eqx.Module` for cached examples instead of a plain dataclass so DataLoader/JIT can batch them
- treat `TreeCache.get_batch_sync(...)` as returning a list of dicts, not a dict
- remove trailing `(1,)` dimensions from cached scalar values before wrapping them with `hax.named(...)`
- run eval/loss under `compute_axis_mapping`, not under `parameter_axis_mapping`
- avoid losing results behind multihost barrier/profiler cleanup ordering

### What Claude Did Not Finish

The standalone eval path is much further along than the main DPO training path.

At the current workspace state:

- `experiments/eval_dpo.py` exists and compiles locally
- `experiments/dpo_bloom_speceval_v2_profile.py` exists and compiles locally
- `train_dpo.py` still computes reference log-probs online inside the loss
- cached-reference eval does not yet appear wired into the main `train_dpo.py` eval hook path
- the worktree is still dirty and includes untracked standalone eval/profiling files

Relevant local verification:

- `uv run python -m py_compile experiments/eval_dpo.py experiments/dpo_bloom_speceval_v2_profile.py` passed
- pytest was not available in this environment, so I did not complete a local test run

### Assessment Of Claude's Non-FSDP Eval-Sharding Idea

The high-level idea is plausible:

- load or place the eval-time model so weights are replicated
- keep batches sharded across the data axis
- remove the FSDP parameter-layout transition during eval

However, the logged implementation proposal was too optimistic in three ways.

#### 1. It changed the wrong seam

Claude proposed making the experiment by changing:

```python
load_model_from_source(..., parameter_axis_mapping=parameter_axis_mapping)
```

to:

```python
load_model_from_source(..., parameter_axis_mapping=compute_axis_mapping)
```

This is not just an eval-layout change. It changes the HF/GCS deserialization path too, because `load_model_from_source(...)` passes that mapping into the loader itself.

That means the experiment would conflate:

- load-time sharding behavior
- deserialization behavior
- runtime eval layout

If the goal is to test whether the FSDP parameter-to-compute layout transition is the bottleneck, that is the wrong first seam to touch.

#### 2. The load-cost reasoning was not accurate for this codepath

The current HF/GCS path builds a full state dict on each host before sharding it into the model. So the difference is not simply:

- FSDP load = each chip loads only its shard
- replicated load = each chip loads the full model

For the current Levanter HF path, both cases already involve heavy per-host state-dict loading before device placement. Replicated eval still may cost more HBM, but the loading story is not as favorable to the current FSDP path as the plan suggested.

#### 3. The speedup estimate outran the evidence

Claude had already recorded the key caveat earlier:

- we had collective op counts
- we did not yet have per-op durations
- the collectives might be overlapped with compute

So a forecast like another `1.5x` to `3x` on top of caching was speculative. The right framing is:

- communication-free eval might help
- we do not yet know how much of the remaining 13.4s/batch is actually recoverable

### Revised Better Plan For Eval Sharding

### Goal

Answer the narrow question first:

- after reference-logprob caching, how much steady-state eval time is still caused by the FSDP parameter-layout transition?

Do not combine this with training-hook integration, checkpoint serialization changes, or other cleanup work.

### Scope

Run this experiment only in cached eval mode.

Reasons:

- cached eval only needs the policy model
- that cuts replicated-model HBM requirements roughly in half versus uncached/build mode
- it isolates the layout question from the already-solved reference-forward-pass problem

### Safer Experiment Design

Keep model loading unchanged at first:

```python
policy_model = load_model_from_source(
    ...,
    parameter_axis_mapping=parameter_axis_mapping,
    ...
)
```

Then explicitly reshard once, after load, into an eval-only layout:

```python
policy_model = hax.shard_with_axis_mapping(policy_model, compute_axis_mapping)
```

or the equivalent `hax.shard(...)` form.

Why this is the better first test:

- it isolates runtime/layout effects from loader effects
- it preserves the currently-working HF/GCS load path
- it matches the actual question we care about: whether eval gets faster when the model already lives in the compute layout

### What To Measure

For both baselines:

- cached eval with normal FSDP-loaded model
- cached eval with one-time explicit reshard to replicated compute layout

collect these separately:

1. policy model load time
2. one-time reshard time
3. warmup / first-compile time
4. steady-state loss time over:
   - 20 batches first
   - then full 46-batch validation if promising
5. total eval wall time
6. peak HBM / OOM behavior

Do not treat "single end-to-end wall time" as the only metric. The compile and reshard one-time costs need to be split out.

### Success Criteria

Call the experiment successful only if all of the following hold:

- cached metrics still match the FSDP-cached baseline
- the trace shows parameter-layout collectives disappear or materially shrink
- steady-state loss time improves enough to matter after excluding one-time reshard/compile cost
- HBM remains stable at the chosen eval parallelism

If collectives disappear but steady-state batches barely improve, stop. That means the remaining bottleneck is local compute, memory bandwidth, or something else.

### Order Of Experiments

#### Phase A: Cached eval, current layout, good measurements

- freeze a clean baseline for cached mode at:
  - `per_device_eval_parallelism=32`
  - `20` batches
  - then full validation if needed
- record:
  - load time
  - warmup time
  - steady-state loss time
  - total eval time

#### Phase B: Cached eval, explicit post-load reshard

- keep the same model load path
- reshard the loaded policy model once into compute mapping
- repeat the exact same measurement protocol

#### Phase C: Only if Phase B is clearly positive

- try `per_device_eval_parallelism=64`
- treat this as a separate capacity/throughput experiment, not part of the original sharding question

If `64` fails or becomes unstable:

- keep `32`
- preserve the simpler replicated-eval result if it helps

### Instrumentation Requirements

The profile should distinguish:

- policy forward
- reference forward
- eval loop load time
- eval loop loss time
- one-time warmup/compile

This is already partly in place from the current local changes:

- named scopes in `train_dpo.py`
- timing metrics in `levanter.callbacks.eval_loss_loop`
- standalone profiling script for steady-state DPO eval

### Concrete Code Sketch

The experiment should look roughly like:

```python
policy_model = load_model_from_source(
    context=model_context,
    Vocab=Vocab,
    model_key=policy_key,
    parameter_axis_mapping=parameter_axis_mapping,
    compute_dtype=trainer_config.mp.compute_dtype,
    cast_to_param=trainer_config.mp.cast_to_param,
    hf_ref=model_name,
)

if args.eval_param_layout == "replicated":
    t_reshard = time.time()
    policy_model = hax.shard_with_axis_mapping(policy_model, compute_axis_mapping)
    logger.info("Resharded policy model for eval in %.1fs", time.time() - t_reshard)
```

with an explicit CLI switch, for example:

```text
--eval_param_layout fsdp
--eval_param_layout replicated
```

That keeps the experiment reversible and makes the comparison honest.

### Risks In The Revised Plan

- One-time reshard may still be non-trivial in cost.
- JIT recompilation is expected because input/output shardings change.
- HBM headroom at `per_device=64` is still speculative.
- If the loader or model carries arrays whose metadata assumes parameter layout in a subtle way, explicit post-load reshard may surface another shape/sharding bug.

These are acceptable risks because they are all directly tied to the hypothesis being tested.

### Recommendation

The next best step is not "switch `load_model_from_source(...)` to `compute_axis_mapping` and hope."

The next best step is:

1. keep the working loader path
2. run cached eval only
3. explicitly reshard the already-loaded policy model once
4. measure steady-state batches and communication before trying higher batch sizes

If that experiment shows a real gain, then we can decide whether it is worth teaching the loader or training eval hooks about an explicit eval-only replicated layout.

## 2026-03-30 Outcome: Standalone Replicated Cached Eval Worked, But Is Not Worth Prioritizing

We ran the standalone replicated cached-eval experiment to completion on Iris.

Identifiers:

- W&B run: `lr15l9n2`
- W&B display name: `jumping-resonance-174`
- Iris job: `/ahmed/dpo-eval-cached-repl-v5p32-20260329-230854`

Run shape:

- mode: `cached`
- eval parameter layout: `replicated`
- hardware: `v5p-32`
- devices: `16`
- hosts: `4`
- checkpoint: `step-849` from `new_dpo_v2_bloom_speceval_v2_marin_instruct_beta0.1_-7_seed2-947c5d`

Controller outcome:

- Iris job finished `JOB_STATE_SUCCEEDED`
- `4/4` tasks succeeded
- no preemptions
- no worker failures

Measured result:

- total eval time: about `592.2s` (`9.9 min`)
- warmup time: about `24.1s`
- batches run: `46`
- data load time: about `8.2s` to `8.8s`
- loss compute time: about `583.3s` to `583.9s`
- average time per batch: about `12.87s`

Metric summary from W&B:

- `eval_cached/loss`: `0.00551`
- `eval_cached/dpo_loss`: `0.00551`
- `eval_cached/dpo_accuracy`: `0.9958`
- `eval_cached/dpo_chosen_reward`: `2.5412`
- `eval_cached/dpo_rejected_reward`: `-11.9235`
- `eval_cached/dpo_margin_policy`: `-45.0787`
- `eval_cached/dpo_margin_ref`: `-189.7258`

Interpretation:

- the standalone replicated-layout path is real and stable on multihost TPU
- cached eval remains overwhelmingly compute-bound after caching
- data loading was only about `1.4%` to `1.5%` of total eval time
- the remaining time is dominated by loss compute, not by input loading

Most important practical conclusion:

- the end-to-end improvement versus the earlier cached full-validation result (`~10.4 min`) is only modest (`~9.9 min` here)
- that is directionally positive, but not large enough to justify more complexity right now

Decision:

- do **not** prioritize integrating eval-only replicated sharding into `train_dpo.py`
- do **not** spend more time teaching the loader or training eval hooks about replicated eval layout right now
- leave the standalone `experiments/eval_dpo.py` support in place as a profiling/benchmark tool
- defer this thread until eval speed becomes a top blocker again

In short: this experiment de-risked the idea, but it did not earn promotion into the mainline DPO path yet.

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

## 2026-03-30 Design Plan: Durable Reference Eval Log-Prob Cache

### Problem

- DPO eval during training still does four forward passes because `loss_function` in `lib/levanter/src/levanter/main/train_dpo.py:402` computes both policy and reference log-probs inline.
- The eval hook added in `lib/levanter/src/levanter/main/train_dpo.py:593` goes through the generic `Trainer.add_eval_hook(...)` path in `lib/levanter/src/levanter/trainer.py:606`, so eval currently reuses the uncached training loss.
- `experiments/eval_dpo.py:79` proved that caching reference log-probs gives about a 2x eval speedup, but that file is a standalone profiling script and the wrong long-term home for production DPO logic.
- Pre-emption is routine on these jobs, so an in-memory-only cache is not acceptable. The cache must survive restarts and be safe to rebuild after partial writes.

### Goals

- Add an opt-in DPO config flag that builds or loads a reference-eval cache before training starts.
- Persist the cache to GCS, then load it into host RAM for repeated eval use within the run.
- Keep the cache scoped to validation/eval only. Do not change the training step, optimizer state, or checkpoint format.
- Reuse Levanter's existing finished-ledger cache semantics from `lib/levanter/src/levanter/store/cache.py:137` so incomplete caches are treated as missing and rebuilt.
- Keep the generic `Trainer` API unchanged. This is DPO-specific behavior and should live in DPO code, not as a trainer-wide abstraction.
- Support both DPO reference modes:
  - `reference.type=separate`
  - `reference.type=adapter_base`

### Non-Goals

- Do not turn `experiments/eval_dpo.py` into a production library module.
- Do not cache training-time reference log-probs.
- Do not put the cache in device RAM or TPU HBM.
- Do not serialize cached reference log-probs inside training checkpoints.
- Do not introduce generic eval-cache machinery for all trainers.

### Proposed Code Organization

- Add a new library module: `lib/levanter/src/levanter/dpo.py`
- Keep `lib/levanter/src/levanter/main/train_dpo.py` as the orchestration layer:
  - build validation dataset specs
  - decide whether caching is enabled
  - call the cache build/load helper before `trainer.add_eval_hook(...)`
- Leave `experiments/eval_dpo.py` as disposable profiling / validation code. After the library path lands, either:
  - reduce it to a thin profiling script that imports the library helpers, or
  - archive/delete it if nobody needs the standalone path anymore

Reason for using `levanter/dpo.py` instead of adding more code to `train_dpo.py`:

- the cache logic is real runtime behavior, not experiment glue
- it is too DPO-specific for `trainer.py`
- it is not purely data formatting, so it does not belong in `data/text/preference.py`
- it gives us one canonical home for reusable DPO runtime code instead of leaving it trapped in the CLI entrypoint
- it is lower churn than introducing a full `levanter/dpo/` package right now, while still setting up a natural migration path if DPO code keeps growing

### Proposed API Shape

```python
@dataclass(frozen=True)
class ReferenceEvalCacheConfig:
    mode: Literal["disabled", "build_or_load"] = "disabled"
    cache_dir: str | None = None


@dataclass(frozen=True)
class ValidationDatasetSpec:
    name: str
    dataset: AsyncDataset[DpoExample]
    source_cache_path: str
    source_split: str
    slice_start: int | None = None
    slice_end: int | None = None
```

```python
if config.reference_eval_cache.mode == "build_or_load":
    validation_specs = prepare_reference_eval_caches(
        validation_specs,
        config=config,
        trainer=trainer,
        tokenizer=tokenizer,
        model_context=model_context,
        Pos=Pos,
    )

for spec in validation_specs:
    trainer.add_eval_hook(spec.dataset, name=spec.name or None)
```

### Core Design Choices

#### 1. Durable sidecar cache, then RAM

- Source of truth: GCS sidecar cache
- Fast serving path during the run: host RAM numpy arrays
- Not worth streaming from GCS every eval batch because the cache is tiny
- Not worth putting in checkpoint state because it is derived, immutable, and easier to rebuild than to checkpoint safely

Approximate size:

- two `float32` values per example
- `8 bytes/example`
- `23,552` examples is about `188 KB`
- even `1,000,000` examples is only about `8 MB`

Conclusion: persist to GCS for durability, then load into RAM per host once.

#### 2. Keep eval caching out of the generic trainer

- `Trainer.add_eval_hook(...)` in `lib/levanter/src/levanter/trainer.py:606` is intentionally generic.
- We should not add DPO-specific cache controls there.
- The DPO loss in `train_dpo.py` should instead accept either:
  - a normal `DpoExample`
  - a cached `CachedDpoExample`
- That keeps the generic trainer unchanged while letting eval compile a separate cached path when needed.

#### 3. Use a custom dataset wrapper, not `AsyncDataset.map(...)`

- `MappedAsyncDataset` in `lib/levanter/src/levanter/data/dataset.py:227` does not pass the example index into the user callback; it only calls `fn(item, ...)`.
- The cache attach step needs the dataset index so it can read `ref_chosen[index]` and `ref_rejected[index]`.
- So the right production shape is the same basic idea validated in `experiments/eval_dpo.py:477`: a small `AsyncDataset` wrapper that returns `CachedDpoExample` by index.

#### 4. Strict cache identity

Cache key must include:

- validation cache identity
- validation slice identity
  - full validation cache
  - or train-cache slice bounds when `validation_split_fraction` is used
- reference identity
- sequence length
- cache schema version

For `reference.type=separate`, reference identity should include:

- `reference.model_path`
- `reference.is_hf`
- model config class / shape if needed for safety

For `reference.type=adapter_base`, reference identity should include the frozen base initialization source, not the current policy weights:

- `initialize_from_hf`
- `initialize_from_checkpoint_path`
- relevant model/tokenizer identity
- adapter type and settings only if they change the base-model view semantics

#### 5. Finished-or-rebuild semantics

- `TreeCache.load(...)` in `lib/levanter/src/levanter/store/cache.py:137` already refuses unfinished caches.
- We should lean into that:
  - if the cache is missing: build
  - if the ledger is unfinished: build
  - if metadata/version mismatches: rebuild
- Important change from current generic cache loading behavior:
  - metadata mismatch should be treated as a cache miss for this path, not merely a warning

### Implementation Outline

1. Add config surface
   - Extend `TrainDpoConfig` in `lib/levanter/src/levanter/main/train_dpo.py`
   - Extend `SimpleDPOConfig` in `experiments/simple_dpo_config.py`
   - Thread the config through `experiments/defaults.py`

2. Refactor validation-set construction
   - Replace the plain `dict[str, AsyncDataset[DpoExample]]` with `ValidationDatasetSpec`
   - Preserve provenance needed for deterministic cache paths:
     - source cache path
     - split name
     - slice bounds when using `validation_split_fraction`

3. Add `levanter/dpo.py`
   - Move reusable DPO runtime helpers out of `train_dpo.py`
   - Move only the validated cache ideas from `experiments/eval_dpo.py`
   - Add:
     - `DpoModel`
     - `dpo_loss_from_logps`
     - `ReferenceEvalCacheConfig`
     - `CachedDpoExample`
     - `CachedReferenceDataset`
     - `ValidationDatasetSpec`
     - deterministic cache-path builder
     - strict cache metadata builder
     - build-or-load helper that uses `SerialCacheWriter` and `TreeCache`
   - Keep the implementation process-safe:
     - all hosts compute
     - process 0 writes
     - all hosts barrier before load/continue

4. Wire the cache into `train_dpo.py`
   - After validation datasets are built and after the reference identity is known, call the cache helper before training starts
   - Replace uncached eval datasets with cached wrappers when available
   - Extend the DPO loss to branch on `CachedDpoExample` and skip the reference forward passes in eval
   - Leave the training path unchanged

5. Add tests
   - Extend `lib/levanter/tests/test_dpo.py` and add a focused new test file if needed
   - Cover:
     - deterministic cache-key changes when source cache / slice / reference / seq len changes
     - unfinished cache rebuild behavior
     - cached vs uncached DPO-loss parity on a small real model/dataset path
     - `validation_split_fraction` provenance produces distinct cache keys from explicit validation caches

6. Update docs
   - Add a short note to `lib/levanter/docs/guides/DPO-Training.md`
   - Document:
     - what the flag does
     - that it writes a durable sidecar cache
     - that the first run pays a one-time build cost
     - that later runs reuse the completed cache

### Notes

- This should be opt-in at first. Default behavior should remain uncached until the path is validated in real training jobs.
- The cache builder should be invoked before training starts, not lazily inside the first eval hook. That makes failure/rebuild behavior obvious and keeps step-0 eval timing less surprising.
- The first implementation should stay conservative and only cache validation reference log-probs. Training-time caching is a separate problem with different correctness and storage tradeoffs.
- The reference cache should remain a DPO-only feature until there is a second real user of the same pattern.

### Future Work

- If DPO-specific runtime code keeps growing, promote `lib/levanter/src/levanter/dpo.py` into a `lib/levanter/src/levanter/dpo/` package later.
- After the library path lands, archive or shrink `experiments/eval_dpo.py` so the production implementation has one canonical home.
- If eval is still slow after reference caching, revisit eval-specific parameter layout / replicated-weight experiments.

## 2026-03-30 Main Merge Recovery: Preserve Chat Stop Tokens

### What happened

- I reviewed the failed Claude merge attempt and confirmed the main mistake: it tried to resolve the DPO conflicts by provenance ("keep our refactor everywhere") instead of by behavior.
- That would have dropped the recent `origin/main` change that writes `generation_config.json` with chat stop tokens for HF exports. For chat models this is important because inference stacks like vLLM need the chat stop token, not just the tokenizer EOS token, to stop generation correctly after DPO.
- The conflict itself was centered in `lib/levanter/src/levanter/main/train_dpo.py`, but the full feature was not limited to conflict markers. Main also added required support in the shared HF export path and corresponding tests/docs.

### Resolution

- I kept the refactored DPO runtime shape from this branch:
  - `train_dpo.py` remains the canonical entrypoint
  - the adapter/reference abstraction stays in place
  - `model_init.py` remains the model/bootstrap helper
- I ported main's generation-config behavior into the refactored export layer instead of resurrecting main's older inline save path.
- Concretely, I threaded `generation_config` through:
  - `lib/levanter/src/levanter/compat/hf_checkpoints.py`
  - `lib/levanter/src/levanter/adaptation.py`
  - `lib/levanter/src/levanter/lora.py`
  - `lib/levanter/src/levanter/main/train_dpo.py`
- I also brought over the DPO-facing config/docs/tests pieces:
  - `hf_generation_eos_token_ids` config plumbing
  - the DPO guide section on generation stop tokens
  - HF export tests and DPO smoke coverage
- This is a better refactor than replaying main literally because merged LoRA exports now get the same `generation_config.json` behavior as regular DPO.

### Merge + verification status

- I merged current `origin/main` into `dpo-lora` and created merge commit `e6dcb2e96`.
- After the merge, `origin/dpo-lora` was confirmed to be `53` commits ahead and `0` commits behind `origin/main`, so the branch contains current main as of `2026-03-30`.
- Verification for the merge-resolution work:
  - `./infra/pre-commit.py --fix ...` on the touched DPO/HF export files: passed
  - `uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/test_hf_export.py lib/levanter/tests/test_hf_checkpoints.py lib/levanter/tests/test_dpo.py lib/levanter/tests/test_lora_dpo.py`: `49 passed, 2 skipped`

## 2026-03-30 Implementation: Durable Reference Eval Log-Prob Cache

### Outcome

- I implemented the pre-emption-safe cached reference-eval path in the real DPO runtime.
- The production implementation lives in `lib/levanter/src/levanter/dpo.py`. `experiments/eval_dpo.py` remains a prototype/profiling artifact and is not the canonical home for this feature.
- The cache is durable on GCS via Levanter's cache ledger semantics, and then loaded into host RAM for repeated eval use during the run.
- The feature is opt-in and scoped to validation/eval only. Training-time reference computation and checkpoint state are unchanged.

### Code organization

- New shared DPO runtime module:
  - `lib/levanter/src/levanter/dpo.py`
- That module now owns:
  - `DpoModel`
  - shared DPO loss helpers
  - `ReferenceEvalCacheConfig`
  - `CachedDpoExample`
  - `CachedReferenceDataset`
  - `ValidationDatasetSpec`
  - deterministic cache metadata / cache path helpers
  - build/load/build-or-load helpers for reference-eval caches
- `lib/levanter/src/levanter/main/train_dpo.py` stays as orchestration:
  - builds validation specs
  - derives reference identity
  - optionally materializes caches before training
  - swaps eval datasets onto cached wrappers
  - leaves training behavior unchanged

### Important implementation details

- `TrainDpoConfig` now has `reference_eval_cache`, and the config is threaded through:
  - `lib/levanter/src/levanter/main/train_dpo.py`
  - `lib/levanter/src/levanter/main/lora_dpo.py`
  - `experiments/simple_dpo_config.py`
  - `experiments/defaults.py`
- Validation datasets now preserve provenance needed for deterministic cache identity:
  - source cache path
  - split name
  - slice bounds when using `validation_split_fraction`
- Cache identity includes:
  - validation cache provenance
  - slice bounds
  - reference identity
  - sequence length
  - schema/version kind
- Cache loading is intentionally strict:
  - missing cache -> build
  - unfinished ledger -> build
  - metadata mismatch -> rebuild
- The eval loss path now accepts either a normal `DpoExample` or a `CachedDpoExample`, so cached eval skips the reference forward passes while the training path remains the normal uncached computation.

### Testing and docs

- I updated `lib/levanter/tests/test_dpo.py` with focused coverage for:
  - config parsing
  - cache-path identity changes
  - metadata mismatch rejection
  - cache build/load behavior
  - cached-vs-uncached DPO-loss parity
- `lib/levanter/tests/test_lora_dpo.py` now imports the shared DPO helpers from `levanter.dpo`.
- I added a short "Reference Eval Cache" section to `lib/levanter/docs/guides/DPO-Training.md`.

### Verification

- `./infra/pre-commit.py --fix .agents/logbooks/dpo-lora-codex.md experiments/defaults.py experiments/simple_dpo_config.py lib/levanter/docs/guides/DPO-Training.md lib/levanter/src/levanter/dpo.py lib/levanter/src/levanter/main/lora_dpo.py lib/levanter/src/levanter/main/train_dpo.py lib/levanter/tests/test_dpo.py lib/levanter/tests/test_lora_dpo.py`
  - passed
- `uv run --project lib/levanter --group test python -m pytest lib/levanter/tests/test_dpo.py lib/levanter/tests/test_lora_dpo.py`
  - passed with `34 passed, 1 skipped`

### Commit and push status

- I committed the cache work as `35d9c444b` with subject `[dpo] Cache reference eval logprobs`.
- I pushed that commit to `origin/dpo-lora`.
- I did not use the repo's literal `make fix` target for this step because in this dirty worktree it would have swept unrelated local modifications and still would not have included the new untracked `lib/levanter/src/levanter/dpo.py`.
- Instead I ran the required repo fixer directly on the scoped file set, then committed and pushed only the DPO cache work.

## 2026-03-30 Handoff: LoRA Resume Babysitting On central1

### Goal

- Keep babysitting the resumed LoRA DPO run for W&B run `endlboq3` until it either:
  - finishes final eval and writes the final merged HF export, or
  - fails / gets preempted and needs another recovery decision
- The specific thing we were trying to validate was that the LoRA resume path is now correct after the adapter-checkpoint fix:
  - trainer state should load from the east5 training checkpoint
  - base model should be reconstructed from the configured central1 HF source
  - resumed LoRA weights should be overlaid on top of that base model

### Final corrected relaunch

- Several earlier central1 relaunches were wrong:
  - one used unsupported deep `draccus` CLI overrides for dataset cache dirs
  - one used `initialize_from_checkpoint_path` against a LoRA trainer checkpoint and failed with missing base arrays
  - one used `trainer.initialize_from` together with `initialize_from_hf` and hit the config validation error
- The clean relaunch path is:
  - keep `initialize_from_hf` in the config
  - do **not** use `trainer.initialize_from`
  - do **not** use `initialize_from_checkpoint_path`
  - instead pass `--trainer.load_checkpoint_path <east5 trainer checkpoint>`
- Correct running job:
  - Iris job id: `/ahmed/lora-bsv2-mi-b0p1-s2-lr7p5e6-b64-v5p8-central1-resume-20260330-2236`
  - zone / hardware: `us-central1-a`, `v5p-8`
  - W&B run: `https://wandb.ai/marin-community/dpo/runs/endlboq3`
  - config: `lib/levanter/config/dpo/lora_dpo_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-6_seed2_v5p8_central1_b64.yaml`
  - checkpoint source for resume: `gs://marin-us-east5/checkpoints/dpo/lora_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-6_seed2_v5p8_b64/endlboq3/step-835`

### What the successful relaunch already proved

- W&B reused `endlboq3`
- The process loaded the east5 trainer checkpoint via `--trainer.load_checkpoint_path`
- The fixed LoRA resume path ran:
  - log line: `Resuming from step 836, using checkpoint policy weights.`
  - log line: `Adapter checkpoints only store trainable weights. Reconstructing the base policy model from the configured source before overlaying resumed adapter parameters.`
- The run rebuilt the base model from the central1 HF shards and resumed training from step `836`
- Training completed successfully on the resumed job
- A final central1 training checkpoint was written to:
  - `gs://marin-us-central1/checkpoints/dpo/lora_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-6_seed2_v5p8_b64/endlboq3/step-849`

### Current status at handoff

- Timestamp of last status refresh: `2026-03-30 23:26:48 PDT`
- Iris state: `JOB_STATE_RUNNING`
- `preemption_count: 0`
- The job is in the **final eval loop**, not dead or hung
- Latest observed eval progress:
  - `147 / 367` validation batches
  - about `13.2s/it`
  - about `48 minutes` remaining from that sample
- There is still **no** merged HF export under:
  - `gs://marin-us-central1/checkpoints/dpo/lora_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-6_seed2_v5p8_b64/merged_hf/endlboq3/`
- That is expected so far because the export hooks have not fired yet; the job is still inside eval

### Important context for the next agent

- This run is **not** using the new reference-eval cache feature
  - the YAML has no `reference_eval_cache` block
  - so final eval is on the old uncached DPO path
- That is why eval is slow:
  - uncached DPO eval still computes policy chosen
  - policy rejected
  - reference chosen
  - reference rejected
- The long final eval time is therefore expected and is not evidence of another resume failure
- There are occasional log lines `Failed to compress input file`
  - these appeared while the job was otherwise healthy
  - they did not correlate with failure, preemption, or checkpoint corruption

### Babysitting instructions for the next agent

- Keep the long cadence; this is now mostly a wait-for-terminal-state problem
- Watch for three things only:
  - preemption / worker death
  - clean end of eval
  - creation of the final merged HF export
- If the job completes normally, verify:
  - terminal success in Iris
  - final eval finished
  - merged HF export appears under `.../merged_hf/endlboq3/`
  - ideally `step-849` or the final-step-equivalent directory created by the export hook
- If the job fails or gets preempted, do **not** go back to `trainer.initialize_from`
  - the known-good restart pattern is the same `central1_b64` YAML plus `--trainer.load_checkpoint_path <latest training checkpoint>`

## 2026-03-31 DPO Schedule Follow-up

### Scope of this follow-up

- User wanted the `SimpleDPOConfig` surface to support a stable "one pass over the train set" mode for DPO experiments instead of hard-coding step budgets.
- User also wanted a default validation cadence that runs:
  - once before training
  - three evenly spaced interior validations
  - once at the end
- This follow-up was code-only. I did **not** launch a new TPU training run in this pass.

### Config / runtime changes

- Updated [`experiments/simple_dpo_config.py`](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/simple_dpo_config.py) so:
  - `num_train_steps: int | None = None`
  - `num_epochs: float = 1.0`
  - `steps_per_eval: int | None = None`
  - added validation in `__post_init__` for non-positive values
- Updated [`experiments/defaults.py`](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/defaults.py) so `default_dpo(...)` now:
  - keeps explicit `num_train_steps` / `steps_per_eval` if they are set
  - otherwise enables runtime auto-resolution through `TrainDpoOnPodConfig(auto_num_epochs=..., auto_validation_runs=5)`
- Updated [`lib/marin/src/marin/training/training.py`](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/marin/src/marin/training/training.py) so DPO runtime launch now:
  - reads tokenizer stats from the concrete cache path
  - uses `total_elements` as the DPO train-set size
  - applies the same validation-split rule as DPO runtime
  - converts `num_epochs` into `num_train_steps` with `BatchSchedule.find_step_containing_offset(...) + 1`
  - resolves exact interior eval steps for the 5-validation schedule

### Shared stats helper

- Added [`lib/marin/src/marin/processing/tokenize/cache_stats.py`](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/marin/src/marin/processing/tokenize/cache_stats.py).
- This is intentionally a **read-side only** helper for this PR:
  - `TokenizedCacheStats`
  - `tokenized_cache_stats_path(...)`
  - `read_tokenized_cache_stats(...)`
- It uses `rigging.filesystem.url_to_fs(...)` plus `open_url(...)` instead of `os.path.join(...)`.
- Exported the helper from [`lib/marin/src/marin/processing/tokenize/__init__.py`](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/marin/src/marin/processing/tokenize/__init__.py).
- I explicitly did **not** refactor the tokenizer write path in this pass; it still writes the same `.stats.json` schema in place.

### DPO eval scheduling changes

- Updated [`lib/levanter/src/levanter/main/train_dpo.py`](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/main/train_dpo.py) to add:
  - `scheduled_eval_steps`
  - `run_initial_eval`
- Implemented:
  - explicit initial validation before training starts
  - exact interior eval steps via a small hook wrapper
  - final validation through the trainer's existing forced end-of-training hook path
- Removed the old extra DPO-side `trainer.run_hooks(last_info, force=True)` call because [`Trainer.train()`](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/trainer.py) already force-runs hooks at the end.

### Review / correctness conclusions

- I reviewed a competing critique carefully. The main conclusions after code inspection and tests:
  - removing the extra DPO-side `run_hooks(..., force=True)` is **not** a regression because `Trainer.train()` already force-runs hooks at the end
  - the synthetic initial eval logs at **step 0**, not step 1, because `StepInfo.step = state.step - 1`
  - using `.stats.json` is the right source of DPO dataset size for this feature; for preference data the meaningful field is `total_elements`
  - private imports from `train_dpo.py` were avoided in the final version by moving the stats read into a shared helper and keeping the tiny DPO-specific component/split logic local in `training.py`

### Experiment/config updates made in this pass

- Updated [`experiments/tune_lora/common.py`](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/tune_lora/common.py):
  - `LoraTuneSpec.num_epochs = 1.0`
  - removed the explicit `num_train_steps=850`
  - removed the explicit `steps_per_eval=200`
- Updated [`experiments/dpo_bloom_speceval_v2.py`](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/dpo_bloom_speceval_v2.py):
  - switched the main script from `num_train_steps=850` to `num_epochs=1.0`
  - removed the explicit `steps_per_eval=200`
- I intentionally left [`experiments/dpo_bloom_speceval_v2_profile.py`](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/dpo_bloom_speceval_v2_profile.py) alone because that script is explicitly built around fixed eval timing for profiling.

### Expected resolved budgets with current Bloom SpecEval v2 cache

- The train tokenizer stats we have been using report `108,765` train preference pairs.
- Therefore:
  - LoRA sweep config in `experiments/tune_lora/common.py` with batch `64` and `num_epochs=1.0` resolves to `1700` train steps
  - auto eval schedule there is 5 total validations:
    - initial
    - steps `425`, `850`, `1275`
    - final
- Main non-LoRA Bloom SpecEval v2 script with batch `128` and `num_epochs=1.0` resolves to `850` train steps
  - auto eval schedule there is 5 total validations:
    - initial
    - steps `213`, `426`, `639`
    - final

### Validation performed

- Ran:
  - `./infra/pre-commit.py --fix lib/marin/src/marin/processing/tokenize/cache_stats.py lib/marin/src/marin/processing/tokenize/__init__.py lib/marin/src/marin/training/training.py tests/test_training.py`
  - `uv run python -m pytest -o addopts='' tests/test_training.py`
  - `cd lib/levanter && uv run --group test python -m pytest tests/test_dpo.py tests/test_lora_dpo.py`
  - `./infra/pre-commit.py --fix experiments/dpo_bloom_speceval_v2.py`
- Results:
  - targeted Marin checks passed
  - `tests/test_training.py`: `8 passed`
  - Levanter DPO tests: `35 passed, 1 skipped`

### Current handoff state

- No new runtime experiment launched in this follow-up.
- Code now supports:
  - explicit step-based DPO configs when a script sets `num_train_steps` / `steps_per_eval`
  - one-epoch auto budgeting plus 5-point eval cadence when those fields are left unset
- The main Bloom SpecEval v2 scripts now opt into the new auto path directly.

## 2026-04-01 DPO LM Eval Integration

### Goal

- Add the standard pretraining LM eval suites to DPO runs without mixing them into preference validation.
- Apply this automatically to both:
  - the standard Bloom SpecEval v2 DPO run
  - the LoRA DPO sweep, via `default_dpo(...)`

### What changed

- Updated [`experiments/defaults.py`](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/defaults.py) so `default_dpo(...)` now always builds a separate `lm_validation_data` from `default_validation_sets(tokenizer=...)`.
- This reuses the same validation bundle used by pretraining:
  - Paloma tokenized eval sets
  - Uncheatable Eval tokenized eval sets
- Updated [`lib/levanter/src/levanter/main/train_dpo.py`](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/main/train_dpo.py) to add:
  - `lm_validation_data: LmDataConfig | None`
  - `lm_validation_prefix: str = "lm_eval"`
- DPO now runs LM evals through `levanter.eval.cb_tagged_lm_evaluate(...)` on the same cadence as DPO preference eval:
  - initial eval before training
  - scheduled interior evals
  - final forced eval at end of training

### Logging behavior

- Preference validation remains under `eval/...` and still reports DPO loss metrics for preference datasets.
- LM eval metrics are logged separately under `lm_eval/...` so they do not collide with preference loss:
  - `lm_eval/paloma/...`
  - `lm_eval/uncheatable_eval/...`
- This keeps DPO-specific validation and LM generalization validation distinct in W&B.

### Implementation details

- I did **not** try to cram `default_validation_sets()` into `PreferenceLmDataConfig`; those datasets are LM eval data, not preference pairs.
- Instead, DPO now carries two parallel validation paths:
  - preference validation for DPO loss
  - LM validation for next-token loss / bpb on Paloma and Uncheatable
- Added small callback wrappers in `train_dpo.py` to prevent duplicate eval runs when:
  - initial eval happens at step 0
  - trainer force-runs hooks at the final step

### Validation performed

- Ran:
  - `make fix`
  - `uv run python -m pytest -o addopts='' tests/test_training.py`
  - `uv run --directory lib/levanter --group test python -m pytest tests/test_dpo.py tests/test_lora_dpo.py`
- Results:
  - `tests/test_training.py`: `9 passed`
  - Levanter DPO tests: `37 passed, 1 skipped`

### Resulting behavior

- Any DPO experiment that goes through `default_dpo(...)` now picks up Paloma + Uncheatable LM evals automatically.
- That includes:
  - [`experiments/dpo_bloom_speceval_v2.py`](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/dpo_bloom_speceval_v2.py)
  - [`experiments/tune_lora/common.py`](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/tune_lora/common.py)

### Current LoRA run behavior

- If a Bloom SpecEval v2 LoRA run is launched now from [`experiments/tune_lora/common.py`](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/tune_lora/common.py):
  - training uses the existing train preference cache with `108,765` train pairs
  - validation uses the deduped validation cache with `2,606` validation pairs
  - `num_epochs=1.0` and batch size `64` resolve to `1,700` train steps
  - DPO preference eval runs 5 times total:
    - initial
    - steps `425`, `850`, `1275`
    - final
  - Paloma + Uncheatable LM evals run on that exact same schedule under `lm_eval/...`
- The branch containing this work was pushed as commit `1e0e5fe9f`:
  - `[dpo] Add LM eval suites to DPO runs`

## Updated Sweep Plan

### Objective

- Expand the LoRA DPO learning-rate sweep around the current Bloom SpecEval v2 setup.
- Use **two seeds per hyperparameter combination**.
- Keep the current executor-native training behavior:
  - `num_epochs=1.0`
  - preference eval on the 5-point schedule
  - Paloma + Uncheatable LM evals via `default_validation_sets()`

### Constraints carried forward

- Keep:
  - `beta=0.1`
  - batch size `64`
  - rank `64`
  - `alpha=64`
  - `dropout=0.0`
  - `reference.type=adapter_base`
  - `reference_eval_cache.mode=build_or_load`
  - `train_seq_len=max_seq_len=4096`
  - `v5p-8`
- Do **not** reuse the exact same slug string for new combinations.
- Reuse the existing slug pattern:
  - `bloom_speceval_v2_marin_instruct_lora_beta0p1_lr{lr}_seed{seed}_b64_v5p8`

### Rationale

- The current scripted sweep is concentrated in the `5e-6` to `1e-5` range.
- The LoRA DPO best-practices note suggests the most promising regime is often **lower**, roughly `5e-7` to `5e-6`.
- So the next useful move is not to push higher than `1e-5`, but to add denser lower-LR points while also filling in missing seed coverage.

### Recommended next LR grid

- Keep existing scripted LRs:
  - `5e-6`
  - `6.25e-6`
  - `7.5e-6`
  - `8.75e-6`
  - `1e-5`
- Add lower-LR points:
  - `1e-6`
  - `2.5e-6`
  - `3.75e-6`
  - `4.5e-6`

### Recommended seed policy

- Run every LR with:
  - `seed=0`
  - `seed=2`

### Smallest sensible next expansion

- If we want to stay conservative on job count, the minimum good expansion is:
  - add `2.5e-6`
  - add `3.75e-6`
  - add missing `seed=0` companions for the currently seed-2-only LRs
- This gives broader LR coverage without exploding the sweep immediately.

### Full suggested new job matrix

- Existing or already represented:
  - `lr=5e-6`, seeds `0, 2`
  - `lr=6.25e-6`, seeds `0, 2`
  - `lr=7.5e-6`, seeds `0, 2`
  - `lr=8.75e-6`, seeds `0, 2`
  - `lr=1e-5`, seeds `0, 2`
- New lower-LR additions:
  - `lr=1e-6`, seeds `0, 2`
  - `lr=2.5e-6`, seeds `0, 2`
  - `lr=3.75e-6`, seeds `0, 2`
  - `lr=4.5e-6`, seeds `0, 2`

### Example slug names for new lower-LR runs

- `bloom_speceval_v2_marin_instruct_lora_beta0p1_lr1e6_seed0_b64_v5p8`
- `bloom_speceval_v2_marin_instruct_lora_beta0p1_lr1e6_seed2_b64_v5p8`
- `bloom_speceval_v2_marin_instruct_lora_beta0p1_lr2p5e6_seed0_b64_v5p8`
- `bloom_speceval_v2_marin_instruct_lora_beta0p1_lr2p5e6_seed2_b64_v5p8`
- `bloom_speceval_v2_marin_instruct_lora_beta0p1_lr3p75e6_seed0_b64_v5p8`
- `bloom_speceval_v2_marin_instruct_lora_beta0p1_lr3p75e6_seed2_b64_v5p8`
- `bloom_speceval_v2_marin_instruct_lora_beta0p1_lr4p5e6_seed0_b64_v5p8`
- `bloom_speceval_v2_marin_instruct_lora_beta0p1_lr4p5e6_seed2_b64_v5p8`

### Expected behavior for each new run

- Each run should:
  - resolve to `1,700` train steps from `num_epochs=1.0`
  - run DPO preference eval at:
    - initial
    - steps `425`, `850`, `1275`
    - final
  - run Paloma + Uncheatable LM evals on the same schedule under `lm_eval/...`

### Implemented scripted sweep

- Added full `9 x 2` wrapper coverage in [`experiments/tune_lora/`](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/tune_lora):
  - learning rates:
    - `1e-6`
    - `2.5e-6`
    - `3.75e-6`
    - `4.5e-6`
    - `5e-6`
    - `6.25e-6`
    - `7.5e-6`
    - `8.75e-6`
    - `1e-5`
  - seeds:
    - `0`
    - `2`

## Iris Multi-Region Launch Plan

### Goal

- Launch each LoRA sweep job on Iris with `v5p-8`.
- Allow Iris to place the job in either:
  - `us-central1`
  - `us-east5`
- We do not care which region wins as long as the job lands on `v5p-8`.

### Syntax

- Use repeated region flags:
  - `--region us-central1 --region us-east5`
- Do not use:
  - `us-east5a`
- `us-east5` is the region.
- `us-east5-a` is the zone.

### Cache status as of 2026-04-01

- Train preference cache is present in both regions.
- Deduped validation preference cache is present in both regions.
- `default_validation_sets()` is present `23/23` in both regions.
- That means the current LoRA sweep is safe to launch with dual-region placement.

### Expected runtime behavior per job

- Each LoRA job should:
  - request `v5p-8`
  - resolve `num_epochs=1.0` to `1,700` train steps
  - run DPO preference eval at:
    - initial
    - steps `425`, `850`, `1275`
    - final
  - run Paloma + Uncheatable LM evals on the same schedule under `lm_eval/...`

### Exact commands to run

Run these from the repo root:

- If launching from this worktree, first make sure local `wandb_claude_data/` and `wandb_data/` are removed, ignored, or moved out of tree; Iris previously rejected a `73MB` source bundle from those directories alone.

```bash
uv run iris --config lib/iris/examples/marin.yaml job run --job-name bloom-speceval-v2-lora-beta0p1-lr1e6-seed0-b64 --tpu v5p-8 --region us-central1 --region us-east5 --no-wait -- uv run python experiments/tune_lora/beta0p1_lr1e6_seed0_b64.py
uv run iris --config lib/iris/examples/marin.yaml job run --job-name bloom-speceval-v2-lora-beta0p1-lr1e6-seed2-b64 --tpu v5p-8 --region us-central1 --region us-east5 --no-wait -- uv run python experiments/tune_lora/beta0p1_lr1e6_seed2_b64.py
uv run iris --config lib/iris/examples/marin.yaml job run --job-name bloom-speceval-v2-lora-beta0p1-lr2p5e6-seed0-b64 --tpu v5p-8 --region us-central1 --region us-east5 --no-wait -- uv run python experiments/tune_lora/beta0p1_lr2p5e6_seed0_b64.py
uv run iris --config lib/iris/examples/marin.yaml job run --job-name bloom-speceval-v2-lora-beta0p1-lr2p5e6-seed2-b64 --tpu v5p-8 --region us-central1 --region us-east5 --no-wait -- uv run python experiments/tune_lora/beta0p1_lr2p5e6_seed2_b64.py
uv run iris --config lib/iris/examples/marin.yaml job run --job-name bloom-speceval-v2-lora-beta0p1-lr3p75e6-seed0-b64 --tpu v5p-8 --region us-central1 --region us-east5 --no-wait -- uv run python experiments/tune_lora/beta0p1_lr3p75e6_seed0_b64.py
uv run iris --config lib/iris/examples/marin.yaml job run --job-name bloom-speceval-v2-lora-beta0p1-lr3p75e6-seed2-b64 --tpu v5p-8 --region us-central1 --region us-east5 --no-wait -- uv run python experiments/tune_lora/beta0p1_lr3p75e6_seed2_b64.py
uv run iris --config lib/iris/examples/marin.yaml job run --job-name bloom-speceval-v2-lora-beta0p1-lr4p5e6-seed0-b64 --tpu v5p-8 --region us-central1 --region us-east5 --no-wait -- uv run python experiments/tune_lora/beta0p1_lr4p5e6_seed0_b64.py
uv run iris --config lib/iris/examples/marin.yaml job run --job-name bloom-speceval-v2-lora-beta0p1-lr4p5e6-seed2-b64 --tpu v5p-8 --region us-central1 --region us-east5 --no-wait -- uv run python experiments/tune_lora/beta0p1_lr4p5e6_seed2_b64.py
uv run iris --config lib/iris/examples/marin.yaml job run --job-name bloom-speceval-v2-lora-beta0p1-lr5e6-seed0-b64 --tpu v5p-8 --region us-central1 --region us-east5 --no-wait -- uv run python experiments/tune_lora/beta0p1_lr5e6_seed0_b64.py
uv run iris --config lib/iris/examples/marin.yaml job run --job-name bloom-speceval-v2-lora-beta0p1-lr5e6-seed2-b64 --tpu v5p-8 --region us-central1 --region us-east5 --no-wait -- uv run python experiments/tune_lora/beta0p1_lr5e6_seed2_b64.py
uv run iris --config lib/iris/examples/marin.yaml job run --job-name bloom-speceval-v2-lora-beta0p1-lr6p25e6-seed0-b64 --tpu v5p-8 --region us-central1 --region us-east5 --no-wait -- uv run python experiments/tune_lora/beta0p1_lr6p25e6_seed0_b64.py
uv run iris --config lib/iris/examples/marin.yaml job run --job-name bloom-speceval-v2-lora-beta0p1-lr6p25e6-seed2-b64 --tpu v5p-8 --region us-central1 --region us-east5 --no-wait -- uv run python experiments/tune_lora/beta0p1_lr6p25e6_seed2_b64.py
uv run iris --config lib/iris/examples/marin.yaml job run --job-name bloom-speceval-v2-lora-beta0p1-lr7p5e6-seed0-b64 --tpu v5p-8 --region us-central1 --region us-east5 --no-wait -- uv run python experiments/tune_lora/beta0p1_lr7p5e6_seed0_b64.py
uv run iris --config lib/iris/examples/marin.yaml job run --job-name bloom-speceval-v2-lora-beta0p1-lr7p5e6-seed2-b64 --tpu v5p-8 --region us-central1 --region us-east5 --no-wait -- uv run python experiments/tune_lora/beta0p1_lr7p5e6_seed2_b64.py
uv run iris --config lib/iris/examples/marin.yaml job run --job-name bloom-speceval-v2-lora-beta0p1-lr8p75e6-seed0-b64 --tpu v5p-8 --region us-central1 --region us-east5 --no-wait -- uv run python experiments/tune_lora/beta0p1_lr8p75e6_seed0_b64.py
uv run iris --config lib/iris/examples/marin.yaml job run --job-name bloom-speceval-v2-lora-beta0p1-lr8p75e6-seed2-b64 --tpu v5p-8 --region us-central1 --region us-east5 --no-wait -- uv run python experiments/tune_lora/beta0p1_lr8p75e6_seed2_b64.py
uv run iris --config lib/iris/examples/marin.yaml job run --job-name bloom-speceval-v2-lora-beta0p1-lr1e5-seed0-b64 --tpu v5p-8 --region us-central1 --region us-east5 --no-wait -- uv run python experiments/tune_lora/beta0p1_lr1e5_seed0_b64.py
uv run iris --config lib/iris/examples/marin.yaml job run --job-name bloom-speceval-v2-lora-beta0p1-lr1e5-seed2-b64 --tpu v5p-8 --region us-central1 --region us-east5 --no-wait -- uv run python experiments/tune_lora/beta0p1_lr1e5_seed2_b64.py
```
