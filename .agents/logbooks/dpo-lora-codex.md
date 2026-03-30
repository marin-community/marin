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
