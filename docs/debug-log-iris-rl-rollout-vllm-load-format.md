# Debugging log for iris RL rollout vLLM load format

Goal: fix the rollout-worker startup failure in the live Iris RL run so the 500-step path can proceed past first rollout generation.

## Initial status

Live run `/ahmed/iris-rl-oom-b120-uc1-r7` reached:
- root launcher running,
- RL coordinator running,
- trainer child running,
- rollout child allocated on `v5p-8`.

The rollout child then failed during async vLLM engine initialization with:
- `pydantic_core._pydantic_core.ValidationError`
- `To load a model from S3, 'load_format' must be 'runai_streamer' or 'runai_streamer_sharded', but got 'dummy'`

The traceback lands in:
- `lib/marin/src/marin/rl/rl_experiment_utils.py`
- `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py`
- `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py`

## Hypothesis 1

RL rollout config still assumes `load_format="dummy"` is valid for all vLLM model paths.
That assumption no longer holds for object-store-backed model artifacts (`gs://...`) because vLLM now validates those paths through the streamer path and rejects `dummy`.

## Changes to make

- `lib/marin/src/marin/rl/rl_experiment_utils.py`
  - choose `runai_streamer` when the resolved model artifact is an object-store path (`gs://` or `s3://`)
  - keep `dummy` for non-object-store paths
- `tests/rl/test_rl_experiment_utils.py`
  - assert `runai_streamer` for GCS-backed model artifacts
  - assert `dummy` remains for non-object-store paths

## Future Work

- [ ] If rollout still fails after this patch, inspect the next async vLLM bootstrap error before changing placement again.
- [ ] Consider centralizing the object-store `load_format` decision with the existing Marin vLLM server helpers.
- [ ] Consider removing the stale `dummy` hardcode from `experiments/exp_iris_rl_direct.py` if that path is still used.

## Results

Patch prepared locally.
Next steps:
- run focused RL tests,
- if green, stop the broken `r7` tree,
- relaunch with the patched bundle,
- continue babysit loop until first rollout arrives or a new concrete failure appears.

## Hypothesis 2

The rollout worker now reaches async vLLM engine init and receives the initial Arrow Flight weight payload, but the in-process weight application path still keys model-specific tensor mappings by the canonical HF model id.

After switching RL bootstrap to a regional `gs://...` model artifact, that weight-update path is now receiving the artifact path instead of the canonical model id. The lookup then fails here:
- `marin.rl.environments.inference_ctx.inflight.worker.WorkerExtension.update_weight`
- `MODEL_MAPPINGS[model_name]`
- `MODEL_TRANSPOSE_KEYS[model_name]`

Observed live failure in `r8`:
- `KeyError: 'No MODEL_MAPPING registered for model: gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f'`

## Changes to make

- `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py`
  - add a separate `canonical_model_name` to `vLLMInferenceContextConfig`
  - keep `model_name` as the actual load path for tokenizer/vLLM
  - use `canonical_model_name` for renderer/model-family selection and weight-mapping lookup
- `lib/marin/src/marin/rl/environments/inference_ctx/async_vllm.py`
  - send `canonical_model_name` through async weight-update RPC instead of the `gs://...` path
- `lib/marin/src/marin/rl/rl_experiment_utils.py`
  - set `canonical_model_name=config.model_config.name` when building RL job config
- tests:
  - `tests/rl/test_rl_experiment_utils.py`
  - `tests/rl/test_inference_ctx.py`

## Results

Patch prepared locally and validated.

Focused validation:
- `uv run pytest -q tests/rl/test_rl_experiment_utils.py tests/rl/test_inference_ctx.py` -> `17 passed`
- `uv run python -m compileall ...` on touched RL files -> success

Next steps:
- run pre-commit on touched files,
- stop broken `r8`,
- relaunch with the canonical-model-name fix,
- continue babysit loop until first rollout batch lands or a new concrete failure appears.

## Hypothesis 3

The canonical-model-name fix allowed the async rollout worker to reach the actual `tpu_inference` weight application path.
That exposed a second bug in the async worker extension: it calls the internal `self.model_runner._sync_weights(...)` entry point directly.

In the live `r9` run, the TPU-side model resolved as `MistralForCausalLM` and fell back to a vLLM-native PyTorch definition. In that fallback mode, the internal `_sync_weights` path crashes with:
- `AttributeError: 'dict' object has no attribute 'flat_state'`

Observed live traceback:
- `marin.rl.environments.inference_ctx.inflight.worker.WorkerExtension.update_weight`
- `self.model_runner._sync_weights(...)`
- `tpu_inference.models.jax.utils.weight_utils.transfer_state_with_mappings`
- `tgt_state.flat_state()`

The public `sync_weights(...)` path is what the existing sync vLLM code and tests already use. That path is the safer contract to call from the worker extension.

## Changes to make

- `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py`
  - switch `WorkerExtension.update_weight` from `self.model_runner._sync_weights(...)` to `self.sync_weights(...)`
  - keep the same mappings / transpose keys / `reshard_fn=None`
- `tests/rl/test_inference_ctx.py`
  - add a focused unit test that verifies `WorkerExtension.update_weight` delegates to public `sync_weights` with an `nnx.State`

## Results

Patch prepared locally and validated.

Focused validation:
- `uv run pytest -q tests/rl/test_inference_ctx.py tests/rl/test_rl_experiment_utils.py` -> `18 passed`
- `uv run python -m compileall lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py tests/rl/test_inference_ctx.py` -> success

Next steps:
- run pre-commit on touched files,
- stop broken `r9`,
- relaunch with the worker-extension sync fix,
- continue babysitting until rollout generation actually begins or another failure appears.
