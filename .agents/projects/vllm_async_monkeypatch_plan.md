# Async-Native vLLM TPU Bootstrap Monkeypatch Plan

## Goal

Keep:

- async-native vLLM serving
- Marin's WorkerExtension weight injection
- no edits to the `vllm-tpu` / `tpu-inference` source tree

Avoid:

- the slow TPU bootstrap paths currently reached by `load_format="dummy"`
  under either:
  - `MODEL_IMPL_TYPE="vllm"` (`v15`)
  - `MODEL_IMPL_TYPE="flax_nnx"` (`v16`)

The target is a Marin-owned runtime patch that makes async-native engine startup
construct the minimum valid TPU runner state quickly, then rely on Marin's
existing streamed shard injection to populate real weights.

## What we learned

### `v15`: forced `MODEL_IMPL_TYPE="vllm"` is wrong

The old async-native backend forced `MODEL_IMPL_TYPE="vllm"`, which sent TPU
startup through:

- `tpu_inference.models.common.model_loader.get_vllm_model(...)`
- `tpu_inference.models.vllm.vllm_model_wrapper.VllmModelWrapper.load_weights()`
- upstream `DummyModelLoader.load_weights(...)`
- upstream `process_weights_after_loading(...)`

That path spent ~15 minutes on dummy random weights and then stalled in
post-load processing before Marin injected any real weights.

### `v16`: switching to `MODEL_IMPL_TYPE="auto"` helped, but not enough

`v16` proved:

- `MODEL_IMPL_TYPE='auto'` took effect
- TPU resolved to `flax_nnx`
- startup entered `model_loader.get_flax_model(...)`

It still timed out after 1200s.

The important detail is why.

In `tpu_inference.models.common.model_loader._get_nnx_model(...)`:

- if `load_format == "dummy"` and the model class is **not**
  `LoadableWithIterator`, the TPU fork takes a concrete random-init + sharded
  model branch
- Llama's JAX class is `LlamaForCausalLM(nnx.Module)`, not
  `LoadableWithIterator`

So for Llama, `flax_nnx + dummy` still means "build a concrete sharded model
with initialized arrays up front", not "cheap abstract placeholder model".

That is why escaping the PyTorch wrapper path did not solve startup latency.

## Honest constraint

We can do this without editing the dependency source tree.

We cannot do it without depending on `vllm-tpu` / `tpu-inference` internals.

The plan below accepts that explicitly.

The key internal contract is:

- `TPUModelRunner.load_model()` expects `get_model(...)` to return a usable
  `(model_fn, compute_logits_fn, ..., state, ..., model)` bundle
- later `WorkerExtension.update_weight()` calls `self.model_runner._sync_weights(...)`
- `_sync_weights(...)` updates `self.state` by shape/dtype/mapping

So the bootstrap path must produce a valid initial `state` tree with the right
shapes/sharding metadata, even if the values are placeholders.

## Recommended plan

### Overview

Implement a Marin-only runtime monkeypatch for supported `flax_nnx` text models
that replaces the expensive `dummy` bootstrap with:

1. abstract model creation
2. cheap concrete zero-state materialization with correct shapes/sharding
3. normal runner wiring (`model_fn`, `compute_logits_fn`, `state`)
4. existing Marin streamed shard injection via `_sync_weights(...)`

This keeps async-native serving and avoids editing the dependency checkout.

### Why this is the best plan

- it targets the actual long pole from `v16`: `get_flax_model(...)`
- it preserves the serving architecture we want
- it preserves the weight injection mechanism we already know works
- it is narrower and lower-risk than replacing AsyncLLM startup wholesale
- it avoids the dead-end of more timeout increases

## Proposed design

### 1. Add a Marin-specific bootstrap mode

Introduce a Marin-only env var or load format marker, for example:

- env var: `MARIN_VLLM_FAST_BOOTSTRAP=1`
- load format: `marin_zero_bootstrap`

Recommendation:

- use **both**

Reason:

- `load_format` gives us an explicit startup mode we can route on
- env var makes it easy to scope monkeypatch behavior to Marin async-native only

### 2. Register a custom load format at runtime

In Marin startup code before `AsyncLLM.from_engine_args(...)`, register:

```python
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader


@register_model_loader("marin_zero_bootstrap")
class MarinZeroBootstrapLoader(BaseModelLoader):
    def download_model(self, model_config):
        return None

    def load_weights(self, model, model_config):
        # Intentionally a no-op; the real behavior is in the TPU monkeypatch
        return None
```

This is not sufficient by itself, but it gives us a clean marker in
`vllm_config.load_config.load_format`.

### 3. Monkeypatch the TPU JAX bootstrap path, not upstream AsyncLLM

Patch a TPU internal seam only when:

- `MARIN_VLLM_FAST_BOOTSTRAP=1`
- `MODEL_IMPL_TYPE` resolves to `flax_nnx`
- `load_format == "marin_zero_bootstrap"`

Primary target:

- `tpu_inference.models.common.model_loader._get_nnx_model`

Reason:

- this is where the current `dummy` branch decides between:
  - concrete random-init + sharded model
  - abstract-model + loader path
- we want a third branch that creates a valid concrete state without random
  initialization or HF weight loading

### 4. Implement a Marin zero-bootstrap branch

For supported models, the patched branch should:

1. build the abstract model with `nnx.eval_shape(...)`
2. traverse parameters and materialize zero arrays with the correct
   shape/dtype/sharding metadata
3. assign those arrays using the TPU helper that already applies sharding
4. call the same final JIT/model wiring that `get_flax_model(...)` normally uses

Pseudo-shape:

```python
def marin_fast_bootstrap_nnx_model(model_class, vllm_config, rng, mesh):
    abstract_model_fn = lambda: model_class(vllm_config, rng, mesh)
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(abstract_model_fn)

    for param_name, param in model.named_parameters():
        zero_weight = jnp.zeros(param.value.shape, dtype=param.value.dtype)
        assign_and_shard_param(param, zero_weight, param_name)

    with jax.set_mesh(mesh):
        jit_model = create_jit_model(model, use_qwix_on_abstract_model=False)

    return jit_model
```

This relies on TPU internal helpers like:

- `assign_and_shard_param(...)`
- param metadata for sharding
- the expected shape of the returned NNX state

### 5. Scope this to supported models only

Do **not** try to make this generic on the first pass.

Supported first-pass models:

- Llama text-only
- possibly Qwen text-only if it behaves similarly

Explicitly reject:

- multimodal models
- MoE models
- quantized variants we have not checked
- architectures that use custom `process_weights_after_loading(...)` semantics

Reason:

- the aim is to get async-native serving working for the Marin fast-load cases,
  not to design a universal TPU bootstrap system in one go

### 6. Keep WorkerExtension injection exactly as-is

Do **not** redesign the injection path at the same time.

Keep:

- `WorkerExtension.update_weight(...)`
- `TPUModelRunner._sync_weights(...)`
- current shard streaming / reshape / mapping logic

This keeps the diff focused on startup bootstrap only.

## Why this should work

`_sync_weights(...)` does not need the initial values to be random or real. It
needs the target `state` tree to exist with:

- matching shapes
- matching dtypes
- compatible sharding metadata

The transfer logic checks exactly that:

- target leaf shape
- target leaf dtype
- then assigns the new value

So the bootstrap state can be zeros if the runner only needs a concrete state
tree before real weight injection.

## Main risks

### Risk 1: zero state is not enough for later TPU initialization

Possible failure:

- after `load_model()` returns, some later runner step expects post-load
  transformations or module flags that the normal loader would have set

Mitigation:

- first target unquantized Llama text-only
- add a minimal optional post-pass only if needed for those modules

### Risk 2: abstract model + zero materialization is still expensive

Possible failure:

- device allocation/sharding itself remains too slow

Mitigation:

- still likely much cheaper than random full-model init + extra processing
- if needed, next step would be to zero-materialize lazily or patch more deeply

### Risk 3: monkeypatch seam is brittle across package versions

This is real.

Mitigation:

- guard the patch tightly
- log the exact package path / branch chosen
- fail fast if expected functions/classes are not present
- keep the monkeypatch isolated in one Marin module

## Implementation steps

### Phase 1: patch infrastructure

1. Add a Marin TPU bootstrap monkeypatch module, for example:
   - `lib/marin/src/marin/inference/vllm_tpu_bootstrap_patch.py`
2. Register `marin_zero_bootstrap` load format
3. Patch `_get_nnx_model(...)` or `get_flax_model(...)` under
   `MARIN_VLLM_FAST_BOOTSTRAP=1`
4. Add explicit logging:
   - `Applying Marin fast TPU bootstrap patch`
   - `Using Marin zero-bootstrap flax_nnx path`

### Phase 2: wire async-native startup to use it

In `vllm_async.py`:

1. set:
   - `MODEL_IMPL_TYPE=auto`
   - `MARIN_VLLM_FAST_BOOTSTRAP=1`
2. change CLI args from:
   - `--load-format dummy`
   to:
   - `--load-format marin_zero_bootstrap`
3. install the monkeypatch before `AsyncLLM.from_engine_args(...)`

### Phase 3: validation

Local:

- existing `tests/vllm/test_vllm_inprocess_backend.py`
- new tests only for:
  - env/default wiring
  - custom load format registration
  - monkeypatch guard behavior

Remote:

- rerun 8B async-native stress test

Success signals:

- startup log shows:
  - `MODEL_IMPL_TYPE='auto'`
  - `load_format='marin_zero_bootstrap'`
  - `Using Marin zero-bootstrap flax_nnx path`
- worker enters `model_loader.get_flax_model`
- startup reaches shard injection without a 15–20 minute stall

## Fallback if `_get_nnx_model` patch is too fragile

Second-best plan:

- patch `get_flax_model(...)` directly for supported architectures
- bypass `_get_nnx_model(...)`
- construct the `(model_fn, compute_logits_fn, ..., state)` bundle directly

This is more invasive but can still live entirely in Marin as a runtime patch.

## Recommendation

Proceed with the monkeypatch plan, but keep the first pass narrow:

- only async-native
- only Marin
- only `flax_nnx`
- only supported text models

Do **not** spend more time trying to make built-in `dummy` startup fast. The
logs already show that both built-in bootstrap paths are the wrong shape for
Marin's async-native fast-load goal.
