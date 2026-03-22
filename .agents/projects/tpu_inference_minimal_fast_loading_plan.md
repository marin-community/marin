# TPU Inference Minimal Fast-Loading Plan

## Goal

Make the smallest possible change in `tpu-inference` that preserves Marin's
fast model-loading path for async-native vLLM, without continuing the current
runtime monkeypatch approach.

This plan is intentionally scoped to **model loading only**.

It does **not** solve the later TPU KV-cache allocation bottleneck that still
blocks full server readiness. That is a separate problem.

## Bottom line

The minimal `tpu-inference` change is:

1. keep using `load_format="dummy"`
2. add one explicit `model_loader_extra_config` flag
3. in `tpu_inference.models.common.model_loader._get_nnx_model(...)`, replace
   the current random-init concrete dummy path for supported `flax_nnx` models
   with an **abstract-state bootstrap** path

That upstreams the exact seam that Marin is currently monkeypatching.

## Why this is the smallest good change

The current expensive branch is in:

- [`model_loader.py:141`](/Users/ahmed/code/vllm_tpu_multi/tpu-inference/tpu_inference/models/common/model_loader.py:141)

Today, for:

- `load_format == "dummy"`
- model class is **not** `LoadableWithIterator`

the TPU loader does this:

- build a concrete sharded model with `@jax.jit`
- random-initialize weights
- optionally `initialize_cache()`

That is exactly the branch that made Llama startup slow before Marin’s patch.

Marin’s current monkeypatch proves that this branch can instead return an
abstract-state model quickly enough for later `_sync_weights(...)` to populate
real weights.

## Why not a new load format

That would be a larger cross-repo change because:

- vLLM’s parser / `LoadConfig` validation would also need to accept the new
  format
- Marin would need CLI changes on top

We already have a smaller explicit control plane:

- `LoadConfig.model_loader_extra_config`
- vLLM already passes it through

Relevant code:

- [`arg_utils.py:514`](/Users/ahmed/code/vllm_tpu_multi/vllm/vllm/engine/arg_utils.py:514)
- [`load.py:70`](/Users/ahmed/code/vllm_tpu_multi/vllm/vllm/config/load.py:70)

So the smallest upstreamable switch is a new flag inside
`model_loader_extra_config`, for example:

```python
{"tpu_abstract_dummy_bootstrap": True}
```

## Proposed `tpu-inference` change

### 1. Add one helper in `model_loader.py`

File:

- [`tpu_inference/models/common/model_loader.py`](/Users/ahmed/code/vllm_tpu_multi/tpu-inference/tpu_inference/models/common/model_loader.py)

Add a helper like:

```python
def _use_abstract_dummy_bootstrap(vllm_config: VllmConfig, model_class: type[Any]) -> bool:
    extra = vllm_config.load_config.model_loader_extra_config or {}
    if not extra.get("tpu_abstract_dummy_bootstrap", False):
        return False
    if vllm_config.load_config.load_format != "dummy":
        return False
    if issubclass(model_class, LoadableWithIterator):
        return False
    if apply_qwix_on_abstract_model(vllm_config):
        return False
    quant_config = getattr(vllm_config.model_config.hf_config, "quantization_config", None)
    if quant_config:
        return False
    return True
```

### 2. Add one helper to build the abstract bootstrap model

Still in `model_loader.py`, add a helper equivalent to the Marin patch:

```python
def _build_abstract_bootstrap_model(model_class, vllm_config, rng, mesh):
    def create_abstract_model():
        return model_class(vllm_config, rng, mesh)

    with jax.set_mesh(mesh):
        model = nnx.eval_shape(create_abstract_model)
        abstract_state = nnx.state(model)

        def seed_rng_variable(path, variable):
            if path and path[0] == "rng" and path[-1] == "key":
                return variable.replace(value=rng)
            return variable

        seeded_state = nnx.map_state(seed_rng_variable, abstract_state)
        nnx.update(model, seeded_state)
    return model
```

This is intentionally narrow:

- no quantization support in v1
- no Qwix support in v1
- only activates when explicitly requested

### 3. Change `_get_nnx_model(...)` to use it

Replace the current branch:

```python
if vllm_config.load_config.load_format == "dummy" and not issubclass(model_class, LoadableWithIterator):
    ...
```

with:

```python
if _use_abstract_dummy_bootstrap(vllm_config, model_class):
    return _build_abstract_bootstrap_model(model_class, vllm_config, rng, mesh)

if vllm_config.load_config.load_format == "dummy" and not issubclass(model_class, LoadableWithIterator):
    ...
```

That preserves current behavior by default and only changes the explicit
fast-bootstrap opt-in path.

## Why this should work

The returned object only needs to satisfy the later `get_flax_model(...)`
contract:

- [`get_flax_model(...)`](/Users/ahmed/code/vllm_tpu_multi/tpu-inference/tpu_inference/models/common/model_loader.py:250)

That function immediately does:

- `graphdef, state = nnx.split(jit_model)`
- constructs jitted `run_model`, `run_compute_logits`, etc.
- returns `state` to the runner

Then later Marin’s worker extension updates that `state` via:

- [`TPUModelRunner._sync_weights(...)`](/Users/ahmed/code/vllm_tpu_multi/tpu-inference/tpu_inference/runner/tpu_runner.py:1771)

So for Marin’s fast-loading path, the important thing is:

- the model/state structure is correct
- the state tree has the right leaves and sharding metadata

It does **not** need real random-initialized weights first.

## Why I do not recommend changing Llama to `LoadableWithIterator`

That would be more invasive because it would push model-specific loading logic
into the model implementation itself.

The plan above changes one common loader seam instead:

- smaller surface area
- easier to keep opt-in
- easier to review

## Marin-side change after upstreaming

Once `tpu-inference` has the new flag, Marin can delete the monkeypatch path
and simply pass:

```python
engine_kwargs["model_loader_extra_config"] = {
    "tpu_abstract_dummy_bootstrap": True,
}
```

This is already supported by Marin’s vLLM argument plumbing:

- [`vllm_async.py:276`](/Users/ahmed/code/marin/lib/marin/src/marin/inference/vllm_async.py:276)
- [`vllm_server.py:465`](/Users/ahmed/code/marin/lib/marin/src/marin/inference/vllm_server.py:465)

## Tests to add in `tpu-inference`

### Unit test

Add a focused test in `tpu-inference/tests/...` that verifies:

- with `load_format="dummy"` and
  `model_loader_extra_config={"tpu_abstract_dummy_bootstrap": True}`
- supported Llama path returns quickly without entering the concrete random-init
  branch

The simplest assertion is behavioral:

- returned model state leaves are abstract / shape-only before real weights

### Fallback test

Verify the existing behavior is unchanged when:

- the flag is absent
- quantization is enabled
- Qwix-on-abstract-model is enabled
- the model is already `LoadableWithIterator`

## Honest limitation

If the goal is only:

- “make model loading fast”

then this plan is the right minimal source change.

If the goal is:

- “make the async-native server become ready fast”

then this plan is necessary but not sufficient.

The experiments already showed the next bottleneck is TPU KV-cache allocation,
which happens after `load_model()` returns:

- [`TPUModelRunner.initialize_kv_cache(...)`](/Users/ahmed/code/vllm_tpu_multi/tpu-inference/tpu_inference/runner/tpu_runner.py:582)
- [`KVCacheManager.initialize_kv_cache(...)`](/Users/ahmed/code/vllm_tpu_multi/tpu-inference/tpu_inference/runner/kv_cache_manager.py:279)

So this plan is the cleanest minimal fix for the **model-loading** layer, not
the full **server-readiness** problem.

## Recommended execution order

1. Upstream the `model_loader_extra_config["tpu_abstract_dummy_bootstrap"]`
   path in `tpu-inference`
2. Remove Marin’s runtime monkeypatch for model bootstrap
3. Re-run async-native startup
4. Then handle KV-cache bootstrap as a separate dependency change
