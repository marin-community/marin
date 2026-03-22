# Fast Startup Execution Plan for vLLM on TPU in Marin

## Purpose

This document is the implementation handoff for getting **fast model loading**
on TPU for both:

- direct `generate()` / offline batch usage
- async-native vLLM server startup

It is more detailed than the earlier design doc. It names the exact bridge
contract, the exact fork patches, the Marin integration changes, the packaging
changes required by the forking policy, and the validation matrix.

## Problem

Marin already has the right checkpoint-loading pipeline:

- stage local HF metadata
- start vLLM with `load_format="dummy"`
- stream safetensor shards from object storage
- reshape / map tensors
- inject them into the live engine

That pipeline exists in both Marin paths today:

- **sync / direct `generate()`** via
  [`start_inprocess_vllm_server()` in `vllm_inprocess.py:215`](/Users/ahmed/code/marin/lib/marin/src/marin/inference/vllm_inprocess.py#L215)
  and
  [`_load_and_inject_streaming()` in `vllm_inprocess.py:380`](/Users/ahmed/code/marin/lib/marin/src/marin/inference/vllm_inprocess.py#L380)
- **async-native server** via
  [`_run_async_server()` in `vllm_async.py:152`](/Users/ahmed/code/marin/lib/marin/src/marin/inference/vllm_async.py#L152)
  and
  [`_load_and_inject_streaming_async()` in `vllm_async.py:294`](/Users/ahmed/code/marin/lib/marin/src/marin/inference/vllm_async.py#L294)

Both paths also already have working weight application:

- sync path: `sync_weights()` inside vLLM `LLM`
- async path:
  [`WorkerExtension.update_weight()` in `worker.py:653`](/Users/ahmed/code/marin/lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py#L653)
  calling
  [`model_runner._sync_weights(...)` in `worker.py:662`](/Users/ahmed/code/marin/lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py#L662)

The startup problems are not in Marin’s streaming pipeline. They are in TPU
bootstrap work that happens **before** real weights are injected.

The experiments established three separate blockers:

1. **Model bootstrap**
   In TPU `flax_nnx`,
   [`_get_nnx_model()` in `model_loader.py:99`](/Users/ahmed/code/tpu-inference/tpu_inference/models/common/model_loader.py#L99)
   sends `load_format="dummy"` plus non-`LoadableWithIterator` models through a
   concrete random-init path at
   [`model_loader.py:141`](/Users/ahmed/code/tpu-inference/tpu_inference/models/common/model_loader.py#L141).
   This is the slow path that hurt Llama.

2. **Sampling RNG on abstract bootstrap state**
   TPU runner still does
   [`nnx.Rngs(jax.random.key(seed)).params()` in `tpu_runner.py:580`](/Users/ahmed/code/tpu-inference/tpu_inference/runner/tpu_runner.py#L580).
   On abstract bootstrap state this was another startup stall.

3. **KV cache initialization**
   Async-native startup still blocks in
   [`KVCacheManager.initialize_kv_cache()` in `kv_cache_manager.py:279`](/Users/ahmed/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L279),
   specifically the first
   [`create_kv_caches(...)` call at `kv_cache_manager.py:315`](/Users/ahmed/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L315).

There is also a routing problem:

- [`_VLLM_PREFERRED_ARCHITECTURES` in `model_loader.py:48`](/Users/ahmed/code/tpu-inference/tpu_inference/models/common/model_loader.py#L48)
  forces `Qwen3MoeForCausalLM` to the TPU `"vllm"` wrapper path.
- But TPU already has a JAX implementation at
  [`qwen3_moe.py:298`](/Users/ahmed/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L298).

## Current Fork Baseline

This plan is grounded in the current clean fork at:

- `/Users/ahmed/code/tpu-inference`
- `HEAD = 7f0436b5`

Current fork facts:

- `LlamaForCausalLM` is a plain `nnx.Module` at
  [`llama3.py:351`](/Users/ahmed/code/tpu-inference/tpu_inference/models/jax/llama3.py#L351)
- `Qwen3ForCausalLM` implements `LoadableWithIterator` at
  [`qwen3.py:316`](/Users/ahmed/code/tpu-inference/tpu_inference/models/jax/qwen3.py#L316)
- `Qwen3MoeForCausalLM` implements `LoadableWithIterator` at
  [`qwen3_moe.py:298`](/Users/ahmed/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L298)
- `Qwen3MoeForCausalLM` is still forced to the TPU `"vllm"` path by
  [`_VLLM_PREFERRED_ARCHITECTURES` in `model_loader.py:48`](/Users/ahmed/code/tpu-inference/tpu_inference/models/common/model_loader.py#L48)
- runner sampling RNG still initializes at
  [`tpu_runner.py:580`](/Users/ahmed/code/tpu-inference/tpu_inference/runner/tpu_runner.py#L580)
- KV bootstrap still allocates in
  [`kv_cache_manager.py:315`](/Users/ahmed/code/tpu-inference/tpu_inference/runner/kv_cache_manager.py#L315)

That means the work splits into:

- new loader/bootstrap logic for Llama-class non-iterator models
- routing fixes for Qwen3-MoE
- runner RNG fix
- a practical KV startup policy

## Forking Policy Compliance

This plan is compliant with
[`forking-policy.md`](/Users/ahmed/code/marin/docs/dev-guide/forking-policy.md).

Required conditions:

1. The code lives in `marin-community/tpu-inference`, not in the Marin monorepo.
2. The fork builds wheels automatically on push to `main`.
3. Marin pins the fork explicitly instead of silently consuming PyPI
   `tpu-inference` transitively through `vllm-tpu`.

Current gap:

- [`lib/marin/pyproject.toml:172`](/Users/ahmed/code/marin/lib/marin/pyproject.toml#L172)
  pins only `vllm-tpu==0.13.2.post6`
- [`uv.lock`](/Users/ahmed/code/marin/uv.lock) still resolves
  `tpu-inference==0.13.2.post6` from PyPI

Compliant integration target in Marin:

```toml
[tool.uv.sources]
tpu-inference = { git = "https://github.com/marin-community/tpu-inference.git", rev = "<commit>" }
```

A wheel URL pinned to a fork-built artifact is also acceptable.

## Decision Summary

This is the recommended implementation shape.

1. **Keep Marin’s current shard streaming and weight injection.**
   Do not redesign the checkpoint pipeline.

2. **Make `tpu-inference` own TPU bootstrap behavior.**
   The fork should absorb the current Marin monkeypatch logic.

3. **Use a narrow bridge key in `model_loader_extra_config` for loader/routing/RNG metadata.**
   Recommended bridge key:
   `model_loader_extra_config["marin_tpu_bootstrap"]`

4. **Do not invent a second KV sizing mechanism inside `tpu-inference`.**
   Use existing upstream `vllm` `num_gpu_blocks_override` for the immediate
   fast-startup path.

5. **Split deliverables honestly.**
   - **Deliverable A**: fork-only changes plus Marin pinning. This gives fast
     model loading and fast async startup if we are willing to run with a small
     serving KV cache.
   - **Deliverable B**: optional follow-up for “fast startup, then promote to
     full serving capacity.” This requires coordinated `vllm-tpu` work because
     scheduler/worker agreement on two KV sizes is not a pure `tpu-inference`
     concern.

## Exact Bridge Contract

For the fork-only bridge, use `model_loader_extra_config` only for bootstrap
metadata that `tpu-inference` actually owns.

Recommended shape:

```python
engine_kwargs = {
    "load_format": "dummy",
    "model_loader_extra_config": {
        "marin_tpu_bootstrap": {
            "model_mode": "abstract_dummy",
            "routing_mode": "prefer_flax_nnx",
            "sampling_mode": "direct_key",
        }
    },
    # Existing vllm cache-config mechanism.
    "num_gpu_blocks_override": 128,
}
```

Notes:

- `model_loader_extra_config` is the bridge because it is already plumbed by
  vLLM and already passed by Marin in
  [`vllm_async.py:280`](/Users/ahmed/code/marin/lib/marin/src/marin/inference/vllm_async.py#L280)
- `num_gpu_blocks_override` should remain owned by `vllm`, not copied into a
  new `tpu-inference`-only config
- the bridge key is intentionally namespaced (`marin_tpu_bootstrap`) so the
  fork can evolve without colliding with unrelated upstream keys

Recommended exact enums for the bridge:

- `model_mode`: `"default" | "abstract_dummy"`
- `routing_mode`: `"default" | "prefer_flax_nnx"`
- `sampling_mode`: `"default" | "direct_key"`

Do **not** put `kv_bootstrap_blocks` into this dict for the first patch.
`num_gpu_blocks_override` already exists in upstream `vllm`:

- [`cache.py:59`](/Users/ahmed/code/vllm_tpu_multi/vllm/vllm/config/cache.py#L59)
- [`arg_utils.py:513`](/Users/ahmed/code/vllm_tpu_multi/vllm/vllm/engine/arg_utils.py#L513)
- [`kv_cache_utils.py:820`](/Users/ahmed/code/vllm_tpu_multi/vllm/vllm/v1/core/kv_cache_utils.py#L820)

## Deliverable A: Fork-Only Implementation

This deliverable is implementable now with only:

- changes in `marin-community/tpu-inference`
- Marin dependency pinning to that fork
- no required `vllm-tpu` source changes

It delivers:

- fast model loading for direct `generate()`
- fast async-native startup **if** the service uses a smaller KV cache via
  `num_gpu_blocks_override`

In other words, Deliverable A is:

- fast startup
- correct weight loading
- native async serving
- with intentionally smaller serving capacity at startup

### `tpu_inference/models/common/model_loader.py`

This is the main patch file.

Add these helpers near the top of the file:

```python
@dataclass(frozen=True)
class MarinTpuBootstrapConfig:
    model_mode: str = "default"
    routing_mode: str = "default"
    sampling_mode: str = "default"


def _marin_bootstrap_config(vllm_config: VllmConfig) -> MarinTpuBootstrapConfig:
    ...


def _should_use_abstract_dummy_bootstrap(model_class: Any, vllm_config: VllmConfig) -> bool:
    ...
```

Exact behavior:

- parse `vllm_config.load_config.model_loader_extra_config`
- read the `marin_tpu_bootstrap` sub-dict if present
- return typed defaults when absent
- reject malformed values with `ValueError`; do not silently ignore bad config

Then change [`_get_nnx_model()` at `model_loader.py:99`](/Users/ahmed/code/tpu-inference/tpu_inference/models/common/model_loader.py#L99):

1. leave the existing iterator-based branch intact
2. split the current slow dummy branch at
   [`model_loader.py:141`](/Users/ahmed/code/tpu-inference/tpu_inference/models/common/model_loader.py#L141)
   into:
   - current concrete random-init branch
   - new abstract dummy branch for non-iterator models
3. gate the new branch with `_should_use_abstract_dummy_bootstrap(...)`

The new abstract dummy branch should:

- call `nnx.eval_shape(create_abstract_model)`
- extract `state = nnx.state(model)`
- rebuild a valid model/state without concrete random dummy weights
- keep existing `create_jit_model(...)` semantics
- avoid Qwix / quantization modes in the first patch

Important scope decision:

- initial target is `LlamaForCausalLM`
- do **not** broaden this patch to every model immediately
- `Qwen3ForCausalLM` and `Qwen3MoeForCausalLM` already have iterator-based JAX
  loading; they mainly need routing, not a new bootstrap branch

Then change [`resolve_model_architecture()` at `model_loader.py:429`](/Users/ahmed/code/tpu-inference/tpu_inference/models/common/model_loader.py#L429):

- add a helper such as `_prefer_flax_nnx_for_bootstrap(vllm_config)`
- if `routing_mode == "prefer_flax_nnx"` and `_get_model_architecture(...)`
  succeeds, return `"flax_nnx"` even for architectures currently listed in
  `_VLLM_PREFERRED_ARCHITECTURES`
- keep the existing `runai_streamer` capability check intact

This is what enables Qwen3-MoE to use its existing JAX path.

### `tpu_inference/runner/tpu_runner.py`

This file needs the bootstrap-safe sampling-key patch.

Add:

```python
def _sampling_key_for_bootstrap(self) -> jax.Array:
    cfg = _marin_bootstrap_config(self.vllm_config)
    key = jax.random.key(self.model_config.seed)
    if cfg.sampling_mode == "direct_key":
        return key
    return nnx.Rngs(key).params()
```

Then change [`TPUModelRunner.load_model()` at `tpu_runner.py:558`](/Users/ahmed/code/tpu-inference/tpu_inference/runner/tpu_runner.py#L558):

- store `self.marin_tpu_bootstrap_config`
- replace the direct assignment at
  [`tpu_runner.py:580`](/Users/ahmed/code/tpu-inference/tpu_inference/runner/tpu_runner.py#L580)
  with `_sampling_key_for_bootstrap()`
- leave the rest of `load_model()` unchanged in the first patch

This patch should be minimal. Do not combine it with unrelated cleanup.

### `tpu_inference/runner/kv_cache_manager.py`

**No required functional patch for Deliverable A.**

Reason:

- the immediate fast-startup path can already use upstream
  `num_gpu_blocks_override`
- `KVCacheManager.initialize_kv_cache()` already consumes the resulting
  `KVCacheConfig`
- we do not need to invent a second KV sizing mechanism in the fork just to get
  startup unstuck

What this means in practice:

- for the first working end-to-end path, async-native startup should set
  `num_gpu_blocks_override` explicitly from Marin
- `tpu-inference` does not need to compute or clamp block counts itself

The only acceptable first-pass changes here are:

- optional logging clarifying allocated block counts
- optional comments/docstrings documenting that a smaller upstream
  `KVCacheConfig` is the supported startup path

Do **not** start with allocator surgery in
[`kv_cache.py:73`](/Users/ahmed/code/tpu-inference/tpu_inference/runner/kv_cache.py#L73).
That is a fallback only if a deliberately small `num_gpu_blocks_override` is
still too slow.

## Deliverable B: Optional Full-Capacity Startup Follow-up

This is the production enhancement if you need:

- fast startup
- then promotion to a larger serving KV cache before or just after traffic

This is **not** a pure `tpu-inference` change, because scheduler and worker must
agree on the cache shape and capacity.

That likely requires coordinated `vllm-tpu` work:

- a first-class bootstrap cache config
- a first-class serving cache config
- engine lifecycle that starts with the bootstrap config and later promotes to
  the serving config

Recommended shape for that later work:

- keep `model_loader_extra_config["marin_tpu_bootstrap"]` only for
  loader/routing/sampling bridge metadata
- add real cache-config fields in `vllm-tpu` / `vllm`, for example:
  - `bootstrap_num_gpu_blocks`
  - `serving_num_gpu_blocks`
- let `tpu-inference` simply consume those resulting configs and provide a clean
  `promote_serving_kv_cache()` path

This follow-up is what gets you “fast startup, then full serving capacity”
without overloading `model_loader_extra_config` forever.

## Detailed Marin Changes

### 1. Centralize engine kwargs construction

Add one Marin helper that returns the TPU fast-startup kwargs for both sync and
async paths.

Recommended shape:

- new helper module under `lib/marin/src/marin/inference/`
- no `*_utils.py`
- for example: `vllm_bootstrap_config.py`

The helper should return:

- `load_format="dummy"`
- `model_loader_extra_config["marin_tpu_bootstrap"]`
- optional `num_gpu_blocks_override`

Then use it in:

- [`vllm_inprocess.py`](/Users/ahmed/code/marin/lib/marin/src/marin/inference/vllm_inprocess.py)
- [`vllm_async.py`](/Users/ahmed/code/marin/lib/marin/src/marin/inference/vllm_async.py)

### 2. Remove the bootstrap monkeypatch path once the fork is pinned

After the forked `tpu-inference` is being resolved by Marin:

- delete or retire
  [`vllm_tpu_bootstrap_patch.py`](/Users/ahmed/code/marin/lib/marin/src/marin/inference/vllm_tpu_bootstrap_patch.py)
- stop installing that monkeypatch from
  [`vllm_async.py:156`](/Users/ahmed/code/marin/lib/marin/src/marin/inference/vllm_async.py#L156)

Do not remove the startup instrumentation until the fork-backed path is
validated on the cluster.

### 3. Pin the fork in Marin

Once the fork has CI-built artifacts:

- update root [`pyproject.toml`](/Users/ahmed/code/marin/pyproject.toml)
  `tool.uv.sources`
- pin `tpu-inference` to a fork revision or wheel URL
- regenerate [`uv.lock`](/Users/ahmed/code/marin/uv.lock)
- verify `uv sync` resolves the forked dependency, not PyPI

This is required by the forking policy and by correctness.

### 4. Preserve request-path architecture direction

Do not revert to the old queue-based in-process server in
[`vllm_inprocess.py:270`](/Users/ahmed/code/marin/lib/marin/src/marin/inference/vllm_inprocess.py#L270).

Serving direction remains:

- direct batch usage can still use sync `generate()`
- real serving should remain native async vLLM

## Exact Test Plan

### `tpu-inference` tests

Extend existing test files rather than creating parallel suites.

#### `tests/e2e/test_model_loader.py`

Add tests for:

- parsing `model_loader_extra_config["marin_tpu_bootstrap"]`
- `resolve_model_architecture()` returning `flax_nnx` for `Qwen3MoeForCausalLM`
  when `routing_mode == "prefer_flax_nnx"`
- `resolve_model_architecture()` keeping current behavior when bridge config is
  absent
- Llama `load_format="dummy"` taking the new abstract branch when
  `model_mode == "abstract_dummy"`
- malformed bridge config raising a real error

#### `tests/runner/test_tpu_runner.py`

Add tests for:

- `sampling_mode == "direct_key"` uses raw `jax.random.key(...)`
- default behavior still uses `nnx.Rngs(...).params()`
- `load_model()` records bootstrap config on the runner

#### `tests/runner/test_kv_cache_manager.py`

No new functional assertions are required for Deliverable A beyond existing
coverage, because KV sizing remains upstream-owned.

Only add tests here if we decide to add:

- explicit logging helpers
- a new promotion/reinitialize API in Deliverable B

#### `tests/e2e/test_rl_integration.py`

Use this file later for Deliverable B if we add explicit promotion/reinit
lifecycle work. It already exercises delete/reinitialize KV cache behavior.

### Marin tests

Extend existing Marin tests that already cover backend startup behavior:

- [test_vllm_inprocess_backend.py](/Users/ahmed/code/marin/tests/vllm/test_vllm_inprocess_backend.py)

Add coverage for:

- engine kwargs helper emits `marin_tpu_bootstrap` correctly
- sync path uses the helper
- async path uses the helper
- optional `num_gpu_blocks_override` plumbing for async startup

## Cluster Validation Matrix

This is the order to validate after the fork is pinned in Marin.

### Llama 8B

1. **Direct batch / sync path**
   - `load_format="dummy"`
   - `marin_tpu_bootstrap.model_mode="abstract_dummy"`
   - no `num_gpu_blocks_override`
   - success criterion: model bootstrap is fast and outputs are correct

2. **Async-native server, small KV**
   - same bootstrap config
   - set `num_gpu_blocks_override` explicitly to a small value
   - success criterion: server becomes ready quickly enough to serve requests

3. **Async-native concurrency sanity**
   - Harbor-style or stress-test concurrency with the chosen small KV cache
   - success criterion: requests complete correctly and startup remains bounded

### Qwen3-MoE

1. **Routing validation**
   - `routing_mode="prefer_flax_nnx"`
   - verify `resolve_model_architecture()` picks `flax_nnx`

2. **Direct batch / sync path**
   - verify JAX iterator-based load path works under the same bridge contract

3. **Async-native server, small KV**
   - same bootstrap config
   - explicit `num_gpu_blocks_override`
   - success criterion: startup path avoids the TPU `"vllm"` wrapper and server
     becomes ready

## Recommended Implementation Order

1. Patch `tpu_inference/models/common/model_loader.py`
   - bridge config parse
   - Llama abstract dummy branch
   - Qwen3-MoE routing override
2. Patch `tpu_inference/runner/tpu_runner.py`
   - bootstrap-safe sampling key
3. Add/adjust `tpu-inference` tests
4. Set up fork CI so `main` builds versioned wheels
5. Pin the fork in Marin via `tool.uv.sources`
6. Add Marin engine-kwargs helper and route both sync and async paths through it
7. Re-run Llama sync and async validation
8. Re-run Qwen3-MoE sync and async validation
9. Remove the Marin bootstrap monkeypatch only after fork-backed cluster runs are green

## Explicit Non-Goals For This Patch Set

- redesigning vLLM request scheduling
- changing Marin’s shard streaming logic
- inventing a new TPU-only KV block override mechanism
- allocator-level surgery in `kv_cache.py` as the first response
- full dual-KV bootstrap/promotion in the same patch set

## Done Criteria

Deliverable A is done when all of these are true:

- `marin-community/tpu-inference` contains the loader/routing/RNG patches
- fork CI builds wheels automatically on push to `main`
- Marin pins the fork explicitly and no longer resolves PyPI `tpu-inference`
- Llama direct `generate()` uses fast bootstrap
- Llama async-native server reaches readiness with explicit small KV sizing
- Qwen3-MoE can route to `flax_nnx` bootstrap mode and start successfully
- Marin no longer needs the TPU bootstrap monkeypatch on the critical path

Deliverable B is separate and only starts if we decide we need fast startup plus
later promotion to larger serving KV capacity.
