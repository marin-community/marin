# Debugging log for vllm-async-startup

Investigate why Marin's async-native vLLM startup on TPU stalls for 15-25+ minutes before the OpenAI server becomes ready.

## Initial status

Remote Iris runs of Marin's new async-native backend showed a long startup gap during async engine creation.
Marin-side logging already confirmed:

- async-native startup reaches `build_async_engine_client_from_engine_args(...)`,
- `enforce_eager=True` is being passed,
- `SKIP_JAX_PRECOMPILE=1` did not materially reduce the delay.

The remaining question is where inside the installed `vllm-tpu` / `tpu-inference` worker startup path time is actually being spent.

## Hypothesis 1

The delay is inside TPU worker startup, not in Marin's frontend thread.

Likely candidates:

- TPU worker `init_device`
- TPU worker `load_model`
- dummy model construction in `tpu_runner.get_model(...)`
- available-memory profiling
- KV-cache init
- warmup / compile after KV-cache init

## Changes to make

Instrument the installed worker/runtime path from Marin without modifying the upstream package source:

- enable a dedicated env var from `lib/marin/src/marin/inference/vllm_async.py`
- use the existing `WorkerExtension` import hook in `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py`
- monkeypatch timing logs onto:
  - `WorkerWrapperBase.init_worker`
  - `TPUWorker.init_device`
  - `TPUWorker.load_model`
  - `TPUWorker.determine_available_memory`
  - `TPUWorker.initialize_from_config`
  - `TPUWorker.compile_or_warm_up_model`
  - `TPUModelRunner.load_model`
  - `TPUModelRunner.initialize_kv_cache`
  - `TPUModelRunner.capture_model`
  - `tpu_runner.get_model`

## Future Work

- [ ] If `tpu_runner.get_model` is still too coarse, add a second pass of instrumentation around dummy-model construction internals.
- [ ] Add an opt-in way to enable the same startup timing for non-native RL async runs if needed.
- [ ] Remove or narrow the instrumentation once the root cause is identified.

## Results

Implemented Marin-side startup timing instrumentation behind `MARIN_VLLM_STARTUP_TIMING`.
Async-native startup now enables that env var by default, and frontend startup logs now print both:

- `SKIP_JAX_PRECOMPILE`
- `MARIN_VLLM_STARTUP_TIMING`

This should make the next remote run show step-level worker timing without changing the packaged `vllm-tpu` dependency.

## Hypothesis 2

The timing hooks are executing in the engine subprocess, but `logger.info()` in that subprocess is not wired to the stdout/stderr stream Iris captures.

That would explain why:

- frontend startup diagnostics appear,
- vLLM's own `(EngineCore_DP0 pid=...)` lines appear,
- Marin's `[marin-vllm-startup] ...` timing lines do not appear at all.

## Changes to make

Change the worker-side timing emission path in `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py`:

- replace `logger.info()` for timing lines with direct `stderr` writes via `print(..., file=sys.stderr, flush=True)`,
- keep the same `[marin-vllm-startup]` prefix so the existing Iris log filters still work,
- add a regression test proving `_startup_log(...)` writes to stderr.

## Results

Updated the subprocess timing hook to emit directly to `stderr` instead of the Python logger.
That keeps the instrumentation in the same worker methods but changes the transport to something Iris actually captures.

The next remote rerun should now show the startup timing lines if the worker extension import path is active.

## Hypothesis 3

The stall is after `WorkerWrapperBase.init_worker` returns, but before the wrapped TPU worker methods emit.

That suggests one of two things:

- the next call is on the wrapper layer (`WorkerWrapperBase.init_device` / delegated `load_model`) rather than directly on `TPUWorker`,
- or the current worker class in this native async configuration is not the TPU class I originally assumed.

The `v4` run also exposed a secondary bug in fallback behavior:

- the original engine subprocess stayed alive holding the TPU lock after startup timeout,
- the subprocess fallback then tried to start another TPU-backed engine,
- that second engine failed immediately because libtpu was still held by the first process.

## Changes to make

Add wrapper-level and post-executor timing:

- instrument `WorkerWrapperBase.init_device`
- add explicit timed wrapper delegate methods for:
  - `load_model`
  - `get_kv_cache_spec`
  - `determine_available_memory`
  - `compile_or_warm_up_model`
- log the actual worker class and configured executor backend at the end of `init_worker`
- instrument `EngineCore._initialize_kv_caches`
- instrument `current_platform.update_block_size_for_backend`

These hooks are late enough in startup to affect the current engine invocation because they are installed during `init_worker`, before `init_device` and KV-cache setup run.

## Results

Added the next layer of Marin-side startup timing at the worker-wrapper boundary and the first EngineCore KV-cache method.

The next rerun should now tell us which of these is true:

- we hang inside `WorkerWrapperBase.init_device`
- we hang inside delegated `WorkerWrapperBase.load_model`
- we complete executor init and then hang in `EngineCore._initialize_kv_caches`
- or we still get no deeper lines, which would mean the stall is in a path even earlier than the late-installed wrapper hooks can see

## Hypothesis 4

The worker-extension hook is simply too late to instrument the executor/core boundary that is hanging.

Because it is imported during `init_worker`, it can only patch methods that execute after that import point.
If the problematic boundary is in executor orchestration immediately after `init_worker`, or in the engine process before later worker methods are entered, then the only reliable Marin-side option is to install monkeypatches before vLLM forks the engine subprocess.

## Changes to make

Install startup timing from `lib/marin/src/marin/inference/vllm_async.py` before `build_async_engine_client_from_engine_args(...)`:

- wrap `EngineCoreProc.__init__`
- wrap `EngineCore._initialize_kv_caches`
- wrap `UniProcExecutor._distributed_args`
- replace `UniProcExecutor._init_executor` with a timed version that emits:
  - before/after `driver_worker.init_worker(...)`
  - before/after `driver_worker.init_device()`
  - before/after `driver_worker.load_model()`
  - before/after `current_platform.update_block_size_for_backend(...)`

Also log the effective `VLLM_WORKER_MULTIPROC_METHOD`, and set its async-native default to `"fork"` so the engine subprocess inherits the monkeypatches on TPU.

## Results

Implemented the pre-fork instrumentation pass in `vllm_async.py`.
This moves the most important startup timing earlier than the worker-extension import path and should let the next remote run expose the exact executor/core boundary that is hanging.

## Hypothesis 5

The first pre-fork instrumentation attempt may itself fail if it reimplements upstream executor logic against the wrong package API.

That risk is real because:

- Marin does not vendor vLLM,
- the cluster runs the packaged `vllm-tpu` build,
- and the sibling source checkout used for code reading is not guaranteed to match constructor signatures exactly.

## Changes to make

After `v6`, remove the custom body replacement for `UniProcExecutor._init_executor` and replace it with safer instrumentation:

- keep pure timing wrappers on the original upstream methods
- add timed delegate methods on `WorkerWrapperBase` for methods normally reached through `__getattr__`

Concretely:

- wrap `UniProcExecutor._init_executor` without changing its implementation
- wrap `UniProcExecutor._distributed_args`
- wrap `EngineCoreProc.__init__`
- wrap `EngineCore._initialize_kv_caches`
- install delegate timing for:
  - `WorkerWrapperBase.load_model`
  - `WorkerWrapperBase.determine_available_memory`
  - `WorkerWrapperBase.compile_or_warm_up_model`

## Results

The `v6` traceback showed the exact instrumentation bug:

- `TypeError: WorkerWrapperBase.__init__() missing 1 required positional argument: 'vllm_config'`

That came from my debug rewrite of `UniProcExecutor._init_executor`, not from upstream vLLM itself.

I removed that rewrite and converted the early instrumentation to wrapper-only hooks so the next run preserves upstream control flow while still exposing the startup boundary timings we need.

## Hypothesis 6

The real TPU startup bottleneck is inside the model load path, not in the executor/core boundary before worker startup.

Evidence from `v7`:

- `UniProcExecutor._init_executor` starts
- `WorkerWrapperBase.init_worker` completes quickly
- `WorkerWrapperBase.init_device` completes in about 25 seconds
- the engine then enters `WorkerWrapperBase.load_model`

That means the earlier “hang before `init_device`” theory was an artifact of instrumentation timing, not the real startup boundary.

## Changes to make

Use `v7` as the pivot point:

- keep the wrapper/executor timing in place
- if `load_model` remains the dominant stall, add the next layer of instrumentation inside:
  - TPU worker `load_model`
  - TPU model runner `load_model`
  - model loader internals reached from that path

The focus should now move down from engine-core orchestration to model-loading internals.

## Results

`v7` is the first trustworthy run that shows:

- worker construction is not the long pole
- device init is not the long pole
- the long-running phase begins at `WorkerWrapperBase.load_model`

That is the strongest narrowing so far.

## Hypothesis 7

The async-native long pole is inside the `MODEL_IMPL_TYPE="vllm"` loader path beneath `tpu_runner.get_model`.

At that point the likely heavy substeps are:

- `model_loader.get_vllm_model`
- `VllmModelWrapper.load_weights`
- `VllmModelWrapper.jit_step_func`
- `VllmModelWrapper.jit_compute_logits_func`

This distinction matters because it separates:

- model/materialization cost from
- JAX compilation cost for the wrapped vLLM model

## Changes to make

Extend Marin-side startup instrumentation in both:

- `lib/marin/src/marin/inference/vllm_async.py`
- `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py`

to cover:

- `model_loader.get_model`
- `model_loader.get_vllm_model`
- `model_loader.get_flax_model`
- `VllmModelWrapper.load_weights`
- `VllmModelWrapper.jit_step_func`
- `VllmModelWrapper.jit_compute_logits_func`

## Results

Added the deeper model-loader timing hooks.

The next remote async-native run should answer one of these:

- we stall in `VllmModelWrapper.load_weights`, which would point to dummy-model construction / CPU model load / host-to-TPU sharding
- we stall in `VllmModelWrapper.jit_step_func` or `VllmModelWrapper.jit_compute_logits_func`, which would point to JAX compilation of the wrapped vLLM step functions
- or we stall before those methods, which would identify another setup boundary inside `model_loader.get_vllm_model`

## Hypothesis 8

If `VllmModelWrapper.load_weights` is the long pole, the next likely split is:

- upstream vLLM model construction
- incremental TPU-aware weight loading
- post-load TPU sharding of the wrapped model

Those are distinct enough that they should be instrumented before the next rerun rather than waiting for another code-reading pass.

## Changes to make

Extend Marin-side startup instrumentation in both:

- `lib/marin/src/marin/inference/vllm_async.py`
- `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py`

to cover:

- `vllm.model_executor.model_loader.get_model`
- `vllm.model_executor.model_loader.utils.initialize_model`
- `vllm.model_executor.model_loader.utils.process_weights_after_loading`
- `IncrementalModelLoader.load_model`
- `RunaiIncrementalModelLoader.load_model`
- `cleanup_sharding.shard_model_to_tpu`
- `cleanup_sharding._shard_module_to_tpu`

## Results

Added the next, coarser `load_weights`-internal boundaries locally.

This will not affect the already-running `v9` job, but it means the next remote run can immediately distinguish:

- model construction cost
- incremental loader cost
- TPU sharding cost

## Hypothesis 9

The installed cluster package may not expose every source-checkout module path I used for optional instrumentation.

That risk materialized on `v10`:

- the `cleanup_sharding` module path was not importable
- and because that optional import sat inside one large `try` block, it suppressed the entire deeper TPU load-model instrumentation bundle

## Changes to make

Refactor the optional startup instrumentation imports in:

- `lib/marin/src/marin/inference/vllm_async.py`
- `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py`

so they degrade independently:

- core TPU load-model hooks in one block
- extended model-loader hooks in a second block
- cleanup-sharding hooks in a third block

## Results

Implemented the import-granularity fix and validated it locally.

That should allow the next run to keep the highest-value deeper hooks even if `cleanup_sharding` is absent in the installed package.

## Hypothesis 10

The installed package mismatch is more specific than just `cleanup_sharding`.

`v11` showed:

- `tpu_inference.models.vllm.vllm_model_loader` is also missing

but that should not suppress upstream `vllm.model_executor.model_loader` timing, which is still valuable and likely present.

## Changes to make

Split the model-loader instrumentation one level further:

- upstream `vllm.model_executor.model_loader` and `utils` in one optional block
- TPU-specific incremental-loader classes in a separate optional block

## Results

Implemented that split and validated it locally.

So the next rerun after `v11` can at least distinguish:

- entry into `VllmModelWrapper.load_weights`
- handoff into upstream `vllm_model_loader.get_model`
- and `initialize_model` / `process_weights_after_loading`

even if the TPU package still lacks the incremental-loader module path from the source checkout.

## Hypothesis 11

`VllmModelWrapper.load_weights` may be calling module-local aliases imported into `tpu_inference.models.vllm.vllm_model_wrapper`, not the upstream module functions I wrapped separately.

That would explain why `v12` shows:

- `START VllmModelWrapper.load_weights`

but never shows:

- `START vllm_model_loader.get_model`

## Changes to make

Instrument the module-local call sites directly on `tpu_inference.models.vllm.vllm_model_wrapper`:

- `vllm_get_model`
- `shard_model_to_tpu`
- `load_lora_model`

## Results

Added alias-level instrumentation to both Marin startup patch sites and validated it locally.

The next rerun after `v12` should tell us whether the stall is:

- before the wrapper reaches `vllm_get_model`
- inside the `vllm_get_model` handoff
- or later in `shard_model_to_tpu`

## Hypothesis 12

The remaining blind spot under `vllm_get_model` is caused by upstream loader aliasing, not lack of instrumentation depth in principle.

For `load_format="dummy"`:

- `BaseModelLoader.load_model` uses module-local imports of `initialize_model` and `process_weights_after_loading`
- `DummyModelLoader.load_weights` uses the module-local alias `initialize_dummy_weights`

So wrapping only the canonical upstream module functions can still miss the real call path.

## Changes to make

Extend both Marin startup patch sites to wrap the exact upstream aliases used in the dummy-loader path:

- `BaseModelLoader.load_model`
- `DummyModelLoader.load_weights`
- `vllm.model_executor.model_loader.base_loader.initialize_model`
- `vllm.model_executor.model_loader.base_loader.process_weights_after_loading`
- `vllm.model_executor.model_loader.dummy_loader.initialize_dummy_weights`
- `vllm.model_executor.model_loader.weight_utils.initialize_dummy_weights`

## Results

Added the dummy-loader alias instrumentation and validated it locally.

The next rerun should now distinguish:

- model construction before dummy init
- dummy weight initialization itself
- post-load processing after dummy init
