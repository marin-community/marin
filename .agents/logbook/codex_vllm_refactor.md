# Codex vLLM Refactor Logbook

## 2026-03-18

### Context

We now have a canonical map of vLLM in Marin in `.agents/projects/vllm_serving_analysis.md`.
The key conclusion is stable:

- fast-loading via dummy load + staged metadata + fsspec shard streaming + `sync_weights()` is the right loading path,
- the current queue-based in-process HTTP server is the wrong serving architecture,
- the replacement should use the standard async vLLM engine and standard OpenAI server stack.

### User Direction

The user explicitly wants code changes now, not more architecture spelunking or local-only verification.
The target is Option B:

- stop using the current hacky `InProcessVllmServerBackend`,
- make Harbor and future scripts use a standard async vLLM server shape,
- preserve Marin's faster, lower-RAM model loading path.

### Current Decision

Refactor toward this shape:

1. Keep the existing eligibility, bootstrap metadata staging, mapping resolution, tensor reshape, and shard-streaming logic.
2. Replace the queue-based `LLM.generate()` HTTP server with an async-engine-based server.
3. Use upstream vLLM OpenAI app initialization instead of Marin's custom `/v1/chat/completions` wrapper.
4. Reuse RL's `WorkerExtension` and RPC serialization path instead of inventing a second async weight-update mechanism.
5. Keep subprocess `vllm serve` as the native fallback backend.

### Working Hypothesis

The first implementation pass will likely stream shards on the frontend process and push them into the async engine via worker-extension RPC.
That gives us:

- standard async serving semantics,
- no all-at-once host-RAM blowup,
- a much cleaner architecture than the queue server.

There is still a performance risk that collective RPC duplicates shard payloads more than we want. If that becomes the bottleneck on cluster, the next step is to move initial shard loading behind a dedicated worker-extension method or use a proper weight-transfer engine. That is a second-order optimization, not a reason to keep the current queue server.

### Immediate Execution Plan

- create a new async native backend in Marin's inference stack,
- wire `VllmEnvironment` to prefer it in `native` mode,
- remove the queue-based serving path from the selected backend,
- update tests to assert backend selection and fallback behavior against the new class names and runtime hooks.

### Notes

A local `vllm` import is not available in this shell, so implementation must rely on repository code plus the local checkout at `/Users/ahmed/code/vllm_tpu_multi/vllm`. Remote-cluster validation will be required after the code lands.

### Update: async-native second pass

I continued fleshing out `lib/marin/src/marin/inference/vllm_async.py` instead of treating the first pass as done.

Changes in this pass:

- explicitly added `--worker-extension-cls marin.rl.environments.inference_ctx.inflight.worker.WorkerExtension` to async native engine startup,
- switched shard injection to use the public async `engine_client.collective_rpc(...)` surface instead of reaching through `engine_client.engine_core` directly,
- added process-level TPU/JAX/vLLM environment defaults inside `vllm_async.py` so native async startup gets the same cache and logging assumptions as other native paths.

### Confirmed upstream constraint

I verified against the local vLLM checkout that `AsyncLLM` does not expose a direct frontend-side path to the model runner for weight injection.
Initial fast loading must therefore go through RPC unless we add a deeper worker-side loading path later.

That means the current architecture is:

- standard async vLLM OpenAI app,
- async engine client lifecycle,
- shard streaming on the frontend,
- worker-extension RPC for per-shard weight injection.

This is the correct serving architecture even if it may not yet be the final optimal weight-transfer architecture.

### Remaining risk

The main remaining unknown is cluster performance of per-shard RPC injection for larger checkpoints.
If this is too expensive, the next optimization should move initial loading closer to the worker / engine core rather than regressing back to the queue-based `LLM.generate()` server.

### Update: startup diagnostics and TPU precompile knob

I reviewed a remote-cluster debug report that blamed `build_async_engine_client_from_engine_args(...)` itself for the long startup.
The strong version of that claim did not hold up against the local vLLM checkout:

- the upstream helper ultimately constructs `AsyncLLM`,
- `AsyncLLM` still uses the async multiprocess engine core in this fork,
- switching to `AsyncLLM.from_engine_args(...)` directly would not remove the background engine process.

What did hold up:

- the TPU fork has a real precompile gate,
- the correct env var is `SKIP_JAX_PRECOMPILE`, not `VLLM_TPU_SKIP_PRECOMPILE`.

Based on that review I made two small hardening changes in `lib/marin/src/marin/inference/vllm_async.py`:

1. Set `SKIP_JAX_PRECOMPILE=1` by default in async-native startup to bias toward fast startup for serving.
2. Emit explicit pre-engine startup diagnostics before `build_async_engine_client_from_engine_args(...)`:
   - bootstrap model path,
   - requested model path,
   - `tensor_parallel_size`,
   - `enforce_eager`,
   - effective `SKIP_JAX_PRECOMPILE`.

This does not resolve all remote-startup risk, but it removes guesswork from the next cluster run.

### Update: remote async-native startup still stalls

After landing the `SKIP_JAX_PRECOMPILE=1` default and startup diagnostics, the user ran the async-native 8B stress job again on Iris.
That result is important because it ruled out the simplest explanation:

- `SKIP_JAX_PRECOMPILE=1` did not materially reduce the long Stage 5 startup delay,
- so the long wait is not explained by the guarded TPU precompile path alone,
- and the effect is not fixed by `enforce_eager=True` either.

What I checked in the local vLLM / TPU code after that run:

- `build_async_engine_client_from_engine_args(...)` still flows into `AsyncLLM`; switching to direct `AsyncLLM.from_engine_args(...)` would not remove the async multiprocess engine core in this fork.
- `EngineCore` startup does more than KV-cache warmup. During construction it first creates the executor, and that executor already runs:
  - `worker.init_device()`
  - `worker.load_model()`
- only after that does `EngineCore` call its KV-cache preparation path:
  - `determine_available_memory()`
  - KV-cache config derivation
  - `initialize_from_config(...)`
  - `compile_or_warm_up_model()`

That means the remaining stall could be in any of these buckets:

1. TPU device initialization
2. Dummy-model construction inside TPU `load_model()` / `get_model(...)`
3. Memory profiling for available KV-cache budget
4. KV-cache initialization
5. Warmup / compile path after KV-cache init

The key correction to my own earlier narrowing:

- I was right that the upstream helper API was not the likely root cause.
- I was too optimistic that `SKIP_JAX_PRECOMPILE` was the main unblocker.
- The current evidence says we need deeper timing inside the V1 engine startup path itself.

### Next debugging move

The next concrete step should be targeted timing logs in the upstream vLLM / TPU code, not more Marin-level env tweaking.
The most useful instrumentation points are:

- `vllm/vllm/v1/executor/multiproc_executor.py`
- `vllm/vllm/v1/engine/core.py`
- `tpu-inference/tpu_inference/worker/tpu_worker.py`
- `tpu-inference/tpu_inference/models/common/model_loader.py`

The goal is to bracket:

- executor init,
- TPU worker `init_device`,
- TPU worker `load_model`,
- `get_model(...)`,
- available-memory profiling,
- KV-cache init,
- warmup / compile.

### Update: Marin-side worker startup timing instrumentation

I implemented the next-best thing to patching the packaged dependency directly: Marin now instruments the installed TPU worker path at runtime.

Implementation shape:

- `lib/marin/src/marin/inference/vllm_async.py`
  - now sets `MARIN_VLLM_STARTUP_TIMING=1` by default for async-native startup
  - includes that env var in the frontend startup diagnostic line

- `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py`
  - uses the existing `WorkerExtension` import path as the hook into worker processes
  - when `MARIN_VLLM_STARTUP_TIMING=1`, monkeypatches timing logs onto installed TPU worker/runtime methods

Instrumented runtime points:

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

Expected log format:

- `[marin-vllm-startup] START ...`
- `[marin-vllm-startup] END ... in <secs>`
- `[marin-vllm-startup] FAIL ... in <secs>`

This should let the next Iris run tell us whether the long delay is in:

- device init,
- dummy model creation,
- available-memory profiling,
- KV-cache init,
- or post-KV warmup.

### Rerun plan

The next remote validation should rerun the same async-native 8B stress job with a fresh job name so the new Marin-side timing logs are easy to isolate.

Recommended rerun:

- script: `experiments/inference/exp_vllm_stress_test.py`
- mode: `native`
- model: `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`
- job name: `vllm-async-stress-8b-eager-v3`

Expected high-signal log lines:

- frontend:
  - `Creating AsyncLLM engine with ... MARIN_VLLM_STARTUP_TIMING='1'`

### Update: worker timing hooks were present but invisible

The next remote result corrected an important assumption in my Marin-side instrumentation pass.

What happened:

- the worker startup timing hooks were installed in the right process path,
- but they emitted via `logger.info()`,
- and those logger handlers in the async engine subprocess were not surfacing to Iris logs.

That means the previous "zero timing output" result does **not** mean the worker extension failed to load.
It only means the transport for those timing lines was wrong.

Correction applied:

- in `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py`,
- the `[marin-vllm-startup] ...` timing lines now write directly to `stderr` with `flush=True`,
- so they should be visible to Iris even from the forked engine worker process.

I also added a regression test in `tests/vllm/test_vllm_inprocess_backend.py` asserting that `_startup_log(...)` writes to stderr.

### What this means for the next run

The next rerun should now distinguish between two cases cleanly:

1. We see `[marin-vllm-startup]` lines.
   - Then we can identify the actual slow startup phase.
2. We still see no `[marin-vllm-startup]` lines.
   - Then the issue is earlier than log transport, likely that the worker-extension import path is not active in this native async startup configuration.

### Update: `v4` run narrowed the stall to after `init_worker`

The `v4` rerun finally produced usable worker-side timing.

What it showed:

- the worker extension import path is active,
- `WorkerWrapperBase.init_worker` starts and ends quickly,
- the original engine process then sits for roughly an hour without reaching ready,
- none of the lower-level TPU worker timing hooks fired after that.

That means the previous absence of TPU worker logs was not because the extension failed to load.
It means the stall is after `init_worker`, but before or around the next wrapper/executor boundary.

The same run also surfaced a second issue:

- when async-native startup timed out,
- Marin fell back to the subprocess backend,
- but the original async engine process was still alive and still held the TPU lock,
- so the fallback subprocess immediately failed with the libtpu "already in use" error.

That fallback collision is real, but it is a follow-up issue.
The primary debugging target is still the first engine's one-hour stall.

### Correction to instrumentation strategy

My earlier TPU-only method hooks were too low in the stack.
Because the worker extension is imported during `init_worker`, the best Marin-side hooks for the next pass are methods that execute *after* `init_worker` returns in the same startup attempt.

I therefore extended the runtime timing in `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py` to include:

- `WorkerWrapperBase.init_device`
- `WorkerWrapperBase.load_model` via an explicit timed delegate method
- `WorkerWrapperBase.get_kv_cache_spec` via an explicit timed delegate method
- `WorkerWrapperBase.determine_available_memory` via an explicit timed delegate method
- `WorkerWrapperBase.compile_or_warm_up_model` via an explicit timed delegate method
- `WorkerWrapperBase.initialize_from_config`
- `EngineCore._initialize_kv_caches`
- `current_platform.update_block_size_for_backend`

I also added a startup line that prints:

- the concrete worker class actually created
- the configured executor backend

This matters because it will tell us whether the runtime is really using the TPU worker class I assumed, or a different wrapper/worker implementation in the async-native path.

### Expected signal on the next run

The next remote rerun should now separate four cases:

1. `WorkerWrapperBase.init_device` starts and hangs.
   - Then the problem is inside device init / worker delegation.
2. `WorkerWrapperBase.load_model` starts and hangs.
   - Then the problem is the first model load path, not KV-cache setup.
3. `EngineCore._initialize_kv_caches` starts and hangs.
   - Then executor init completed and the stall is in KV-cache sizing/init or warmup.
4. None of those lines appear after `init_worker`.
   - Then the stall is in an even narrower gap immediately after worker construction, and we will need an earlier hook than the worker extension can currently provide.

### Update: `v5` repeated the same signature

The next remote rerun (`v5`) did not produce any new timing beyond the `init_worker` boundary.

What held constant:

- `WorkerWrapperBase.init_worker` still completed quickly.
- None of the later worker-wrapper or TPU-worker timing hooks fired.
- The stall still persisted for 25+ minutes with no additional startup progress.

That is strong evidence that the worker-extension-based instrumentation has reached its limit.

Reason:

- those hooks are only installed when the worker extension module is imported during `init_worker`,
- but the missing logs now show the problematic boundary is earlier than the next observable calls from that late-installed hook set,
- so the next instrumentation pass needs to be installed before the engine subprocess begins its executor/core startup sequence.

### Revised next move

The highest-leverage next step is to instrument the installed vLLM executor/core classes from Marin's frontend process before async engine creation.

Target instrumentation points:

- `vllm.v1.executor.uniproc_executor.UniProcExecutor._init_executor`
- `vllm.v1.engine.core.EngineCore.__init__`
- `vllm.v1.engine.core.EngineCoreProc.__init__`

The goal is to log exact before/after boundaries for:

- executor construction
- `driver_worker.init_worker(...)`
- `driver_worker.init_device()`
- `driver_worker.load_model()`
- the handoff from `EngineCoreProc` to `EngineCore`

Why this is the logical next step:

- on single-host TPU, upstream forces `distributed_executor_backend = "uni"`,
- `AsyncLLM` still starts an engine subprocess, but that subprocess then uses `UniProcExecutor`,
- so pre-fork monkeypatching from Marin should let the engine subprocess inherit the tighter startup instrumentation.

### Operational note

If the next run still hangs, I should stop the job once it repeats the known signature instead of waiting for the full upstream timeout.

At that point:

- the job is no longer yielding new information,
- it continues holding TPU resources,
- and the timeout path is known to create a misleading fallback failure because the first hung engine still owns libtpu.

### Update: pre-fork executor/core instrumentation landed

I implemented the next instrumentation pass in `lib/marin/src/marin/inference/vllm_async.py`.

Key change:

- Marin now installs startup timing hooks from the frontend process before calling `build_async_engine_client_from_engine_args(...)`.

Why this matters:

- the engine subprocess is started via vLLM multiprocessing,
- on this TPU path the default worker multiprocessing method is `fork`,
- so early monkeypatching in Marin should be inherited by the engine subprocess before it begins the executor/core startup sequence.

What is instrumented in this pass:

- `vllm.v1.engine.core.EngineCoreProc.__init__`
- `vllm.v1.engine.core.EngineCore._initialize_kv_caches`
- `vllm.v1.executor.uniproc_executor.UniProcExecutor._distributed_args`
- `vllm.v1.executor.uniproc_executor.UniProcExecutor._init_executor`

Inside the patched `UniProcExecutor._init_executor`, Marin now emits explicit boundaries for:

- `driver_worker.init_worker(...)`
- `driver_worker.init_device()`
- `driver_worker.load_model()`
- `current_platform.update_block_size_for_backend(...)`

I also made the async startup log print:

- `VLLM_WORKER_MULTIPROC_METHOD`

and set its default explicitly to `"fork"` in async-native startup to match the expected inherited-monkeypatch debugging path.

Validation:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py`
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py tests/vllm/test_vllm_inprocess_backend.py .agents/logbook/codex_vllm_refactor.md docs/debug-log-vllm-async-startup.md`

Both passed before the next cluster run.

### Update: launched `v6`

I submitted the next remote debug run with the new pre-fork instrumentation:

- Iris job: `/ahmed/vllm-async-stress-8b-eager-v6`
- script: `experiments/inference/exp_vllm_stress_test.py`
- mode: `native`
- model: `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`
- flags: `--enforce-eager`

Monitoring plan for this run:

1. quick early log check to confirm the new instrumentation appears
2. if the startup is still in progress, check again on roughly a 15-minute cadence
3. if the run repeats the known "hung startup with no new signal" pattern, stop it instead of waiting for the full upstream timeout

### Update: cleaned up stale `v5`

While checking controller state, I found that the previous debug job `/ahmed/vllm-async-stress-8b-eager-v5` was still marked `running`.

That job is no longer useful:

- it was already superseded by the `v6` instrumentation run,
- it represents a known-hung startup signature,
- and it was continuing to consume TPU resources for no new signal.

I stopped it with `iris job stop`.

Current active debug job:

- `/ahmed/vllm-async-stress-8b-eager-v6`

### Update: `v6` produced immediate signal, but exposed an instrumentation bug

The `v6` run produced a much better result than `v4`/`v5` in one respect: the early pre-fork instrumentation is definitely active in the engine subprocess.

Observed lines from `v6`:

- frontend:
  - `Creating AsyncLLM engine ... VLLM_WORKER_MULTIPROC_METHOD='fork'`
  - `installed early async startup instrumentation ... pid=1`

- engine subprocess:
  - `START EngineCoreProc.__init__ pid=276`
  - `START UniProcExecutor._init_executor pid=276`
  - `FAIL UniProcExecutor._init_executor in 0.00s pid=276`
  - `FAIL EngineCoreProc.__init__ in 0.00s pid=276`

This is useful because it proves:

- the pre-fork instrumentation path is working,
- the engine subprocess is inheriting Marin's monkeypatches,
- and the next thing to inspect is the actual exception thrown by my patched `UniProcExecutor._init_executor`.

Important correction:

- `v6` does **not** yet tell us anything new about the original vLLM startup stall,
- because my debug monkeypatch failed before the engine got far enough to reproduce the original behavior.

### Immediate next move

Fetch the full `EngineCore_DP0` traceback from `v6`, fix the monkeypatch bug locally, rerun with a fresh job name, and only then continue narrowing the real async-native startup bottleneck.

### Update: `v6` root cause was my `_init_executor` rewrite

I pulled the full `EngineCore_DP0` traceback from `v6`.
The failure was not in upstream vLLM startup logic. It was in my debug monkeypatch:

- `TypeError: WorkerWrapperBase.__init__() missing 1 required positional argument: 'vllm_config'`

That is important because it means the packaged `vllm-tpu` in the cluster is not API-identical to the sibling source checkout I had been reading.

The relevant correction:

- do **not** reimplement upstream `UniProcExecutor._init_executor` in Marin for debugging,
- instead, wrap the original method and install timed delegate methods on `WorkerWrapperBase`.

I changed the early instrumentation accordingly:

- removed the custom `_init_executor` body replacement,
- kept pure timing wrappers on:
  - `EngineCoreProc.__init__`
  - `UniProcExecutor._init_executor`
  - `UniProcExecutor._distributed_args`
  - `EngineCore._initialize_kv_caches`
- added early timed delegate methods for `WorkerWrapperBase` calls that are normally reached through `__getattr__`:
  - `load_model`
  - `determine_available_memory`
  - `compile_or_warm_up_model`

This version preserves upstream control flow and should therefore be much safer for the next remote run.

Validation after the fix:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py`
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py tests/vllm/test_vllm_inprocess_backend.py .agents/logbook/codex_vllm_refactor.md docs/debug-log-vllm-async-startup.md`

Both passed.

### Update: launched `v7`

I submitted the next remote run after removing the unsafe `_init_executor` rewrite:

- Iris job: `/ahmed/vllm-async-stress-8b-eager-v7`
- script: `experiments/inference/exp_vllm_stress_test.py`
- mode: `native`
- model: `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`
- flags: `--enforce-eager`

This run is the first one that should reflect:

- early pre-fork instrumentation
- no executor-body rewrite
- preserved upstream executor control flow

That makes `v7` the first trustworthy candidate for pinpointing the real async-native startup bottleneck.

### Update: `v7` located the real boundary

The first meaningful `v7` timing lines are the clearest signal so far:

- `START EngineCoreProc.__init__ pid=276`
- `START UniProcExecutor._init_executor pid=276`
- `START UniProcExecutor._distributed_args pid=276`
- `END UniProcExecutor._distributed_args in 0.00s pid=276`
- `START WorkerWrapperBase.init_worker pid=276 rpc_rank=0`
- `worker created worker_class=...TPUWorkerWithExtension executor_backend='uni'`
- `END WorkerWrapperBase.init_worker in 2.41s ...`
- `START WorkerWrapperBase.init_device pid=276`
- `END WorkerWrapperBase.init_device in 24.80s ...`
- `START WorkerWrapperBase.load_model pid=276`

This is the first run that cleanly falsifies the previous theory that the stall was before `init_device`.

Current best conclusion:

- async-native startup on this TPU path is **not** hanging in executor orchestration before worker startup,
- it is progressing through:
  - executor init
  - worker construction
  - device init
- and is now stalled in the model load path reached through `WorkerWrapperBase.load_model`.

That strongly points the next debugging effort toward:

- TPU worker `load_model`
- TPU model runner `load_model`
- dummy/load-format-specific model loader internals

rather than the engine core handshake / executor boundary.

### Immediate operating plan

- keep `v7` running long enough to see whether `WorkerWrapperBase.load_model` ever ends
- use the scheduled later checkpoint to determine whether the stall is truly all within load-model time
- if it remains stuck there, the next code change should instrument inside the load-model path itself, not the executor wrapper layer

### Update: stopped `v7` after isolating `load_model`

I stopped `/ahmed/vllm-async-stress-8b-eager-v7` once it became clear that:

- the run had reached `WorkerWrapperBase.load_model`
- no deeper timing lines were appearing
- and continuing the run would only burn TPU time without a new code change

This was the right stopping point because the next logical step is now clear:

- push the early instrumentation one layer deeper into the TPU load path itself

Specifically, the next pass should instrument early from Marin:

- `TPUWorker.load_model`
- `TPUModelRunner.load_model`
- `tpu_runner.get_model`

and any model-loader entry point immediately below that if needed.

### Update: deeper load-path hooks landed

I pushed the early pre-fork instrumentation one level deeper in `lib/marin/src/marin/inference/vllm_async.py`.

New early TPU load-path hooks:

- `TPUWorker.load_model`
- `TPUModelRunner.load_model`
- `tpu_runner.get_model`

These are installed from Marin before async engine startup, in the same early path that already proved reliable for:

- `EngineCoreProc.__init__`
- `UniProcExecutor._init_executor`
- `WorkerWrapperBase.init_device`
- `WorkerWrapperBase.load_model`

Validation after this change:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py`
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py tests/vllm/test_vllm_inprocess_backend.py .agents/logbook/codex_vllm_refactor.md docs/debug-log-vllm-async-startup.md`

Both passed.

This sets up the next run (`v8`) to answer the next concrete question:

- does the long stall begin at `TPUWorker.load_model`
- or deeper at `TPUModelRunner.load_model`
- or only once `tpu_runner.get_model` is entered

### Update: launched `v8`

I submitted the next remote run with the deeper load-model instrumentation:

- Iris job: `/ahmed/vllm-async-stress-8b-eager-v8`
- script: `experiments/inference/exp_vllm_stress_test.py`
- mode: `native`
- model: `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`
- flags: `--enforce-eager`

This run should distinguish:

1. stall at `TPUWorker.load_model`
2. stall at `TPUModelRunner.load_model`
3. stall only after entering `tpu_runner.get_model`

### Update: `v8` narrowed the stall to `tpu_runner.get_model`

The first `v8` timing lines answer the question unambiguously:

- `START WorkerWrapperBase.load_model`
- `START TPUWorker.load_model`
- `START TPUModelRunner.load_model`
- `START tpu_runner.get_model`

and then no further completion line yet.

That means the current longest-running phase is no longer just "the load-model path" in general.
It has been narrowed to the `tpu_runner.get_model` path itself.

This is the most precise boundary we have reached so far.

### Next instrumentation target

The next code-reading / code-change step should target the first major substeps inside `tpu_runner.get_model`, likely including:

- model loader selection
- dummy-model construction path
- any JAX / mesh / sharding setup done immediately inside or beneath `get_model`

The running `v8` job should be left alive long enough to see whether `tpu_runner.get_model` eventually ends, but the next patch no longer needs to touch the executor layer.

### Update: stopped `v8` after confirming `get_model` is the long pole

I let `v8` run long enough to test whether `tpu_runner.get_model` would complete on its own.

Observed outcome:

- after more than 10 minutes, there was still no `END tpu_runner.get_model`
- and no deeper timing lines below that boundary

That is enough to stop the job and move the debugging focus into `model_loader`.

The next instrumentation pass should therefore target:

- `get_vllm_model`
- any helper directly beneath it that differentiates the dummy-load path from the streamed-weight path
- ideally the exact JAX-jitted model creation step inside the dummy branch
- worker timing:
  - `[marin-vllm-startup] START TPUWorker.init_device ...`
  - `[marin-vllm-startup] END TPUWorker.load_model in ...`
  - `[marin-vllm-startup] START tpu_runner.get_model ...`
  - `[marin-vllm-startup] END TPUWorker.determine_available_memory in ...`
  - `[marin-vllm-startup] END TPUWorker.compile_or_warm_up_model in ...`

If the run still stalls, these logs should identify the exact startup phase that owns the delay.

### Update: next instrumentation target is `get_vllm_model`

I reviewed the TPU model-loading path beneath `tpu_runner.get_model`.

For `MODEL_IMPL_TYPE="vllm"`, the path is:

1. `tpu_runner.get_model(...)`
2. `model_loader.get_vllm_model(...)`
3. `VllmModelWrapper.load_weights()`
4. `VllmModelWrapper.jit_step_func()`
5. `VllmModelWrapper.jit_compute_logits_func()`

That is a much better target than continuing to instrument executor startup.

The next Marin-side patch should add pre-fork timing around:

- `model_loader.get_vllm_model`
- `model_loader.get_flax_model` as a safety check
- `VllmModelWrapper.load_weights`
- `VllmModelWrapper.jit_step_func`
- `VllmModelWrapper.jit_compute_logits_func`

Expected payoff from the next remote run:

- if `load_weights` is the long pole, the stall is likely the dummy vLLM model construction / CPU load / shard-to-TPU path
- if `jit_step_func` or `jit_compute_logits_func` is the long pole, the async-native stall is dominated by JAX compilation of the wrapped vLLM model
- if `get_vllm_model` starts but none of the wrapper methods do, there is another boundary inside model-loader setup that still needs to be broken out

### Update: added deeper model-loader instrumentation

I patched the Marin-side instrumentation in both:

- `lib/marin/src/marin/inference/vllm_async.py`
- `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py`

to emit startup timing for:

- `model_loader.get_model`
- `model_loader.get_vllm_model`
- `model_loader.get_flax_model`
- `VllmModelWrapper.load_weights`
- `VllmModelWrapper.jit_step_func`
- `VllmModelWrapper.jit_compute_logits_func`

This is the first instrumentation pass that can separate:

- weight materialization / model construction cost
- from JAX compilation cost of the wrapped vLLM step functions

Before another remote run, the local narrow test suite and targeted pre-commit checks should be re-run to make sure the extra wrappers did not break the startup helpers.

### Update: deeper instrumentation validated locally

Validation after the model-loader instrumentation patch:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py`
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py tests/vllm/test_vllm_inprocess_backend.py docs/debug-log-vllm-async-startup.md .agents/logbook/codex_vllm_refactor.md`

Both passed.

That means the next useful step is no longer local editing.
It is another Iris run with the same 8B async-native stress command and a fresh job name so the deeper timing lines can be observed cleanly.

### Update: submitted `v9`

Submitted:

`/ahmed/vllm-async-stress-8b-eager-v9`

Command:

```bash
uv run iris --config=lib/iris/examples/marin.yaml job run \
  --tpu v5p-8 --memory 24GB --region us-central1 \
  --extra tpu --extra vllm \
  --job-name vllm-async-stress-8b-eager-v9 --no-wait \
  -- python experiments/inference/exp_vllm_stress_test.py \
  --model gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f \
  --num-prompts 50 --max-concurrent 4 --max-tokens 128 \
  --max-model-len 4096 --mode native --enforce-eager
```

Next operational step:

- do a short initial log sanity check to confirm the new model-loader hooks appear
- if the job is healthy, continue checking on a roughly 15-minute cadence until the next decisive timing boundary appears

### Update: added a stress-test startup timeout knob

The current stress script hardcodes `timeout_seconds=3600` when it enters `VllmEnvironment`.

That is too slow for iterative TPU startup debugging because a clearly hung async-native startup can hold a TPU for an hour before the script exits on its own.

I patched `experiments/inference/exp_vllm_stress_test.py` to accept:

- `--startup-timeout <seconds>`

and pass that through to `VllmEnvironment(timeout_seconds=...)`.

Default remains `3600`, but future debug runs should use a shorter timeout such as `900` once the current `v9` run is done.

### Update: pre-staged the next `load_weights` split for `v10`

While `v9` is running, I patched the next deeper instrumentation layer locally so the next bundle can break down `VllmModelWrapper.load_weights` without another edit cycle.

Added timing around:

- `vllm.model_executor.model_loader.get_model`
- `vllm.model_executor.model_loader.utils.initialize_model`
- `vllm.model_executor.model_loader.utils.process_weights_after_loading`
- `IncrementalModelLoader.load_model`
- `RunaiIncrementalModelLoader.load_model`
- `cleanup_sharding.shard_model_to_tpu`
- `cleanup_sharding._shard_module_to_tpu`

This does not affect the already-submitted `v9` run, but it means the next follow-up run can immediately answer whether `load_weights` time is dominated by:

- upstream vLLM model creation,
- TPU-aware incremental weight loading,
- or the final model sharding/migration to TPU.

### Update: next-cut instrumentation validated locally

Validation after the extra `load_weights`-internal instrumentation patch:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py`
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py experiments/inference/exp_vllm_stress_test.py tests/vllm/test_vllm_inprocess_backend.py docs/debug-log-vllm-async-startup.md .agents/logbook/codex_vllm_refactor.md`

Both passed.

So the local tree is ready for a `v10` run as soon as `v9` provides the next remote signal.

### Update: made async-native fallback an explicit policy

The current automatic subprocess fallback is bad for debugging async-native TPU startup:

- the hung async-native engine can keep the TPU lock,
- the subprocess fallback then starts on a dirty TPU state,
- and the fallback failure adds noisy secondary errors that are not relevant to the native path.

I patched `VllmEnvironment` to accept:

- `native_startup_failure_mode="fallback" | "raise"`

with default still `"fallback"`.

I also patched `experiments/inference/exp_vllm_stress_test.py` to accept:

- `--native-startup-failure-mode fallback|raise`

For future debug runs, the right choice is:

- `--native-startup-failure-mode raise`

so the job fails cleanly at the native startup failure instead of launching the subprocess path.

Validation:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py`
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_server.py experiments/inference/exp_vllm_stress_test.py tests/vllm/test_vllm_inprocess_backend.py .agents/logbook/codex_vllm_refactor.md`

Both passed, including a new test that verifies `native_startup_failure_mode="raise"` does not fall back.

### Update: `v9` delay was worker allocation churn, not startup silence

I used `iris job bug-report /ahmed/vllm-async-stress-8b-eager-v9 --tail 50` to explain why repeated `job logs` polls were empty.

Important result:

- the job entered `running` immediately at the Iris level,
- but the task had two failed worker attempts before a healthy worker finally started,
- and the current worker only began at `2026-03-19T08:11:07Z`.

Observed task history:

- attempt 0: `worker_failed` with `Request timed out`
- attempt 1: `worker_failed` with `Request timed out`
- attempt 2: current healthy worker, now running

So the earlier lack of logs should not be interpreted as more async-native startup evidence.
It was mostly cluster-side worker churn before the task reached a stable worker.

Operational consequence:

- now that the current worker has actually started, log polling should resume immediately because the startup instrumentation may begin appearing soon

### Update: `v9` instrumentation is now live on the healthy worker

Once the third worker attempt started, the `v9` logs began showing the expected async-native startup markers.

Observed on the healthy worker:

- Marin frontend:
  - `Creating AsyncLLM engine with ... enforce_eager=True SKIP_JAX_PRECOMPILE='1' MARIN_VLLM_STARTUP_TIMING='1'`
- Engine core:
  - `START EngineCoreProc.__init__`
  - `START UniProcExecutor._init_executor`
  - `START UniProcExecutor._distributed_args`
  - `END UniProcExecutor._distributed_args in 0.00s`
  - `START WorkerWrapperBase.init_worker`
  - `END WorkerWrapperBase.init_worker in 0.00s`
  - `START WorkerWrapperBase.init_device`

This confirms:

- the current job is now executing the intended async-native path
- the timing instrumentation is active on the real worker attempt
- the next decisive boundary should come after `init_device`, exactly as in the earlier `v7`/`v8` runs

Immediate next step:

- let `v9` run long enough to capture whether it reaches:
  - `WorkerWrapperBase.load_model`
  - `model_loader.get_vllm_model`
  - `VllmModelWrapper.load_weights`
  - or stalls before those boundaries

### Update: `v9` narrowed the long pole to `VllmModelWrapper.load_weights`

The next `v9` timing slice is decisive.

Observed sequence:

- `END WorkerWrapperBase.init_device in 22.89s`
- `START WorkerWrapperBase.load_model`
- `START TPUWorker.load_model`
- `START TPUModelRunner.load_model`
- `START tpu_runner.get_model`
- `START model_loader.get_vllm_model`
- `START VllmModelWrapper.load_weights`

and no deeper completion line from that bundle.

This means the async-native startup bottleneck is now narrowed past:

- executor orchestration,
- TPU device init,
- and model-loader selection,

into the `VllmModelWrapper.load_weights` path itself.

That is enough signal to stop `v9` and move to `v10`, because the local tree already contains the next instrumentation split inside `load_weights`:

- upstream vLLM model loader entry
- incremental TPU model loader methods
- post-load TPU sharding boundaries

### Update: stopped `v9` after the `load_weights` boundary

I terminated `/ahmed/vllm-async-stress-8b-eager-v9` immediately after it proved the bottleneck had entered `VllmModelWrapper.load_weights`.

That was the right cutoff because:

- `v9` had no deeper `load_weights`-internal instrumentation
- leaving it running would only burn TPU time without adding new signal
- the local tree was already ready for the next, deeper run

### Update: submitted `v10` with deeper `load_weights` instrumentation

Submitted:

`/ahmed/vllm-async-stress-8b-eager-v10`

Key differences vs `v9`:

- includes the next instrumentation split inside `VllmModelWrapper.load_weights`
- uses `--startup-timeout 900`
- uses `--native-startup-failure-mode raise`

Command:

```bash
uv run iris --config=lib/iris/examples/marin.yaml job run \
  --tpu v5p-8 --memory 24GB --region us-central1 \
  --extra tpu --extra vllm \
  --job-name vllm-async-stress-8b-eager-v10 --no-wait \
  -- python experiments/inference/exp_vllm_stress_test.py \
  --model gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f \
  --num-prompts 50 --max-concurrent 4 --max-tokens 128 \
  --max-model-len 4096 --mode native --enforce-eager \
  --startup-timeout 900 \
  --native-startup-failure-mode raise
```

Expected payoff:

- if `v10` shows `vllm_model_loader.get_model` or `IncrementalModelLoader.load_model` as the long pole, then the problem is still in model construction / incremental load
- if `v10` reaches `cleanup_sharding.shard_model_to_tpu`, then the long pole is the final CPU-to-TPU sharding/migration path

### Update: `v10` revealed an instrumentation packaging mismatch

The first `v10` bug-report is already useful.

It showed this warning during async-native startup:

- `Could not install early TPU load-model instrumentation: No module named 'tpu_inference.layers.vllm.process_weights'`

That means I grouped the new optional `cleanup_sharding` import too early in `vllm_async.py`.
On the installed cluster package, that module path is not importable under the same name as the source checkout.

Consequence:

- the whole deeper TPU load-model instrumentation block aborted early
- so `v10` is not a valid run for the new `load_weights`-internal boundaries

Immediate fix required:

- split the optional imports into smaller `try` blocks
- let missing `cleanup_sharding` instrumentation degrade gracefully
- still install the higher-value wrappers for:
  - `vllm_model_loader.get_model`
  - `IncrementalModelLoader.load_model`
  - `RunaiIncrementalModelLoader.load_model`
  - `VllmModelWrapper.load_weights`

### Update: fixed the optional import granularity and stopped `v10`

I patched both:

- `lib/marin/src/marin/inference/vllm_async.py`
- `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py`

to separate:

- core TPU load-model instrumentation
- extended model-loader instrumentation
- cleanup-sharding instrumentation

into independent `try` blocks.

That means a missing `cleanup_sharding` module no longer suppresses:

- `TPUWorker.load_model`
- `model_loader.get_vllm_model`
- `VllmModelWrapper.load_weights`
- `IncrementalModelLoader.load_model`

I then terminated `/ahmed/vllm-async-stress-8b-eager-v10`, because it was already running with the broken grouped-import behavior and could not produce the desired deeper signal.

Validation:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py`
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py tests/vllm/test_vllm_inprocess_backend.py .agents/logbook/codex_vllm_refactor.md docs/debug-log-vllm-async-startup.md`

Both passed.

### Update: submitted `v11` with corrected optional-import handling

Submitted:

`/ahmed/vllm-async-stress-8b-eager-v11`

This run keeps the same debugging posture as `v10`:

- `--startup-timeout 900`
- `--native-startup-failure-mode raise`

but now uses the corrected instrumentation import structure so a missing cleanup-sharding module should no longer block:

- `model_loader.get_vllm_model`
- `VllmModelWrapper.load_weights`
- `vllm_model_loader.get_model`
- `IncrementalModelLoader.load_model`

### Update: split upstream vLLM loader hooks away from missing TPU loader hooks

The first `v11` bug-report showed a second packaging mismatch:

- `tpu_inference.models.vllm.vllm_model_loader` is also missing in the installed package

That meant I was still over-grouping optional imports and accidentally suppressing:

- `vllm.model_executor.model_loader.get_model`
- `vllm.model_executor.model_loader.utils.initialize_model`
- `vllm.model_executor.model_loader.utils.process_weights_after_loading`

I patched both Marin instrumentation sites again so:

- upstream vLLM loader hooks are installed independently
- TPU incremental-loader hooks are attempted separately

Validation:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py`
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py tests/vllm/test_vllm_inprocess_backend.py .agents/logbook/codex_vllm_refactor.md docs/debug-log-vllm-async-startup.md`

Both passed.

This change does not affect the already-running `v11`, but it means the next rerun can expose upstream vLLM model-loader boundaries even if the TPU incremental-loader module remains unavailable.

### Update: `v11` still narrows only to `VllmModelWrapper.load_weights`

The decisive `v11` timing slice is:

- `END WorkerWrapperBase.init_device in 26.06s`
- `START WorkerWrapperBase.load_model`
- `START TPUWorker.load_model`
- `START TPUModelRunner.load_model`
- `START tpu_runner.get_model`
- `START model_loader.get_vllm_model`
- `START VllmModelWrapper.load_weights`

with no deeper boundary in that already-submitted bundle.

That means `v11` reproduced the same final narrowing as `v9`, but it is still useful because it confirms:

- the core deeper hooks survive the optional-import refactor
- the remaining missing signal is exactly the upstream `vllm_model_loader` / `utils` split

So `v11` should be stopped and replaced with `v12`, which is the first run that can expose:

- `vllm_model_loader.get_model`
- `vllm_model_loader_utils.initialize_model`
- `vllm_model_loader_utils.process_weights_after_loading`

### Update: stopped `v11` and submitted `v12`

I terminated `/ahmed/vllm-async-stress-8b-eager-v11` as soon as it confirmed the remaining boundary was still `VllmModelWrapper.load_weights`.

Then I submitted:

`/ahmed/vllm-async-stress-8b-eager-v12`

with the same debug posture:

- `--startup-timeout 900`
- `--native-startup-failure-mode raise`

but now on the tree that also installs upstream vLLM model-loader timing independently of the missing TPU incremental-loader module.

This is the first run that should be able to answer:

- does `VllmModelWrapper.load_weights` enter `vllm_model_loader.get_model`
- and if so, does time accumulate in `initialize_model` or later in `process_weights_after_loading`

### Update: `v12` showed no entry into upstream `vllm_model_loader.get_model`

The decisive `v12` timing slice is:

- `END WorkerWrapperBase.init_device in 22.51s`
- `START WorkerWrapperBase.load_model`
- `START TPUWorker.load_model`
- `START TPUModelRunner.load_model`
- `START tpu_runner.get_model`
- `START model_loader.get_vllm_model`
- `START VllmModelWrapper.load_weights`

and then still no:

- `START vllm_model_loader.get_model`
- `START vllm_model_loader_utils.initialize_model`
- `START vllm_model_loader_utils.process_weights_after_loading`

That means the next bottleneck is narrower than "inside load_weights somewhere after handing off to upstream vLLM".

The two realistic explanations are:

1. time is being spent in the setup code at the top of `VllmModelWrapper.load_weights` before the `vllm_get_model(...)` call
2. the call goes through the module-local alias imported into `vllm_model_wrapper.py`, so patching `vllm.model_executor.model_loader.get_model` is not enough to see it

This is enough signal to stop `v12` and instrument the module-local aliases inside `tpu_inference.models.vllm.vllm_model_wrapper`, especially:

- `vllm_get_model`
- `shard_model_to_tpu`

### Update: added alias-level instrumentation inside `vllm_model_wrapper`

I patched both Marin instrumentation sites to wrap the module-local call sites that `VllmModelWrapper.load_weights` actually uses:

- `vllm_model_wrapper.vllm_get_model`
- `vllm_model_wrapper.shard_model_to_tpu`
- `vllm_model_wrapper.load_lora_model`

Validation:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py`
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py tests/vllm/test_vllm_inprocess_backend.py .agents/logbook/codex_vllm_refactor.md docs/debug-log-vllm-async-startup.md`

Both passed.

This is the first patch that can distinguish:

- pre-`vllm_get_model` setup inside `load_weights`
- from the `vllm_get_model` handoff itself
- from the later `shard_model_to_tpu` phase

### Update: stopped `v12` and submitted `v13`

I terminated `/ahmed/vllm-async-stress-8b-eager-v12` once it showed the maximum signal available from that bundle.

Then I submitted:

`/ahmed/vllm-async-stress-8b-eager-v13`

with the same debug posture:

- `--startup-timeout 900`
- `--native-startup-failure-mode raise`

but now on the alias-level instrumentation tree.

This is the first run that can answer:

- does `VllmModelWrapper.load_weights` reach `vllm_model_wrapper.vllm_get_model`
- or is the stall entirely inside the setup code before that call

### Update: `v13` is still in worker build, no runtime signal yet

The first `v13` bug-report shows:

- job state: `running`
- task state: `building`
- worker: `marin-tpu_v5p_8-us-central1-a-20260319-0826-f6720091-worker-0`
- task has not started yet

So there is currently no new async-native startup evidence from `v13`.

The correct operational move is to keep monitoring until the task leaves `building` and starts emitting container logs, not to interpret the empty logs as another engine startup stall.

### Update: `v13` still building after another 10 minutes

I rechecked `v13` after another wait interval.

Result:

- job still `running`
- task still `building`
- same worker still healthy
- task still has not started

So there is still no engine-level signal from `v13`.

Given that the worker is healthy and there are no failed attempts yet, the right move remains:

- keep the same job alive
- continue periodic `bug-report` / log checks
- avoid creating more cluster churn by resubmitting another copy of the same run

### Update: `v13` hit repeated worker timeouts and was reassigned

The later `v13` bug-report changed materially:

- first worker timed out
- second worker also timed out
- the job has now been reassigned to a third healthy worker

Current snapshot:

- job state: `running`
- task state: `assigned`
- preemptions: `1`
- attempts:
  - attempt 0: worker failed, request timed out
  - attempt 1: worker failed, request timed out
  - attempt 2: new worker assigned, not started yet

This means there is still no new async-native engine signal from `v13`.

Operationally, the best move is still to keep the same job alive and continue monitoring.
At this point the dominant source of delay is cluster worker churn, not the async-native code path itself.

### Update: autoscaler status confirms repeated v5p-8 worker churn

I checked:

`uv run iris --config=lib/iris/examples/marin.yaml rpc controller get-autoscaler-status`

The useful takeaway is not the full JSON dump but the operational pattern:

- `v13` keeps getting routed to `tpu_v5p_8-us-central1-a`
- workers are repeatedly timing out before the task reaches container runtime
- the controller is continuing to scale up fresh `v5p-8` slices in `us-central1-a`
- older idle/problematic slices are being scaled down while new ones come online

So the current blocker is clearly cluster execution churn, not a new async-native code-path ambiguity.

Best current action:

- leave `/ahmed/vllm-async-stress-8b-eager-v13` alive
- let it continue trying on fresh `v5p-8` workers
- resume log inspection only after the task actually starts and emits runtime logs

### Update: final `v13` outcome was the 900s async-native startup timeout

The later `v13` state did eventually move past worker churn and into a real runtime attempt.

Final Iris result:

- job state: `failed`
- error: `TimeoutError: In-process vLLM server did not become ready within 900s`
- final runtime attempt duration: about `16m 35s`

The important log sequence from the successful runtime attempt is:

- `START VllmModelWrapper.load_weights`
- `START vllm_model_wrapper.vllm_get_model`
- vLLM log: `Initializing vLLM model with random weights, weight loading skipped.`
- no `END vllm_model_wrapper.vllm_get_model`
- no `START vllm_model_wrapper.shard_model_to_tpu`
- eventually the frontend times out waiting for `/v1/models`

This is the strongest narrowing so far.

It means the current async-native long pole is not just "inside load_weights".
It is specifically inside the `vllm_model_wrapper.vllm_get_model` phase, before control returns to the wrapper and before the later `shard_model_to_tpu` phase begins.

So the next instrumentation cut should target the first heavy phases beneath `vllm_get_model` itself, not the outer wrapper anymore.

### Update: reconciled Claude's logbook with the current tree

I read `.agents/logbook/claude_vllm_refactor.md` and compared it against the current checkout.

The parts that materially change next steps are:

- `VLLM_ENABLE_V1_MULTIPROCESSING=0` is already set and is not sufficient
- switching to direct `AsyncLLM.from_engine_args(...)` also did not avoid the engine subprocess
- so the next work should stay focused on loader internals beneath `vllm_get_model`, not on reworking engine construction again

One additional aliasing detail matters for the next patch:

- for `load_format=dummy`, upstream `BaseModelLoader.load_model` uses module-local imports of
  `initialize_model` and `process_weights_after_loading`
- `DummyModelLoader.load_weights` uses the module-local alias `initialize_dummy_weights`

So the next instrumentation needs to wrap those exact aliases, not just the canonical upstream
module functions, otherwise the logs can still miss the real boundary.

### Update: added dummy-loader alias instrumentation and validated it

I patched both Marin startup instrumentation sites to wrap the exact upstream dummy-loader aliases:

- `BaseModelLoader.load_model`
- `DummyModelLoader.load_weights`
- `vllm_base_loader.initialize_model`
- `vllm_base_loader.process_weights_after_loading`
- `vllm_dummy_loader.initialize_dummy_weights`
- `vllm_weight_utils.initialize_dummy_weights`

Validation:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py`
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py .agents/logbook/codex_vllm_refactor.md docs/debug-log-vllm-async-startup.md tests/vllm/test_vllm_inprocess_backend.py`

Both passed.

This is the first patch that can cleanly distinguish whether the stall under
`vllm_model_wrapper.vllm_get_model` is in:

- upstream model construction,
- dummy weight initialization,
- or post-load processing.

### Update: submitted `v14`

Submitted:

`/ahmed/vllm-async-stress-8b-eager-v14`

Command:

```bash
uv run iris --config=lib/iris/examples/marin.yaml job run \
  --tpu v5p-8 --memory 24GB --region us-central1 \
  --extra tpu --extra vllm \
  --job-name vllm-async-stress-8b-eager-v14 --no-wait \
  -- python experiments/inference/exp_vllm_stress_test.py \
  --model gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f \
  --num-prompts 50 --max-concurrent 4 --max-tokens 128 \
  --max-model-len 4096 --mode native --enforce-eager \
  --startup-timeout 900 \
  --native-startup-failure-mode raise
```

Expected payoff from `v14`:

- if the next log is `START vllm_base_loader.initialize_model`, the long pole is model construction
- if the next log is `START vllm_dummy_loader.initialize_dummy_weights`, the long pole is dummy init
- if both complete and it stalls later, then post-load processing is the remaining heavy phase

### Update: early `v14` result narrows the stall to dummy weight initialization

The first `v14` bug-report already answered the next question.

Observed sequence:

- `START VllmModelWrapper.load_weights`
- `START vllm_model_wrapper.vllm_get_model`
- `START BaseModelLoader.load_model`
- `START vllm_base_loader.initialize_model`
- `END vllm_base_loader.initialize_model in 8.50s`
- `START DummyModelLoader.load_weights`
- `START vllm_dummy_loader.initialize_dummy_weights`

That is a major narrowing.

It means:

- upstream model construction is not the 900s long pole
- the current long pole is inside dummy weight initialization itself
- we still have not reached post-load processing

So the next instrumentation cut, if needed, should go one step deeper under dummy init rather than continuing to explore broader loader structure.

### Update: 10-minute check on `v14`

I rechecked `/ahmed/vllm-async-stress-8b-eager-v14` after about 10 minutes.

Current state:

- job still `running`
- single healthy worker
- no worker churn this time

Important current log sequence:

- `START vllm_model_wrapper.vllm_get_model`
- `START BaseModelLoader.load_model`
- `START vllm_base_loader.initialize_model`
- `END vllm_base_loader.initialize_model in 8.50s`
- `START DummyModelLoader.load_weights`
- `START vllm_dummy_loader.initialize_dummy_weights`

and then no completion line yet for dummy init.

So as of this check, the current narrowest long pole is:

- `vllm_dummy_loader.initialize_dummy_weights`

That is a better answer than we had before this run.

### Update: final `v14` result confirms the stall is inside dummy weight initialization

I checked `/ahmed/vllm-async-stress-8b-eager-v14` again after the 10-minute snapshot.

Final state:

- job `failed`
- failure: `TimeoutError: In-process vLLM server did not become ready within 900s`
- task runtime: about 16m31s

Final high-signal startup sequence:

- `START WorkerWrapperBase.init_device`
- `END WorkerWrapperBase.init_device in 25.50s`
- `START WorkerWrapperBase.load_model`
- `START TPUWorker.load_model`
- `START TPUModelRunner.load_model`
- `START tpu_runner.get_model`
- `START model_loader.get_vllm_model`
- `START VllmModelWrapper.load_weights`
- `START vllm_model_wrapper.vllm_get_model`
- `START BaseModelLoader.load_model`
- `START vllm_base_loader.initialize_model`
- `END vllm_base_loader.initialize_model in 8.50s`
- `START DummyModelLoader.load_weights`
- `START vllm_dummy_loader.initialize_dummy_weights`
- no corresponding `END` line before the 900s timeout fired

That is the cleanest boundary so far.

What `v14` rules out:

- engine-core orchestration before `init_device`
- TPU device initialization itself
- upstream model construction in `initialize_model`

What remains live:

- the heavy work is inside upstream dummy weight initialization for the vLLM dummy loader path
- we still have not shown whether the hot substep is parameter enumeration, per-parameter random array creation, post-processing inside dummy init, or a TPU/JAX interaction called from that routine

Most logical next debugging cut:

- instrument one level deeper under `vllm_dummy_loader.initialize_dummy_weights`
- do it carefully to avoid per-parameter log spam; prefer coarse wrappers around the helper(s) it delegates to rather than raw logging on every tensor

### Update: source inspection under dummy init

I inspected the upstream dummy-loader implementation after `v14`.

The relevant structure is:

- `DummyModelLoader.load_weights` calls `initialize_dummy_weights(model, model_config)`
- `initialize_dummy_weights(...)` iterates `model.state_dict().values()`
- for each parameter it calls `initialize_single_dummy_weight(...)`
- on TPU that helper creates CPU random data, copies into the parameter, and then calls `torch._sync(param)`

So the next instrumentation cut should focus on `initialize_single_dummy_weight(...)` or an aggregated wrapper around the per-parameter loop, because that is the first concrete upstream substep that can explain a many-minute stall inside dummy initialization.

### Update: added aggregated dummy-init progress instrumentation

I added a shared helper in `lib/marin/src/marin/rl/environments/inference_ctx/startup_debug.py` and wired it into both:

- `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py`
- `lib/marin/src/marin/inference/vllm_async.py`

What changed:

- stop treating `initialize_dummy_weights(...)` as a single black-box timing region
- replace the generic wrapper on that function with aggregated progress logging around the real per-parameter loop
- keep the existing no-flood constraint by logging:
  - loop start / loop end
  - periodic progress (`params=... numel=... elapsed=...`)
  - large-parameter boundaries (`dummy-init.param-start` / `dummy-init.param-end`)
  - slow-parameter completions

Why this is the right next cut:

- `v14` proved the stall is inside `vllm_dummy_loader.initialize_dummy_weights`
- the next unknown is whether we are making steady progress through many parameters or stalling on one specific large tensor / `torch._sync` call

Local validation:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py`
- `./infra/pre-commit.py lib/marin/src/marin/rl/environments/inference_ctx/startup_debug.py lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py lib/marin/src/marin/inference/vllm_async.py tests/vllm/test_vllm_inprocess_backend.py`

Both passed.

### Update: submitting `v15`

Next run uses the same 8B async-native debug path, but with a longer startup timeout so the new dummy-init progress lines have time to show whether we are progressing or wedged on a specific parameter.

Submitted:

`/ahmed/vllm-async-stress-8b-eager-v15`

Command:

```bash
uv run iris --config=lib/iris/examples/marin.yaml job run \
  --tpu v5p-8 --memory 24GB --region us-central1 \
  --extra tpu --extra vllm \
  --job-name vllm-async-stress-8b-eager-v15 --no-wait \
  -- python experiments/inference/exp_vllm_stress_test.py \
  --model gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f \
  --num-prompts 50 \
  --max-concurrent 4 \
  --max-tokens 128 \
  --max-model-len 4096 \
  --mode native \
  --enforce-eager \
  --startup-timeout 1800 \
  --native-startup-failure-mode raise
```

Expected new signals from `v15`:

- `START dummy-init.loop ...`
- `dummy-init.progress params=... numel=... elapsed=...`
- `dummy-init.param-start ... name=...`
- `dummy-init.param-end ... name=... elapsed=...`

These should tell us whether upstream dummy init is steadily progressing or wedged on one specific large tensor / TPU sync boundary.

### Update: 20-minute check on `v15`

I waited about 20 minutes and checked `/ahmed/vllm-async-stress-8b-eager-v15`.

Current state at check time:

- job still `running`
- single healthy worker
- no worker churn

High-signal progression:

- `END WorkerWrapperBase.init_device in 23.68s`
- `START WorkerWrapperBase.load_model`
- `START TPUWorker.load_model`
- `START TPUModelRunner.load_model`
- `START tpu_runner.get_model`
- `START model_loader.get_vllm_model`
- `START VllmModelWrapper.load_weights`
- `START vllm_model_wrapper.vllm_get_model`
- `START BaseModelLoader.load_model`
- `START vllm_base_loader.initialize_model`
- `END vllm_base_loader.initialize_model in 9.90s`
- `START DummyModelLoader.load_weights`
- `END DummyModelLoader.load_weights in 879.53s`
- `START vllm_base_loader.process_weights_after_loading`

This is the first run that has clearly advanced past the previous `v14` stopping point.

What this means:

- the old 900s timeout was masking forward progress in dummy loading
- upstream dummy load is extremely expensive on this path, but it is not strictly wedged
- the next live boundary is now `vllm_base_loader.process_weights_after_loading`

Important anomaly:

- the new aggregated `dummy-init.*` progress lines did not appear at all
- that means the wrapper around `initialize_dummy_weights(...)` did not hit the active inner call path in the runtime we are actually executing
- so the lack of inner progress is now itself a debugging signal: either the active loader path differs from the expected upstream alias path, or the call site binds a different helper than the one we wrapped

### Update: recommendation after the follow-up live check on `v15`

I rechecked `v15` after the 20-minute snapshot.

Current live state:

- job still `running`
- still on the same healthy worker
- still no worker churn
- latest visible boundary remains `START vllm_base_loader.process_weights_after_loading`

Recommendation:

- yes, keep waiting on this run for now

Reasoning:

- `v15` is the first run that clearly progressed beyond the old `v14` timeout boundary
- there is no crash signature, no TPU lock conflict, and no scheduler churn
- the longer `1800s` startup timeout is doing useful work here by allowing us to see deeper into the true load path

Practical read:

- killing this run now would throw away the first clean chance to learn how expensive `process_weights_after_loading` is on the real path
- the next decision point should be when the run either:
  - emits a new boundary after `process_weights_after_loading`, or
  - reaches the full startup timeout and fails

### Update: final `v15` result and architectural rethink

I rechecked `/ahmed/vllm-async-stress-8b-eager-v15` after it finished.

Final state:

- job `failed`
- failure: `TimeoutError: In-process vLLM server did not become ready within 1800s`
- runtime: about 31 minutes

Final high-signal boundaries:

- `END WorkerWrapperBase.init_device in 23.68s`
- `END vllm_base_loader.initialize_model in 9.90s`
- `END DummyModelLoader.load_weights in 879.53s`
- `START vllm_base_loader.process_weights_after_loading`
- no matching `END` before the 1800s timeout

This changes the interpretation.

`v15` proves the startup path is making forward progress, but it is doing the wrong work for Marin's goal.

The async-native path currently forces:

- `MODEL_IMPL_TYPE=vllm`
- `load_format=dummy`

That sends TPU startup into the PyTorch-wrapper path in `tpu_inference.models.vllm.vllm_model_wrapper.VllmModelWrapper.load_weights()`, which:

1. calls upstream `vllm_get_model(...)`
2. runs `DummyModelLoader.load_weights(...)` to fill the full model with random weights
3. runs upstream `process_weights_after_loading(...)`
4. only after that returns control to the wrapper, where later TPU sharding / Marin weight injection can happen

So the async-native server is currently paying for a huge random-weight dummy load and post-load processing pass that Marin does not actually want, because we overwrite those weights immediately afterward with real streamed shards.

This is the wrong architecture for fast loading.

Most important new conclusion:

- the current problem is no longer "mysterious engine hang"
- it is that `vllm_async.py` is forcing the TPU `MODEL_IMPL_TYPE="vllm"` loader path, which is structurally inconsistent with Marin's proven fast-load story

The old direct/working Marin paths did **not** force that env var. On TPU, `MODEL_IMPL_TYPE=auto` resolves to `flax_nnx` for Llama unless the architecture is in the TPU fork's `_VLLM_PREFERRED_ARCHITECTURES` set.

That means the next move should not be more timing probes inside the current forced-`vllm` path.

The next move should be to realign async-native startup with the same JAX/flax-nnx loading model Marin already relies on elsewhere, and stop forcing `MODEL_IMPL_TYPE=vllm` in `vllm_async.py`.

### Update: switched async-native startup back to TPU `MODEL_IMPL_TYPE=auto`

I made the architectural correction in `lib/marin/src/marin/inference/vllm_async.py`.

Change:

- stop forcing `MODEL_IMPL_TYPE=vllm`
- default async-native startup to `MODEL_IMPL_TYPE=auto`
- add `MODEL_IMPL_TYPE` to the startup log line so the next run proves which TPU model-loader path we actually took

Why:

- `v15` showed the forced `vllm` path spends about 15 minutes in `DummyModelLoader.load_weights` and then stalls in `process_weights_after_loading`
- that path constructs and post-processes random full-model weights before Marin injects the real streamed weights
- this is exactly the wrong shape for Marin fast loading
- the TPU fork's natural `auto` resolution picks `flax_nnx` for Llama, which is much closer to the JAX path Marin's earlier fast-loading work relied on

Validation:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py`
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py tests/vllm/test_vllm_inprocess_backend.py`

Both passed.

Expected signal from the next run:

- startup log contains `MODEL_IMPL_TYPE='auto'`
- the worker logs should prefer `model_loader.get_flax_model` instead of `model_loader.get_vllm_model`

### Update: submitted `v16`

Submitted:

`/ahmed/vllm-async-stress-8b-eager-v16`

Command:

```bash
uv run iris --config=lib/iris/examples/marin.yaml job run \
  --tpu v5p-8 --memory 24GB --region us-central1 \
  --extra tpu --extra vllm \
  --job-name vllm-async-stress-8b-eager-v16 --no-wait \
  -- python experiments/inference/exp_vllm_stress_test.py \
  --model gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f \
  --num-prompts 50 \
  --max-concurrent 4 \
  --max-tokens 128 \
  --max-model-len 4096 \
  --mode native \
  --enforce-eager \
  --startup-timeout 1200 \
  --native-startup-failure-mode raise
```

Primary success signal:

- startup log shows `MODEL_IMPL_TYPE='auto'`
- worker path enters `model_loader.get_flax_model`
- we do **not** spend another 15+ minutes in `DummyModelLoader.load_weights` on the forced `vllm` path

Primary failure signal:

- startup log still shows `MODEL_IMPL_TYPE='vllm'`, which would mean some other environment source is still forcing the old path

### Update: final `v16` result

I checked the final logs for `/ahmed/vllm-async-stress-8b-eager-v16`.

Final state:

- job `failed`
- failure: `TimeoutError: In-process vLLM server did not become ready within 1200s`
- runtime: about 21 minutes

Important thing `v16` *did* prove:

- startup log showed `MODEL_IMPL_TYPE='auto'`
- TPU loader resolved `MODEL_IMPL_TYPE 'auto'` to `'flax_nnx'`
- the worker path entered `model_loader.get_flax_model`

So the architecture correction worked. We are no longer on the forced PyTorch `vllm` wrapper path from `v15`.

High-signal startup sequence:

- `Creating AsyncLLM engine ... MODEL_IMPL_TYPE='auto' ...`
- `END WorkerWrapperBase.init_device in 23.28s`
- `START WorkerWrapperBase.load_model`
- `START TPUWorker.load_model`
- `START TPUModelRunner.load_model`
- `START tpu_runner.get_model`
- `START model_loader.get_flax_model`
- no matching `END model_loader.get_flax_model` before the 1200s timeout

What this means:

- we **did** learn something important from `v16`
- the previous long pole in the forced-`vllm` path was real, and removing it changed the active bottleneck
- the current long pole is now inside the JAX/flax-nnx loader path beneath `model_loader.get_flax_model`

So the problem is no longer the upstream vLLM dummy-loader wrapper.
The next debugging/fix work should focus on the `flax_nnx` load path itself.

### Update: assessment of `claude_vllm_refactor.md`

Claude's logbook is mixed.

What held up:

- `VLLM_ENABLE_V1_MULTIPROCESSING=0` does not remove the engine core subprocess on TPU in this vLLM build
- switching from `build_async_engine_client_from_engine_args(...)` to direct `AsyncLLM.from_engine_args(...)` does not materially change that
- the current async-native startup path does a large amount of real loader work before Marin injects any real weights

What later runs disproved or weakened:

- the problem is not well explained as "the subprocess is the whole problem"
- `v15` showed the forced-`vllm` path was making slow forward progress through dummy load + post-load processing, not deadlocked
- `v16` showed that even after switching to `MODEL_IMPL_TYPE=auto`, the startup still times out, now inside `model_loader.get_flax_model`

So the strongest current interpretation is broader:

- the AsyncLLM TPU startup path is still doing too much up-front model construction / dummy-load work, regardless of whether the active implementation is `vllm` or `flax_nnx`
- the real goal should be a Marin-specific async-native bootstrap path that skips these dummy-loader phases entirely, while preserving async-native serving and WorkerExtension weight injection

### Update: honest assessment of whether Marin can change vLLM model loading without editing the dependency

There is a real public extension point in upstream vLLM:

- `vllm.model_executor.model_loader.register_model_loader(load_format)`

So in a narrow sense, yes: Marin can change which loader runs by registering a custom `load_format` at runtime and passing that `load_format` through engine args.

But for the thing we actually want — an async-native bootstrap that skips the expensive dummy initialization path while still leaving the TPU engine in a state where WorkerExtension `_sync_weights` can load the real streamed shards — the honest answer is less clean.

Current assessment:

- changing `MODEL_IMPL_TYPE` and `load_format` from Marin is possible without forking the package
- replacing the loader with something truly fast is **probably not possible using only stable public APIs**
- the reason is that the TPU fork's `get_flax_model(...)` / `get_vllm_model(...)` logic and the model-runner expectations are internal and determine when models become concrete, when post-load processing runs, and what state `_sync_weights` expects

So there are two levels of "without changing internals":

1. **Without editing the dependency source tree**: probably yes, via runtime registration / monkeypatching.
2. **Without relying on vLLM / tpu-inference internals at all**: probably no.

### Update: clarification on what "TPU internals" means

When I say Marin would still depend on TPU internals, I mean the implementation details in the `vllm-tpu` / `tpu-inference` codebase, such as:

- `tpu_inference.models.common.model_loader.get_model(...)`
- `get_flax_model(...)`
- `get_vllm_model(...)`
- `TPUModelRunner.load_model(...)`
- `TPUModelRunner._sync_weights(...)`

These are the pieces that define what initial `state` shape the runner expects and what later weight injection updates.

One extra detail from source inspection:

- Llama's JAX class (`LlamaForCausalLM`) is **not** `LoadableWithIterator`
- so on the `flax_nnx` path with `load_format='dummy'`, it does **not** take the `jax_dummy` abstract-loader branch
- instead it takes the concrete random-init + sharded-model branch in `_get_nnx_model(...)`

That explains why `v16` can still spend 20 minutes in `get_flax_model(...)` even though it escaped the forced PyTorch `vllm` wrapper path.

### Update: monkeypatch plan after TPU source review

I reviewed the TPU loader code under `/Users/ahmed/code/vllm_tpu_multi` to ground the next design step.

Important confirmations:

- `TPUModelRunner.load_model()` still blocks on `get_model(...)` returning the full runtime bundle (`model_fn`, `compute_logits_fn`, `state`, etc.), so async-native server readiness cannot bypass TPU bootstrap entirely.
- In `tpu_inference.models.common.model_loader._get_nnx_model(...)`, `load_format == "dummy"` plus a model class that is **not** `LoadableWithIterator` takes the concrete random-init branch.
- `LlamaForCausalLM` is `nnx.Module`, not `LoadableWithIterator`, so current Llama async-native startup necessarily takes that expensive branch on the `flax_nnx` path.
- Qwen JAX models *do* implement `LoadableWithIterator`, which is why the TPU fork already has a cheaper `jax_dummy` flow for them.
- The strongest Marin-only plan is therefore a runtime monkeypatch around the `flax_nnx` bootstrap branch, not more changes to AsyncLLM construction.

I captured the concrete implementation plan in `.agents/projects/vllm_async_monkeypatch_plan.md`.

### Update: implemented first-pass Marin fast TPU bootstrap monkeypatch

I implemented the first pass of the Marin-only async-native TPU bootstrap patch.

Code changes:

- Added `lib/marin/src/marin/inference/vllm_tpu_bootstrap_patch.py`
- Wired it into async-native startup in `lib/marin/src/marin/inference/vllm_async.py`
- Wired it into the worker-process import path in `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py`
- Extended `tests/vllm/test_vllm_inprocess_backend.py`

What the patch does:

- Adds a guarded runtime monkeypatch for `tpu_inference.models.common.model_loader._get_nnx_model(...)`
- Only activates when `MARIN_VLLM_FAST_BOOTSTRAP=1`
- Only intercepts the `flax_nnx` Llama `dummy` bootstrap branch
- Replaces the expensive random-init path with:
  - `nnx.eval_shape(...)`
  - zero-valued parameter materialization via `assign_and_shard_param(...)`
  - normal JIT/cache wiring
- Leaves all unsupported architectures and non-matching load paths on the original dependency behavior

Important implementation note:

- The written plan proposed switching async-native CLI args to a new load format marker immediately.
- I intentionally did **not** do that in the first pass.
- I kept `load_format='dummy'` and gated the new behavior only with `MARIN_VLLM_FAST_BOOTSTRAP=1`.
- Reason: this preserves fallback behavior for unsupported architectures if the monkeypatch is absent or the model is not yet covered.

Additional logging added:

- async startup logs now include `MARIN_VLLM_FAST_BOOTSTRAP`
- worker/engine stderr should now show:
  - `Applying Marin fast TPU bootstrap patch`
  - `Using Marin zero-bootstrap flax_nnx path ...`
  - `START Marin zero-bootstrap materialization ...`
  - `END Marin zero-bootstrap materialization ...`

Local validation:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py` -> `20 passed`
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_tpu_bootstrap_patch.py lib/marin/src/marin/inference/vllm_async.py lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py tests/vllm/test_vllm_inprocess_backend.py` -> OK

Next high-signal remote run:

- rerun 8B async-native eager stress test
- confirm we see the new Marin zero-bootstrap log lines
- measure whether startup now gets past `model_loader.get_flax_model(...)` quickly enough to reach streamed shard injection

### Update: submitted v17

Submitted the first remote validation run for the new Marin zero-bootstrap patch:

- job: `/ahmed/vllm-async-stress-8b-eager-v17`
- command: `python experiments/inference/exp_vllm_stress_test.py --model gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f --num-prompts 50 --max-concurrent 4 --max-tokens 128 --max-model-len 4096 --mode native --enforce-eager --startup-timeout 1200 --native-startup-failure-mode raise`

Initial controller state:

- job state: `running`
- task state: `assigned`
- worker: `marin-tpu_v5p_8-us-central1-a-20260319-2130-db37fd4b-worker-0`
- no failures / preemptions so far

Next thing to watch in logs:

- `Applying Marin fast TPU bootstrap patch`
- `Using Marin zero-bootstrap flax_nnx path`
- `START Marin zero-bootstrap materialization`
- `END Marin zero-bootstrap materialization`
- whether startup reaches weight injection / OpenAI readiness before the 1200s timeout

### Update: v17 failed because the patch never installed

`/ahmed/vllm-async-stress-8b-eager-v17` failed with the same 1200s startup timeout, but the important result is why.

Key log line from the worker process before engine startup:

- `WARNING:marin.inference.vllm_tpu_bootstrap_patch: Could not import TPU modules for Marin fast bootstrap patch: cannot import name 'LoadableWithIterator' from 'tpu_inference.models.jax.utils.weight_utils'`

This means `v17` did **not** actually exercise the new zero-bootstrap path.

Corroborating signals:

- none of the new patch log lines appeared:
  - `Applying Marin fast TPU bootstrap patch`
  - `Using Marin zero-bootstrap flax_nnx path`
  - `START Marin zero-bootstrap materialization`
- the engine went straight from:
  - `START model_loader.get_flax_model`
  to
  - 20 minutes later, timeout

Conclusion:

- the first-pass patch was too brittle for the installed `tpu_inference` package on the cluster
- the next fix is straightforward: remove the `LoadableWithIterator` import dependency and gate only on the supported architecture name (`LlamaForCausalLM`)

### Update: fixed the v17 patch-install failure

I removed the `LoadableWithIterator` import dependency from `vllm_tpu_bootstrap_patch.py`.

Why:

- the installed `tpu_inference` package on the cluster does not export `LoadableWithIterator` from `tpu_inference.models.jax.utils.weight_utils`
- that import failure caused the entire monkeypatch to be skipped in `v17`

What changed:

- the patch now imports only `assign_and_shard_param`
- activation is now gated only on:
  - `MARIN_VLLM_FAST_BOOTSTRAP=1`
  - supported architecture name (`LlamaForCausalLM`)
  - `load_format in {'dummy', 'marin_zero_bootstrap'}`
  - no Qwix-on-abstract path
  - no quantization config

Result:

- local tests still pass
- targeted pre-commit checks pass
- this should let the worker process actually install the patch on the cluster package

### Update: submitted v18 after fixing the patch-install failure

Submitted:

- job: `/ahmed/vllm-async-stress-8b-eager-v18`
- same 8B async-native eager stress command as `v17`

Initial state:

- job state: `running`
- task state: `building`
- worker: `marin-tpu_v5p_8-us-central1-a-20260320-0204-ba1b0775-worker-0`

This is the first run that should actually tell us whether the Marin zero-bootstrap path executes in the worker process.

### Update: v18 still failed because the patch depended on another missing TPU helper

`/ahmed/vllm-async-stress-8b-eager-v18` again timed out at 1200s without reaching the zero-bootstrap path.

New blocking log line:

- `WARNING:marin.inference.vllm_tpu_bootstrap_patch: Could not import TPU modules for Marin fast bootstrap patch: cannot import name 'assign_and_shard_param' from 'tpu_inference.models.jax.utils.weight_utils'`

Interpretation:

- `v18` still did not exercise the new path
- the installed TPU package on cluster exposes a narrower API surface than the source checkout under `~/code`
- depending on helper exports from `tpu_inference.models.jax.utils.weight_utils` is the wrong strategy

Fix applied after `v18`:

- removed the remaining TPU helper import dependency from the patch
- implemented Marin-local `_assign_and_shard_param(...)` logic using only JAX sharding primitives and param metadata
- this should make the runtime patch independent of whether the packaged TPU wheel exports `assign_and_shard_param`

### Update: submitted v19 after removing all TPU helper export dependencies

Submitted:

- job: `/ahmed/vllm-async-stress-8b-eager-v19`

Why this run matters:

- the monkeypatch no longer imports any TPU helper symbols beyond the model-loader module itself
- zero-state materialization now uses Marin-local JAX sharding logic
- this should be the first run where helper-export drift in the packaged TPU wheel is no longer a blocker to patch installation

### Update: v19 was the first run where the patch actually activated

`/ahmed/vllm-async-stress-8b-eager-v19` is the first run that proved the Marin zero-bootstrap patch is being executed on the cluster.

Positive signal:

- logs showed `Applying Marin fast TPU bootstrap patch`
- logs showed `Using Marin zero-bootstrap flax_nnx path arch=LlamaForCausalLM load_format='dummy'`

The run then failed fast with a real bug in the patch itself:

- `AttributeError: 'LlamaForCausalLM' object has no attribute 'named_parameters'`

Interpretation:

- the overall monkeypatch seam is correct
- the failure is now inside Marin-owned code, not in dependency import drift
- the next fix is to traverse the NNX model via `nnx.iter_graph(...)` instead of assuming a `named_parameters()` method on `LlamaForCausalLM`

Applied fix after `v19`:

- switched zero-bootstrap parameter discovery to `nnx.iter_graph(...)`
- assign zero state only to `nnx.Param` nodes
- seed `rng` variables when encountered, matching the TPU fork's NNX initialization pattern

### Update: submitted v20 after fixing NNX parameter traversal

Submitted:

- job: `/ahmed/vllm-async-stress-8b-eager-v20`

Goal of this run:

- confirm the zero-bootstrap path gets past NNX parameter discovery
- see whether the next live boundary is zero-state materialization itself, `create_jit_model(...)`, or a later TPU runner step

### Update: v20 reached zero-bootstrap materialization

Current active run:

- job: `/ahmed/vllm-async-stress-8b-eager-v20`
- current known state: `running`

Important new positive signals from logs:

- `Applying Marin fast TPU bootstrap patch`
- `Using Marin zero-bootstrap flax_nnx path arch=LlamaForCausalLM load_format='dummy'`
- `START Marin zero-bootstrap materialization arch=LlamaForCausalLM params=291`

Interpretation:

- the patch installs cleanly
- the monkeypatch seam is correct
- the NNX traversal fix worked
- the live boundary is now the actual zero-state materialization / subsequent JIT handoff, not patch activation or parameter discovery

Preparation for next run if needed:

- added aggregated progress logging inside the zero-bootstrap materialization loop
- this should let the next run distinguish between:
  - one or two huge tensors dominating
  - steady full-model sharding cost
  - later slowdown in `create_jit_model(...)` or cache init

### Update: v20 timed out inside zero-bootstrap materialization

Final result for `/ahmed/vllm-async-stress-8b-eager-v20`:

- job failed with `TimeoutError: In-process vLLM server did not become ready within 1200s`
- the run never advanced beyond:
  - `START Marin zero-bootstrap materialization arch=LlamaForCausalLM params=291`

What this means:

- the patch seam is correct
- the patch installs and reaches the intended fast-bootstrap branch
- the current bottleneck is now the zero-state materialization itself (or the immediate post-materialization JIT/cache step), not TPU loader selection or patch activation

Next hypothesis:

- either zero-state sharding is steadily progressing but too slowly for 1200s
- or one/few huge parameters dominate the time
- or the loop finishes and the slowdown is in `create_jit_model(...)` / `initialize_cache()` immediately after

Next experiment:

- rerun with the newly added zero-bootstrap progress logging
- increase startup timeout to 1800s to capture more signal from the same path

### Update: submitted v21 with zero-bootstrap progress logging

Submitted:

- job: `/ahmed/vllm-async-stress-8b-eager-v21`
- same 8B async-native eager stress command as `v20`, but with `--startup-timeout 1800`

Purpose:

- use the new `zero-bootstrap.progress` / `zero-bootstrap.param-*` logs to determine whether materialization is steadily progressing, dominated by a few tensors, or finishing before a later JIT/cache step

### Update: v21 identified the first dominating zero-bootstrap tensor

Current state for `/ahmed/vllm-async-stress-8b-eager-v21`:

- job is still running
- zero-bootstrap progress logging is active
- current high-signal line:
  - `zero-bootstrap.param-start index=1/291 name=model.embed.embedding numel=525336576`

Interpretation:

- the progress instrumentation works
- the first parameter alone is enormous (~525M elements)
- the immediate question is whether `model.embed.embedding` eventually completes in a reasonable time or effectively dominates the whole bootstrap budget

Next short-term action:

- wait another 10 minutes on `v21`
- check whether `zero-bootstrap.param-end` for `model.embed.embedding` appears
- if it does not, the next experiment should target that parameter specifically rather than the whole generic materialization loop

### Update: after 10 more minutes, v21 is still on the first tensor

Follow-up check on `/ahmed/vllm-async-stress-8b-eager-v21`:

- still no `zero-bootstrap.param-end` for the first parameter
- current live tensor remains:
  - `index=1/291 name=model.embed.embedding numel=525336576`

Interpretation:

- the bootstrap does not appear to be dominated by many small parameters
- the first giant embedding tensor is likely the main cost driver
- the next design change is likely to target bulk/sharded creation of very large tensors rather than incremental per-parameter `device_put`

### Update: v21 confirmed the first embedding tensor is effectively wedged

Final result for `/ahmed/vllm-async-stress-8b-eager-v21`:

- job failed with `TimeoutError: In-process vLLM server did not become ready within 1800s`
- the run never emitted `zero-bootstrap.param-end` for:
  - `index=1/291 name=model.embed.embedding numel=525336576`
- no later `zero-bootstrap.progress` or `END Marin zero-bootstrap materialization` lines appeared

Interpretation:

- the current Marin zero-bootstrap implementation is not merely "slow overall"
- it is getting stuck or spending nearly the entire startup budget on the first giant embedding parameter
- the problem is specifically the current per-parameter materialization path:
  - `jnp.zeros(shape, dtype)`
  - followed by per-parameter assignment / sharding
- that points to the large-tensor creation path itself, not the patch seam, not later JIT/cache wiring, and not later streamed weight injection

Next experiment:

- replace the Python per-parameter zero-materialization loop with a single jitted whole-state materialization built from the abstract NNX state
- use `nnx.map_state(...)` + `jax.jit(..., out_shardings=nnx.get_partition_spec(state))`
- this should let JAX create and shard the full zero state in one program rather than compiling / dispatching tensor-by-tensor

Reasoning:

- `v21` gave enough evidence that waiting longer on the current loop will not buy new information
- if the whole-state JIT path materially improves startup, the bottleneck was the incremental tensor creation path
- if it still times out, the next live boundary will narrow to `create_jit_model(...)` / cache init or to the whole-state JIT itself

### Update: replaced per-parameter zero loop with whole-state JIT materialization

Code change applied after `v21`:

- removed the Python per-parameter zero/bootstrap loop from `lib/marin/src/marin/inference/vllm_tpu_bootstrap_patch.py`
- replaced it with whole-state materialization from the abstract NNX state:
  - `abstract_state = nnx.state(model)`
  - `state_shardings = nnx.get_partition_spec(abstract_state)`
  - `nnx.map_state(...)` to replace abstract values with zero arrays / seeded RNG keys
  - `jax.jit(..., out_shardings=state_shardings)` to materialize and shard the full state in one program
- split logs into two explicit boundaries:
  - `START/END Marin zero-bootstrap state materialization`
  - `START/END Marin zero-bootstrap create_jit_model`

Why this is the right next cut:

- `v21` showed the first giant embedding tensor never finished under the incremental loop
- the likely bad behavior was per-tensor zero creation / transfer / dispatch, not the overall monkeypatch seam
- whole-state JIT gives JAX one chance to build and shard the state instead of walking 291 params from Python

Local validation:

- direct CPU/JAX smoke check of `_materialize_zero_state_tree(...)` succeeded
- `pytest -q tests/vllm/test_vllm_inprocess_backend.py` passed (`20 passed`)
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_tpu_bootstrap_patch.py tests/vllm/test_vllm_inprocess_backend.py` passed

### Update: submitting v22 for whole-state materialization

Submitted next run:

- job: `/ahmed/vllm-async-stress-8b-eager-v22`
- same async-native 8B eager stress command as `v21`
- startup timeout reduced back to `1200` to shorten iteration time now that the implementation changed substantially

Key expected signals:

- if `END Marin zero-bootstrap state materialization` appears quickly, the per-parameter loop was the core problem
- if it hangs before that line, the whole-state JIT itself is now the live bottleneck
- if it gets past state materialization and then stalls, the next boundary is `create_jit_model(...)` / `initialize_cache()`

### Update: v22 reached the new whole-state materialization boundary

Current state for `/ahmed/vllm-async-stress-8b-eager-v22` after the first 10-minute check:

- job is still `running`
- there was one worker retry due to infra timeout; current attempt is healthy
- high-signal logs now show the new boundary:
  - `Using Marin zero-bootstrap flax_nnx path arch=LlamaForCausalLM load_format='dummy'`
  - `START Marin zero-bootstrap state materialization arch=LlamaForCausalLM params=291 numel=8030261248`

Interpretation:

- the code path changed exactly as intended
- we are no longer in the old per-parameter `zero-bootstrap.param-start ... model.embed.embedding` loop
- the live question is now whether the whole-state JIT materialization finishes within the startup budget or itself becomes the new long pole

Immediate action:

- wait another 10 minutes on `v22`
- if `END Marin zero-bootstrap state materialization` appears, the next live boundary becomes `create_jit_model(...)`
- if it does not, the whole-state JIT is still too expensive and the next experiment should target a lower-level lazy or staged state bootstrap

### Update: v22 proved whole-state materialization is still too expensive

Final result for `/ahmed/vllm-async-stress-8b-eager-v22`:

- job failed with `TimeoutError: In-process vLLM server did not become ready within 1200s`
- the run entered:
  - `START Marin zero-bootstrap state materialization arch=LlamaForCausalLM params=291 numel=8030261248`
- it never emitted:
  - `END Marin zero-bootstrap state materialization`
  - `START Marin zero-bootstrap create_jit_model`
- so the whole-state JIT change did move the code path, but it still did not finish within the startup budget

Important interpretation update:

- the dominant problem is not just Python per-parameter overhead anymore
- any startup path that fully materializes the 8B parameter state inside `load_model()` appears too expensive for async-native startup on this TPU path
- that means the next viable experiment is no longer "faster zero creation"
- the next viable experiment is an **abstract-state bootstrap** that returns the NNX model/state structure without allocating the full parameter buffers during `load_model()`

Why this is plausible:

- `get_flax_model(...)` ultimately splits the returned NNX model into `graphdef` and `state`
- `_sync_weights(...)` only needs target leaf shapes/dtypes and can replace target leaves with concrete sharded arrays later
- TPU `determine_available_memory()` is cheap in this fork; it just reads HBM usage
- normal engine init does not call `compile_or_warm_up_model()` in this path

So the next experiment is:

- patch the Marin fast-bootstrap branch to return the abstract NNX model directly from `nnx.eval_shape(...)`
- seed RNG keys only
- skip all dummy zero-state materialization during `load_model()`
- then let later streamed weight injection concretize the state

Risk:

- if some later engine-init step implicitly requires concrete parameter arrays even before inference, this will fail differently
- but that is still the right next cut because `v22` established that full-state materialization at startup is a dead end

### Update: submitted v23 for abstract-state bootstrap

Submitted:

- job: `/ahmed/vllm-async-stress-8b-eager-v23`
- same 8B async-native eager stress command as `v22`
- still using `--startup-timeout 1200` so the next failure, if any, is bounded tightly

Code change behind `v23`:

- the Marin fast-bootstrap branch now returns the abstract NNX model from `nnx.eval_shape(...)`
- it seeds RNG keys but does **not** allocate or materialize the full parameter state during `load_model()`
- local validation passed (`21 passed` + targeted pre-commit)

Expected signals:

- if `START/END Marin abstract-state bootstrap` appears quickly and engine init progresses, then the previous blocker really was full-state materialization during `load_model()`
- if engine init now fails in a later step, we will finally have moved the live boundary beyond model construction
- if even abstract bootstrap fails, then some later assumption in `get_flax_model(...)` / engine init requires concrete arrays earlier than expected

### Update: v23 broke through the `load_model()` bottleneck

First high-signal result from `/ahmed/vllm-async-stress-8b-eager-v23`:

- the new abstract-state path activated correctly:
  - `Using Marin abstract-state flax_nnx path arch=LlamaForCausalLM load_format='dummy'`
- bootstrap completed quickly:
  - `START Marin abstract-state bootstrap ...`
  - `END Marin abstract-state bootstrap ... elapsed=2.22s`
  - `END Marin fast-bootstrap model prep ... elapsed=2.22s`
  - `END model_loader.get_flax_model in 3.00s`
  - `END tpu_runner.get_model in 3.00s`

Interpretation:

- this is the first successful elimination of the `load_model()` startup bottleneck
- the previous dead ends were all caused by forcing full-state materialization during model construction
- abstract-state bootstrap is the first Marin-only patch that preserves async-native serving while making model construction fast enough to move the live boundary forward

Immediate next action:

- keep `v23` running
- wait another 10 minutes and inspect the next boundary after `load_model()`
- the likely remaining candidates are now:
  - `TPUWorker.load_model` completion / wrapper return
  - `EngineCore._initialize_kv_caches`
  - `initialize_from_config(...)` / KV cache allocation
  - some later server app init before `/v1/models` becomes ready

### Update: v23 final outcome and current narrowest boundary

Final result for `/ahmed/vllm-async-stress-8b-eager-v23`:

- job still failed with `TimeoutError: In-process vLLM server did not become ready within 1200s`
- but the important boundary moved dramatically:
  - `END Marin abstract-state bootstrap ... elapsed=2.22s`
  - `END Marin fast-bootstrap model prep ... elapsed=2.22s`
  - `END model_loader.get_flax_model in 3.00s`
  - `END tpu_runner.get_model in 3.00s`
- after that, no later `END TPUModelRunner.load_model` or `END TPUWorker.load_model` lines appeared before timeout

What this means:

- the abstract-state bootstrap solved the original `load_model()` bottleneck in `get_flax_model(...)`
- the current live bottleneck is now **inside the tail of `TPUModelRunner.load_model()` after `tpu_runner.get_model(...)` has already returned**
- that is a much smaller target than before; only a few statements remain there:
  - multimodal function binding
  - optional `drafter.load_model(self.state)`
  - `nnx.Rngs(jax.random.key(...)).params()` for sampling RNG setup
  - `is_multimodal_model` flag derivation
  - final HBM log line

Current next action:

- patch detailed post-`get_model` instrumentation directly into `TPUModelRunner.load_model()` in Marin’s worker monkeypatch path
- rerun as `v24`
- use that run to identify the exact statement that still holds engine init open after abstract-state bootstrap

Status of local code at this moment:

- the detailed `TPUModelRunner.load_model()` instrumentation patch is in progress
- tests still pass (`21 passed`)
- only formatter cleanup remains before submitting the next Iris run

### Update: prepared v24 with detailed post-`get_model` instrumentation

Local work completed after `v23`:

- added detailed instrumentation inside Marin's monkeypatched `TPUModelRunner.load_model()` wrapper
- the new logs split the tail after `tpu_runner.get_model(...)` into explicit substeps:
  - `TPUModelRunner.load_model.after_get_model`
  - `TPUModelRunner.load_model.after_multimodal_bind`
  - `TPUModelRunner.load_model.drafter`
  - `TPUModelRunner.load_model.rng_params_for_sampling`
  - `TPUModelRunner.load_model.is_multimodal_model`
  - `TPUModelRunner.load_model.log_init_model`
- local validation passed:
  - `pytest -q tests/vllm/test_vllm_inprocess_backend.py`
  - `./infra/pre-commit.py lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py tests/vllm/test_vllm_inprocess_backend.py`

Purpose of next run:

- identify the exact post-`get_model` statement that still blocks `TPUModelRunner.load_model()` after abstract-state bootstrap solved the earlier model-construction bottleneck

### Update: switching to a direct `TPUModelRunner.load_model` tail patch for v25

New hypothesis after `v24`:

- the detailed post-`get_model` substep logs did not appear at all
- that means the nested wrapper strategy for `TPUModelRunner.load_model` was not reliable enough to trust as a diagnostic tool in the worker process

Suggested change applied:

- replaced the nested `TPUModelRunner.load_model` timing scheme with one direct monkeypatched method that owns:
  - top-level `START/END TPUModelRunner.load_model`
  - `TPUModelRunner.load_model.get_model_tuple`
  - `TPUModelRunner.load_model.multimodal_bind`
  - `TPUModelRunner.load_model.drafter`
  - `TPUModelRunner.load_model.rng_params_for_sampling`
  - `TPUModelRunner.load_model.is_multimodal_model`
  - `TPUModelRunner.load_model.log_init_model`
- added an install marker:
  - `installed detailed TPUModelRunner.load_model instrumentation`

Local validation:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py` passed (`21 passed`)
- `python -m black ...` + `./infra/pre-commit.py ...` passed

Next command:

- submit `/ahmed/vllm-async-stress-8b-eager-v25`
- purpose: identify the exact tail statement that still blocks startup after the abstract-state bootstrap breakthrough

### Update: adding engine-process `faulthandler` for v26

New hypothesis after `v25`:

- the direct `TPUModelRunner.load_model` tail patch still did not surface the new substep labels in the worker logs
- so the remaining blocker is likely still in Python code after `tpu_runner.get_model(...)`, but the monkeypatch boundary itself is not trustworthy enough as a diagnostic mechanism

Suggested change applied:

- added `MARIN_VLLM_STARTUP_FAULTHANDLER=1` and `MARIN_VLLM_STARTUP_FAULTHANDLER_SECS=300` defaults in `vllm_async.py`
- enabled periodic `faulthandler.dump_traceback_later(..., repeat=True)` in the worker startup path when startup timing is enabled
- this should print full Python stack traces from the engine process to stderr every 300s while startup is hung

Local validation:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py` passed (`21 passed`)
- `python -m black ...` + `./infra/pre-commit.py ...` passed

Next command:

- submit `/ahmed/vllm-async-stress-8b-eager-v26`
- goal: capture a live engine-process traceback during startup hang so the next hypothesis is based on the exact blocked line rather than the last timing boundary

### Update: moving detailed tail instrumentation into the early async startup path for v27

Latest finding from `v26`:

- the remote logs still showed only the early timing wrappers from `vllm_async._install_early_async_startup_instrumentation()`
- none of the deeper `worker.py` tail-step labels or faulthandler markers appeared in the engine logs
- but the key boundary stayed stable:
  - `END model_loader.get_flax_model in 2.92s`
  - `END tpu_runner.get_model in 2.92s`
  - then no later `END TPUModelRunner.load_model` before timeout

Hypothesis:

- the only instrumentation path that the cluster is reliably surfacing is the early installer in `vllm_async.py`
- so the next useful cut is to move the detailed `TPUModelRunner.load_model` tail wrapper and faulthandler enablement into that early installer, instead of relying on the worker-side monkeypatch path

Suggested change applied:

- added early faulthandler enablement directly in `vllm_async.py`
- replaced the generic early `TPUModelRunner.load_model` timer with a detailed wrapper in `vllm_async.py` that logs:
  - `TPUModelRunner.load_model.get_model_tuple`
  - `TPUModelRunner.load_model.multimodal_bind`
  - `TPUModelRunner.load_model.drafter`
  - `TPUModelRunner.load_model.rng_params_for_sampling`
  - `TPUModelRunner.load_model.is_multimodal_model`
  - `TPUModelRunner.load_model.log_init_model`
- added a local regression test for the early detailed wrapper

Local validation:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py` passed (`22 passed`)
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py tests/vllm/test_vllm_inprocess_backend.py` pending re-run after removing an unused import caught by Ruff

Command run next:

- submit `/ahmed/vllm-async-stress-8b-eager-v27`
- purpose: prove the early installer now owns the visible post-`get_model` tail logs and identify the exact remaining blocking statement after `tpu_runner.get_model(...)` returns

Follow-through:

- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py tests/vllm/test_vllm_inprocess_backend.py` passed after removing the unused import
- submitted `/ahmed/vllm-async-stress-8b-eager-v27`

### Update: v27 converted the startup mystery into a concrete tuple-contract bug

Result for `/ahmed/vllm-async-stress-8b-eager-v27`:

- the new early instrumentation path finally showed the detailed tail logs from `TPUModelRunner.load_model`
- the engine did **not** hang after `tpu_runner.get_model(...)`
- instead it failed immediately after `get_model(...)` returned

High-signal sequence:

- `installed detailed early TPUModelRunner.load_model instrumentation`
- `START TPUModelRunner.load_model.get_model_tuple`
- `END model_loader.get_flax_model in 3.10s`
- `END tpu_runner.get_model in 3.10s`
- `FAIL TPUModelRunner.load_model.get_model_tuple in 3.10s`
- traceback root cause:
  - `ValueError: not enough values to unpack (expected 8, got 7)`

What this means:

- the abstract-state fast bootstrap is working well enough to return from `get_flax_model(...)` quickly
- the `v27` failure was caused by my diagnostic wrapper assuming the newer 8-value `get_model()` contract
- the cluster's installed `tpu_inference` build is using the older 7-value contract
- so `v27` surfaced a **real package-version skew** between my reference checkout and the cluster wheel, not a new Marin bootstrap bug

Hypothesis:

- once the early diagnostic wrapper accepts both 7-value and 8-value `get_model()` tuples, the next run will move past `TPUModelRunner.load_model.get_model_tuple`
- that should finally reveal the real post-`get_model` blocker, if any still exists

Suggested change applied:

- made the early `TPUModelRunner.load_model` diagnostic wrapper in `vllm_async.py` accept both tuple lengths:
  - `8` values -> newer contract with `combine_hidden_states_fn`
  - `7` values -> legacy contract; set `self.combine_hidden_states_fn = None`
- added a log line for the observed tuple length
- added a regression test covering the legacy 7-tuple contract

Local validation:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py` passed (`23 passed`)
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py tests/vllm/test_vllm_inprocess_backend.py` pending current run

Command run next:

- submit `/ahmed/vllm-async-stress-8b-eager-v28`
- purpose: get past the instrumentation-induced tuple mismatch and identify the next actual startup boundary after `TPUModelRunner.load_model.get_model_tuple`

Follow-through:

- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py tests/vllm/test_vllm_inprocess_backend.py` passed
- submitted `/ahmed/vllm-async-stress-8b-eager-v28`

### Update: v28 narrowed the live blocker to sampling-RNG initialization

Result for `/ahmed/vllm-async-stress-8b-eager-v28`:

- the legacy 7-tuple compatibility patch worked
- startup moved past:
  - `TPUModelRunner.load_model.get_model_tuple`
  - `TPUModelRunner.load_model.multimodal_bind`
- and then stalled reproducibly at:
  - `START TPUModelRunner.load_model.rng_params_for_sampling`

Important details:

- no `END TPUModelRunner.load_model.rng_params_for_sampling` ever appeared
- after 5 minutes, the parent-process faulthandler fired, but it only showed the main thread waiting on engine readiness; it did not surface a child stack deeper than the last timing boundary
- local CPU behavior shows `jax.random.key(seed)` and `nnx.Rngs(...).params()` are normally trivial, so a multi-minute TPU stall there is abnormal

Hypothesis:

- the TPU runner's use of `nnx.Rngs(jax.random.key(seed)).params()` is the next incompatible operation on this fast-bootstrap path
- later sampling only needs a `PRNGKeyArray` that can be passed to `jax.random.split`, so a direct `jax.random.key(seed)` should be sufficient and may avoid the stall entirely

Suggested change applied:

- added `MARIN_VLLM_DIRECT_SAMPLING_KEY=1` as the async-native default
- changed the early monkeypatched `TPUModelRunner.load_model` path to:
  - use `jax.random.key(self.model_config.seed)` directly when that env is enabled
  - retain the old `nnx.Rngs(...).params()` path behind the switch for comparison
- added logging for the selected RNG init mode
- added local regression coverage for the direct-key path

Local validation:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py` passed (`24 passed`)
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py tests/vllm/test_vllm_inprocess_backend.py` pending current run

Command run next:

- submit `/ahmed/vllm-async-stress-8b-eager-v29`
- purpose: determine whether bypassing `nnx.Rngs(...).params()` lets startup proceed into `is_multimodal_model`, `log_init_model`, KV-cache init, or weight injection

Follow-through:

- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py tests/vllm/test_vllm_inprocess_backend.py` passed
- terminated `/ahmed/vllm-async-stress-8b-eager-v28` once the live blocker was isolated at `rng_params_for_sampling`
- submitted `/ahmed/vllm-async-stress-8b-eager-v29`

### Update: v29 cleared the sampling-RNG stall and moved startup into KV-cache init

Live result so far for `/ahmed/vllm-async-stress-8b-eager-v29`:

- the direct-key sampling RNG path worked immediately
- startup now gets all the way through `TPUModelRunner.load_model()`
- the next visible phase is now KV-cache initialization

High-signal sequence:

- `TPUModelRunner.load_model.rng_params_for_sampling mode=direct_key`
- `END TPUModelRunner.load_model.rng_params_for_sampling in 0.00s`
- `END TPUModelRunner.load_model in 3.01s`
- `END TPUWorker.load_model in 3.01s`
- `END WorkerWrapperBase.load_model in 3.01s`
- `END UniProcExecutor._init_executor in 26.10s`
- `START EngineCore._initialize_kv_caches`
- `START WorkerWrapperBase.determine_available_memory`
- `END WorkerWrapperBase.determine_available_memory in 0.00s`
- `START WorkerWrapperBase.initialize_from_config`

What this means:

- the previous stall at `rng_params_for_sampling` was real and the direct-key workaround is effective
- the async-native startup path is now past:
  - model construction
  - get_model return-shape mismatch
  - sampling RNG init
- the new narrowest live boundary is inside KV-cache / runner initialization after `EngineCore._initialize_kv_caches`

Next immediate action:

- keep `v29` running
- wait for the next visible `initialize_from_config` / KV-cache boundary before changing code again

### Update: v29 moved the bottleneck from load_model to KV-cache initialization

Result for `/ahmed/vllm-async-stress-8b-eager-v29`:

- the direct-key workaround fixed the `rng_params_for_sampling` stall
- startup now completes the entire `TPUModelRunner.load_model()` phase quickly
- the new live boundary is inside KV-cache initialization

High-signal sequence:

- `END TPUModelRunner.load_model in 3.01s`
- `END TPUWorker.load_model in 3.01s`
- `END WorkerWrapperBase.load_model in 3.01s`
- `END UniProcExecutor._init_executor in 26.10s`
- `START EngineCore._initialize_kv_caches`
- `START WorkerWrapperBase.determine_available_memory`
- `END WorkerWrapperBase.determine_available_memory in 0.00s`
- `START WorkerWrapperBase.initialize_from_config`
- then no later completion line before I terminated the job to deepen the instrumentation

Hypothesis:

- the next long pole is inside the actual KV-cache allocation path, not in model loading anymore
- on the TPU side, that path is:
  - `TPUWorker.initialize_from_config`
  - `TPUModelRunner.initialize_kv_cache`
  - `KVCacheManager.initialize_kv_cache`
  - `kv_cache.create_kv_caches`

Suggested change applied:

- added early async instrumentation for:
  - `TPUWorker.initialize_from_config`
  - `TPUModelRunner.initialize_kv_cache`
  - `KVCacheManager.initialize_kv_cache`
  - `kv_cache.create_kv_caches`

Local validation:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py` passed (`24 passed`)
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py tests/vllm/test_vllm_inprocess_backend.py` pending current run

Command run next:

- submit `/ahmed/vllm-async-stress-8b-eager-v30`
- purpose: isolate whether the new long pole is worker-side config init, model-runner KV-cache init, manager-level allocation, or the underlying `create_kv_caches` JAX allocation call

Follow-through:

- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py tests/vllm/test_vllm_inprocess_backend.py` passed
- submitted `/ahmed/vllm-async-stress-8b-eager-v30`

### Update: v30 isolated the new boundary inside KVCacheManager.initialize_kv_cache

Result for `/ahmed/vllm-async-stress-8b-eager-v30`:

- the new KV-cache timers fired correctly
- startup moved through:
  - `TPUWorker.initialize_from_config`
  - `TPUModelRunner.initialize_kv_cache`
  - `KVCacheManager.initialize_kv_cache`
- but it did **not** reach the first visible `kv_cache.create_kv_caches` call before I terminated it

High-signal sequence:

- `START WorkerWrapperBase.initialize_from_config`
- `START TPUWorker.initialize_from_config`
- `START TPUModelRunner.initialize_kv_cache`
- `START KVCacheManager.initialize_kv_cache`
- then no later `create_kv_caches` or completion line

Hypothesis:

- the next long pole is most likely in `KVCacheManager.maybe_reinitialize_input_batch(...)`, which runs before `create_kv_caches(...)`
- if that is not it, then the remaining gap is the manager's layer/spec bookkeeping loop immediately before the first cache allocation call

Suggested change applied:

- added early instrumentation for `KVCacheManager.maybe_reinitialize_input_batch`

Local validation:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py` passed (`24 passed`)
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py tests/vllm/test_vllm_inprocess_backend.py` pending current run

Command run next:

- submit `/ahmed/vllm-async-stress-8b-eager-v31`
- purpose: determine whether the new long pole is the input-batch reinitialization helper or the remaining manager logic before the first `create_kv_caches` call

### Update: v31 proved the KV-cache stall happens after maybe_reinitialize_input_batch

Result for `/ahmed/vllm-async-stress-8b-eager-v31`:

- the added helper timer fired
- `KVCacheManager.maybe_reinitialize_input_batch` returned immediately
- startup still stalled inside `KVCacheManager.initialize_kv_cache` before the first visible `create_kv_caches` call

High-signal sequence:

- `START KVCacheManager.initialize_kv_cache`
- `START KVCacheManager.maybe_reinitialize_input_batch`
- `END KVCacheManager.maybe_reinitialize_input_batch in 0.00s`
- then no `START kv_cache.create_kv_caches` before the 1200s startup timeout

Hypothesis:

- the remaining long pole is in the manager's tensor/spec bookkeeping immediately before the first `create_kv_caches` invocation
- the most useful next cut is a detailed wrapper around `KVCacheManager.initialize_kv_cache` itself

Suggested change applied:

- replaced the generic `KVCacheManager.initialize_kv_cache` timer with a detailed wrapper that logs:
  - `layer_name_to_spec`
  - per-tensor shape/block metadata
  - `tensor_create` before and after each `create_kv_caches` call
  - shared-KV-layer mapping

Local validation:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py` passed (`24 passed`)
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py tests/vllm/test_vllm_inprocess_backend.py` pending current run

Command run next:

- submit `/ahmed/vllm-async-stress-8b-eager-v32`
- purpose: identify whether the remaining long pole is layer/spec bookkeeping or the first actual `create_kv_caches` JAX allocation

### Update: v32 is cluster-gated, not code-gated

Current state for `/ahmed/vllm-async-stress-8b-eager-v32`:

- job is `pending`
- pending reason: insufficient `v5p-8` TPU capacity in `us-central1-a`
- no runtime logs yet, so this is not a new code result

Immediate action:

- wait 10 minutes and re-check `v32`
- do not patch code again until the queued run either starts or the capacity situation changes

### Update: v31 final result and v32 queue state

Final result for `/ahmed/vllm-async-stress-8b-eager-v31`:

- `KVCacheManager.maybe_reinitialize_input_batch` returned immediately
- the startup timeout still fired before readiness
- the live boundary remained inside `KVCacheManager.initialize_kv_cache`
- specifically, there was still no first `create_kv_caches` call before timeout

High-signal sequence:

- `START KVCacheManager.initialize_kv_cache`
- `START KVCacheManager.maybe_reinitialize_input_batch`
- `END KVCacheManager.maybe_reinitialize_input_batch in 0.00s`
- then no `START kv_cache.create_kv_caches`
- then `TimeoutError: In-process vLLM server did not become ready within 1200s`

Interpretation:

- the new long pole is after input-batch reinit and before first KV allocation
- this is consistent with the manager-level bookkeeping / first-tensor setup gap

Current state for `/ahmed/vllm-async-stress-8b-eager-v32`:

- job is still `pending`
- pending reason: insufficient `v5p-8` TPU capacity in `us-central1-a`
- no runtime evidence yet from the new detailed manager wrapper

Immediate action:

- wait another 10 minutes and re-check `v32`
- keep the code unchanged until the queued run either starts or TPU capacity remains blocked long enough to justify changing cluster strategy

### Update: v32 isolated the first KV allocation as the new long pole

Result for `/ahmed/vllm-async-stress-8b-eager-v32`:

- cluster capacity cleared and the run started normally
- the Marin abstract-state bootstrap remained fixed:
  - `END model_loader.get_flax_model in 3.01s`
  - `END TPUModelRunner.load_model in 3.01s`
- KV-cache instrumentation then narrowed the remaining startup stall to the first real cache allocation

High-signal sequence:

- `START KVCacheManager.initialize_kv_cache`
- `START KVCacheManager.maybe_reinitialize_input_batch`
- `END KVCacheManager.maybe_reinitialize_input_batch in 0.00s`
- `START KVCacheManager.initialize_kv_cache.layer_name_to_spec`
- `END KVCacheManager.initialize_kv_cache.layer_name_to_spec in 0.00s ... groups=1 layer_specs=32`
- `KVCacheManager.initialize_kv_cache.tensor index=1/32 layer=layer.0 num_blocks=2757 block_size=256 num_kv_heads=8 head_size=128`
- `START KVCacheManager.initialize_kv_cache.tensor_create index=1/32`
- then no completion line before the faulthandler dump

Interpretation:

- the model-loader/bootstrap problem is no longer the blocker
- the new long pole is inside the first `create_kv_caches(...)` call itself
- likely candidates are:
  - `get_kv_cache_shape_with_mesh(...)`
  - `NamedSharding(...)` / sharding setup
  - `jax.jit(_allocate, out_shardings=...)`
  - the first `sharded_allocate()` execution

HYPOTHESIS:

- the dominant latency is in the first KV tensor creation path, not in manager bookkeeping anymore
- the next useful discriminator is to split `create_kv_caches(...)` into shape, sharding, JIT construction, and per-layer allocate timings

SUGGESTED CHANGE:

- replace the generic `kv_cache.create_kv_caches` timer with a detailed wrapper that logs:
  - `shape`
  - `sharding`
  - `make_jit`
  - each `allocate index=i/n`

COMMAND RUN:

- validate locally and submit `/ahmed/vllm-async-stress-8b-eager-v33`

Validation for the `v33` patch:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py` passed (`25 passed`)
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py tests/vllm/test_vllm_inprocess_backend.py` passed

COMMAND RUN:

- submitted `/ahmed/vllm-async-stress-8b-eager-v33`
- purpose: split `create_kv_caches(...)` into `shape`, `sharding`, `make_jit`, and `allocate index=i/n` so the next boundary is exact

### Update: v33 proved the `create_kv_caches` patch missed the live alias

Result for `/ahmed/vllm-async-stress-8b-eager-v33`:

- startup again reached:
  - `START KVCacheManager.initialize_kv_cache.tensor_create index=1/32`
- but there were still no `START kv_cache.create_kv_caches...` substep logs before the faulthandler dump

Interpretation:

- the new detailed wrapper itself did not fire on the live call path
- inspecting `tpu_inference.runner.kv_cache_manager` showed why:
  - it does `from tpu_inference.runner.kv_cache import create_kv_caches`
  - so its runtime call resolves a module-global alias, not `kv_cache.create_kv_caches` directly
- this means `v33` did not invalidate the KV-allocation hypothesis; it only exposed an aliasing miss in the instrumentation

HYPOTHESIS:

- once the `kv_cache_manager.create_kv_caches` alias is redirected to the detailed wrapper, the next run will show whether the long pole is in:
  - `shape`
  - `sharding`
  - `make_jit`
  - or the first `allocate index=1/1`

SUGGESTED CHANGE:

- after installing the detailed `kv_cache.create_kv_caches` wrapper, also overwrite `kv_cache_manager_module.create_kv_caches` with the wrapped function

COMMAND RUN:

- validate locally and submit `/ahmed/vllm-async-stress-8b-eager-v34`

Validation for the `v34` patch:

- `pytest -q tests/vllm/test_vllm_inprocess_backend.py` passed (`25 passed`)
- `./infra/pre-commit.py lib/marin/src/marin/inference/vllm_async.py tests/vllm/test_vllm_inprocess_backend.py` passed

COMMAND RUN:

- submitted `/ahmed/vllm-async-stress-8b-eager-v34`
- purpose: confirm the detailed `create_kv_caches(...)` wrapper now covers the live `kv_cache_manager` alias path

### Update: v34 isolated eager KV allocation volume as the startup bottleneck

Result for `/ahmed/vllm-async-stress-8b-eager-v34`:

- the detailed `create_kv_caches(...)` wrapper finally covered the live path
- startup reached:
  - `START kv_cache.create_kv_caches`
  - `END kv_cache.create_kv_caches.shape in 0.00s shape=(2757, 256, 8, 2, 128)`
  - `END kv_cache.create_kv_caches.sharding in 0.00s`
  - `END kv_cache.create_kv_caches.make_jit in 0.00s`
  - `START kv_cache.create_kv_caches.allocate index=1/1`
- then it stalled there until the faulthandler dump

Interpretation:

- the remaining long pole is not model loading anymore
- it is the first actual execution of `sharded_allocate()` for the KV cache tensor
- for the current 8B/4096 configuration, each layer KV tensor is ~2.69 GiB and 32 layers total ~86.16 GiB, which matches the observed HBM cap almost exactly
- this strongly suggests async-native startup is now dominated by eager full-KV allocation at startup, not by Marin's fast model bootstrap

HYPOTHESIS:

- lowering `gpu_memory_utilization` should reduce `num_blocks`, shrink the KV tensor, and materially reduce or eliminate the startup stall
- if that works, the next design decision is whether async-native should intentionally reserve less KV cache at startup or whether a deeper lazy-KV strategy is needed

SUGGESTED CHANGE:

- add `--gpu-memory-utilization` to `experiments/inference/exp_vllm_stress_test.py`
- submit a reduced-cache run next

COMMAND RUN:

- validate locally and submit `/ahmed/vllm-async-stress-8b-eager-v35` with a lower `--gpu-memory-utilization`

Validation for the `v35` experiment setup:

- `./infra/pre-commit.py experiments/inference/exp_vllm_stress_test.py lib/marin/src/marin/inference/vllm_async.py tests/vllm/test_vllm_inprocess_backend.py` passed
- `python experiments/inference/exp_vllm_stress_test.py --help` shows the new `--gpu-memory-utilization` flag

COMMAND RUN:

- submitted `/ahmed/vllm-async-stress-8b-eager-v35`
- config: same 8B async-native run, but with `--gpu-memory-utilization 0.25`
- purpose: test whether reducing eager KV allocation size eliminates the startup stall at `sharded_allocate()`

### Update: v35 weakened the raw-HBM-size hypothesis and pointed at first-call compile

Result for `/ahmed/vllm-async-stress-8b-eager-v35` with `--gpu-memory-utilization 0.25`:

- the reduced-capacity setting took effect:
  - `total_hbm_limit_cap_gb=23.94GiB`
  - `GPU KV cache size: 195,840 tokens`
  - `num_blocks` dropped from `2757` to `765`
- despite that large reduction, startup still stalled at exactly the same place:
  - `START kv_cache.create_kv_caches.allocate index=1/1`
- no completion line appeared before the 5-minute faulthandler dump

Interpretation:

- reducing the KV allocation size by roughly 3.6x did **not** eliminate the first-allocation stall
- that makes a pure "too many bytes to allocate" explanation less convincing
- the stronger remaining hypothesis is that the first `sharded_allocate()` call is paying JAX/XLA compile cost, and `make_jit` is cheap only because compilation is lazy until first execution

HYPOTHESIS:

- a second run with the *same* reduced-KV configuration may hit the JAX compilation cache and get past the first `allocate` quickly
- if it does, the remaining issue is cache warmup strategy; if it does not, then the allocation itself remains too expensive and we need a different KV-init strategy

SUGGESTED CHANGE:

- no code change for the next experiment
- rerun the exact same `gpu_memory_utilization=0.25` configuration to test warm-cache behavior directly

COMMAND RUN:

- submit `/ahmed/vllm-async-stress-8b-eager-v36` with the same flags as `v35`

COMMAND RUN:

- submitted `/ahmed/vllm-async-stress-8b-eager-v36`
- config: identical to `v35` (`--gpu-memory-utilization 0.25`)
- purpose: test whether the first `sharded_allocate()` becomes fast on a warm compilation cache

### Update: v36 is still in progress after a worker retry

Current state for `/ahmed/vllm-async-stress-8b-eager-v36`:

- first worker attempt failed with `Request timed out`
- Iris retried the task on a fresh `v5p-8` worker
- the compile-cache experiment is still valid because `JAX_COMPILATION_CACHE_DIR` points to Marin's shared compilation cache path

Immediate action:

- wait another 10 minutes and inspect the second attempt before making any further code changes

### Update: v36 did not materially benefit from warm compile cache

Result for `/ahmed/vllm-async-stress-8b-eager-v36`:

- two early worker attempts were lost to worker request timeouts
- the final non-preempted attempt reached the same reduced-KV configuration as `v35`
- startup still stalled at the first `kv_cache.create_kv_caches.allocate index=1/1`
- that call finally returned only after `1145.76s`, which is effectively the full `1200s` startup budget
- the server still failed with `TimeoutError: In-process vLLM server did not become ready within 1200s`

Interpretation:

- the warm-cache rerun did not make the first KV allocation cheap enough to matter
- this weakens the compile-cache hypothesis substantially
- the remaining blocker is now best described as: TPU async-native startup spends ~19 minutes in the first KV-cache allocation itself, even after fast model bootstrap and reduced `gpu_memory_utilization`
- at this point, more loader work is no longer on the critical path; the next meaningful work should target KV-cache initialization strategy rather than model loading

Next-step direction:

- stop iterating on model bootstrap for now
- investigate whether TPU KV caches can be initialized more lazily or with a reduced/minimal bootstrap path, analogous to what GPU V1 does with temporary minimal allocations in some code paths

### Assessment: most promising next direction after v36

Most promising path:

- explicitly control TPU KV cache bootstrap size rather than continuing to optimize model loading
- the strongest concrete mechanism is `num_gpu_blocks_override`, not more bootstrap loader work

Why this is the current best bet:

- `v34` and `v35` showed the long pole is the first KV allocation, not model bootstrap
- `v36` showed warm-cache behavior still leaves the first KV allocation at ~1146s, so compile-cache alone is not a real solution
- upstream V1 GPU code already uses a temporary minimal KV cache bootstrap for profiling (`_init_minimal_kv_cache_for_profiling`), which is evidence that a minimal-KV startup pattern is legitimate
- TPU currently auto-sizes KV cache all the way to the HBM budget, and that strategy is what looks incompatible with async-native startup latency

Practical recommendation:

- add `num_gpu_blocks_override` support through Marin
- sweep small fixed values (for example `64`, `128`, `256`) to find the smallest cache that still supports the desired Harbor / stress-test concurrency
- if that works, use an explicit small KV bootstrap instead of the current auto-full-HBM allocation

### Long-term recommendation: make TPU KV bootstrap a first-class dependency feature

Current best long-term answer:

- keep native async vLLM serving (`AsyncLLM` / OpenAI server path)
- keep Marin's fast weight injection path
- stop spending primary effort on model bootstrap
- make TPU KV-cache bootstrap a first-class feature in `tpu-inference` / `vllm-tpu`

Recommended design:

1. add explicit TPU KV init policy controls, for example:
   - `bootstrap_num_gpu_blocks` or `bootstrap_kv_cache_memory_bytes`
   - `serving_num_gpu_blocks` or `serving_gpu_memory_utilization`
2. start the engine with a tiny KV cache purely to become ready quickly
3. load / sync weights
4. then either:
   - keep serving with a tuned smaller KV cache, or
   - explicitly resize / reinitialize KV cache to the desired serving target before traffic

Why this is the most production-grade direction:

- it removes Marin-specific runtime monkeypatch dependence from the critical path
- it makes startup deterministic and policy-driven instead of auto-allocating nearly all HBM
- it gives Harbor and future async-native consumers an explicit service-capacity knob
- it matches the actual bottleneck uncovered by the experiments:
  - fast model bootstrap is now solved enough
  - TPU KV-cache initialization is the remaining startup blocker
- there is already conceptual precedent upstream on GPU:
  - V1 GPU code uses a minimal-KV bootstrap pattern for profiling via `num_gpu_blocks_override`

What is *not* the right long-term bet:

- more model-loader work
- compile-cache hope as the primary fix
- just lowering `gpu_memory_utilization`
- permanent Marin monkeypatches as the production solution

Practical bridge before dependency changes:

- expose `num_gpu_blocks_override` through Marin
- sweep small fixed values per model / service
- use an explicit small KV bootstrap instead of the current auto-sized full-HBM allocation

Bottom line:

- the best short-term bridge is explicit small KV sizing from Marin
- the best long-term production answer is first-class TPU KV bootstrap control in `tpu-inference` / `vllm-tpu`
