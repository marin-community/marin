# vLLM Implementations in Marin: Canonical Map and Serving Analysis

## Purpose

This document is the canonical map of how vLLM is implemented and consumed in Marin as of
2026-03-18.

It exists because the repository currently mixes two separate questions:

1. How do weights get into vLLM?
2. How do requests reach vLLM?

Most of the confusion came from collapsing those into a single "vLLM approach." They are
separate axes.

- **Load path**: `runai_streamer` vs `load_format="dummy"` + staged metadata + Levanter fsspec + `sync_weights()`
- **Request path**: direct batch `LLM.generate(...)` vs Marin's custom HTTP queue server vs async engine / `vllm serve`

The main conclusion is unchanged:

- Marin's fast-loading path is mostly solved.
- Marin's current in-process HTTP serving path is still architecturally wrong for concurrent HTTP workloads.
- The serving fix is to combine the fast-loading path with the async engine pattern already used in RL.

---

## Executive Summary

There is not one vLLM implementation in Marin. There are several:

1. **Direct offline `vllm.LLM`**
   Used when Marin or a library instantiates `LLM(...)` and calls `llm.generate(...)` directly.
2. **`VllmEnvironment` in-process server**
   Uses `LLM(...)` inside the current process, fast-loads weights with dummy load plus `sync_weights()`,
   and serves a custom OpenAI-compatible FastAPI app.
3. **`VllmEnvironment` subprocess server**
   Spawns `vllm serve` and uses vLLM's built-in OpenAI-compatible HTTP API.
4. **`VllmEnvironment` docker sidecar server**
   Same as subprocess, but containerized.
5. **RL async engine path**
   Uses `AsyncLLM.from_engine_args(...)` plus `WorkerExtension` for weight updates. This is not
   yet Marin's serving backend, but it is the key reference design for what serving should become.

The core tradeoff is:

- **Direct batch `LLM.generate(...)`** is fine when all prompts are available up front.
- **Current in-process HTTP serving** is correctness-complete but throughput-poor because it routes
  HTTP requests through a queue into blocking `LLM.generate(...)` calls.
- **`vllm serve` or `AsyncLLM`** are the right shape for concurrent HTTP because they continuously
  batch requests.

---

## Canonical Backend Map

### 1. Direct Offline `vllm.LLM`

**What it is**

A caller instantiates `vllm.LLM(...)` directly and calls `llm.generate(...)` directly from Python.
There is no HTTP server and no `VllmEnvironment`.

**Where it exists**

- `lib/marin/src/marin/evaluation/evaluators/evalchemy_evaluator.py`
- `lib/marin/src/marin/evaluation/evaluators/simple_evaluator.py`
- `experiments/inference/exp_vllm_inprocess_direct.py`
- `experiments/inference/exp_vllm_batch_test.py`
- `experiments/inference/exp_vllm_70b_smoke_test.py`
- `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py` in RL sync mode

**Load path**

Whatever the caller passes into `LLM(...)`.

- In Evalchemy, object-store checkpoints are auto-configured to use `load_format="runai_streamer"`.
- In `SimpleEvaluator`, the model is loaded directly with `LLM(model=..., enforce_eager=False, ...)`.
- In RL sync mode, initial model creation uses vLLM load settings, and later weight swaps happen via
  direct `sync_weights()`.

**Request path**

Single Python call:

```python
outputs = llm.generate(prompts, sampling_params)
```

**When it is a good fit**

When the caller already has the whole batch and does not need an HTTP API.

**Important property**

This path is not inherently slow. If the caller passes a real batch to a single `generate()` call,
`vllm.LLM` can batch internally just fine.

---

### 2. `VllmEnvironment` In-Process Server

**What it is**

Marin starts `vllm.LLM(...)` inside the current process, stages tokenizer/config metadata locally,
creates a dummy model skeleton, streams safetensor shards from object storage with Levanter fsspec,
and injects weights via `sync_weights()`. It then serves a custom OpenAI-compatible FastAPI app.

**Main code**

- `lib/marin/src/marin/inference/vllm_server.py`
- `lib/marin/src/marin/inference/vllm_inprocess.py`

**Load path**

Fast path:

1. Stage bootstrap metadata locally.
2. `LLM(model=<local bootstrap dir>, load_format="dummy", ...)`
3. Discover remote safetensor shards.
4. Download shard data via Levanter fsspec.
5. Convert / reshape tensors.
6. Call `sync_weights()` to inject weights into the dummy skeleton.

This is Marin's TPU fast-loading path.

**Request path**

Not async engine. Marin wraps `LLM.generate(...)` behind a queue-based FastAPI app:

1. HTTP handler enqueues request.
2. One worker thread drains the queue.
3. Worker groups requests by sampling params.
4. Worker calls blocking `llm.generate(...)`.
5. Results are handed back to waiting HTTP requests.

**Eligibility**

This path is only attempted when `VllmEnvironment` is in `native` mode and
`evaluate_inprocess_eligibility(...)` passes. Today that means, at minimum:

- `model.path` is set
- `model.path` is `gs://` or `s3://`
- `load_format` is absent or compatible with dummy loading
- model family mappings exist in `MODEL_MAPPINGS` / `MODEL_TRANSPOSE_KEYS`
- bootstrap metadata can be staged locally
- raw `extra_args` contain only supported flags

**Important current detail**

`--served-model-name` is now explicitly supported by the in-process backend. It no longer forces
fallback by itself.

**Strength**

Best current load path for supported TPU object-store checkpoints.

**Weakness**

Wrong request shape for concurrent HTTP. The queue still funnels requests into blocking
`LLM.generate(...)` calls, so aggregate throughput collapses under many independent HTTP requests.

---

### 3. `VllmEnvironment` Native Subprocess Server

**What it is**

Marin spawns `vllm serve` as a subprocess and talks to its OpenAI-compatible HTTP API.

**Main code**

- `lib/marin/src/marin/inference/vllm_server.py`

**Load path**

vLLM-native loading.

For object-store checkpoints, `VllmEnvironment` automatically injects `load_format="runai_streamer"`
when needed.

**Request path**

vLLM's async serving engine with continuous batching.

**Strength**

Correct architecture for concurrent HTTP serving.

**Weakness**

Slow object-store loading on TPU compared with Marin's dummy-load plus fsspec path.

---

### 4. `VllmEnvironment` Docker Sidecar Server

**What it is**

Same basic serving model as subprocess `vllm serve`, but launched in Docker.

**Main code**

- `lib/marin/src/marin/inference/vllm_server.py`

**Load path**

Same family as subprocess server, just containerized.

**Request path**

vLLM's built-in async HTTP serving.

**Why it matters**

This is a transport / deployment variant of the subprocess server, not a distinct model-loading idea.

---

### 5. RL Async Engine Path

**What it is**

RL has already integrated vLLM's async engine directly via:

- `AsyncLLM.from_engine_args(...)`
- `WorkerExtension` for worker-side weight updates
- `SyncVLLMWrapper` as a sync facade over the async engine

**Main code**

- `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py`

**Load path**

Async engine initialization plus worker-extension RPC for weight updates.

**Request path**

Async engine, not the custom queue server.

**Why it matters**

This is the best in-repo proof that Marin can combine custom weight handling with the async engine.
It is the clearest template for the serving fix.

---

## Consumer Map

The important distinction is that evaluators are consumers of one of the backends above.

| Consumer | What it actually uses today | Notes |
|---|---|---|
| `EvalchemyEvaluator` | Direct offline `vllm.LLM` through lm-eval's `VLLM` wrapper | No HTTP server. Does not go through `VllmEnvironment`. |
| `LmEvaluationHarnessEvaluator` | `VllmEnvironment` | lm-eval talks to Marin over `local-completions` or `local-chat-completions`. Backend may be in-process, subprocess, or docker. |
| `HarborEvaluator` | `VllmEnvironment` when `model.path` is set; otherwise no local vLLM server | Harbor always consumes HTTP. In native mode it may use in-process if eligible, otherwise subprocess. |
| `SimpleEvaluator` | Direct offline `vllm.LLM` | Pure batch API usage. |
| RL `vLLMInferenceContext` in `SYNC` mode | Direct offline `vllm.LLM` | Uses direct generation plus direct `sync_weights()` on the driver worker. |
| RL `vLLMInferenceContext` in `ASYNC` mode | RL async engine path | Already uses `AsyncLLM + WorkerExtension`. |
| `experiments/inference/exp_vllm_stress_test.py` | `VllmEnvironment` | Exercises serving, not direct batch generation. |
| `experiments/inference/exp_vllm_eval.py` | `VllmEnvironment` | Uses lm-eval's HTTP `local-completions` path, not Evalchemy's direct `LLM` path. |
| `exp_vllm_inprocess_direct.py`, `exp_vllm_batch_test.py`, `exp_vllm_70b_smoke_test.py` | Direct offline `vllm.LLM` | Useful for isolating load/inject and batch-generation behavior without HTTP. |

---

## How `VllmEnvironment` Chooses a Backend

When code uses `VllmEnvironment`, the backend is chosen like this:

0. If no mode is passed, Marin resolves mode from `MARIN_VLLM_MODE`, defaulting to `docker`.
1. If mode resolves to `docker`, Marin uses the docker sidecar backend.
2. If mode resolves to `native`, Marin checks in-process eligibility.
3. If eligibility passes, Marin tries the in-process backend first.
4. If in-process startup fails, Marin automatically falls back to subprocess `vllm serve`.
5. If eligibility fails up front, Marin goes straight to subprocess `vllm serve`.

Important details:

- `engine_kwargs` such as `max_model_len`, `tensor_parallel_size`, `gpu_memory_utilization`,
  `model_loader_extra_config`, and `enforce_eager` are read directly by the in-process `LLM(...)`
  constructor.
- Raw `extra_args` are separate from `engine_kwargs`.
- `--served-model-name` is currently the only extra CLI flag explicitly handled by the in-process path.
- For object-store models on subprocess paths, `VllmEnvironment` may auto-add `load_format="runai_streamer"`.

---

## The Real Architectural Split

The cleanest mental model is this matrix:

| Request path | Load path | Current example in Marin | Fit |
|---|---|---|---|
| Direct batch `LLM.generate(...)` | Slow vLLM-native loading | Evalchemy today | Good for batch eval, bad cold start on GCS models |
| Direct batch `LLM.generate(...)` | Fast dummy-load + fsspec + `sync_weights()` | Direct smoke / batch experiments; parts of RL sync workflow | Good for batch testing and controlled offline usage |
| Custom HTTP queue -> blocking `LLM.generate(...)` | Fast dummy-load + fsspec + `sync_weights()` | Current in-process serving backend | Correctness works, concurrency throughput is poor |
| Async engine / `vllm serve` | Slow vLLM-native loading | Current subprocess and docker serving backends | Good serving shape, slow TPU object-store load |
| Async engine + worker extension | Potentially fast dummy-load + custom weight injection | RL async path today; proposed serving architecture | Best long-term direction for HTTP serving |

This is the key point:

- Marin already has the **fast load path**.
- Marin already has an in-repo example of the **right async engine pattern**.
- What Marin does not yet have is a serving backend that combines those two.

---

## Why the Current In-Process Server Underperforms

The current in-process server is not slow because `vllm.LLM` itself is slow.
It is slow because of how HTTP requests are fed into it.

The queue worker does roughly this:

```python
while True:
    reqs = dequeue_some_requests()
    outputs = llm.generate(prompts_from(reqs), sampling_params)
    fulfill_futures(outputs)
```

That looks reasonable, but `llm.generate(...)` is blocking.

While one call is running:

- new HTTP requests can arrive,
- but they cannot enter the running generation call,
- so they wait for the next loop iteration,
- which usually means the worker sees another tiny batch.

So even if each individual `generate()` call runs at healthy internal token throughput, the
aggregate HTTP throughput is poor because the system keeps paying per-call overhead and never gets
true continuous batching.

This is why the serving problem is architectural, not just a flag-tuning problem.

---

## What Each Consumer Actually Wants

### Evalchemy

Evalchemy wants batch generation, not HTTP serving.

That means:

- direct `LLM.generate(...)` is a reasonable request path,
- `VllmEnvironment` is not relevant to its current implementation,
- the biggest improvement opportunity is faster weight loading, not serving redesign.

### Harbor

Harbor wants OpenAI-compatible HTTP serving under concurrent load.

That means:

- the request path matters more than the library surface,
- current queue-based in-process serving is the wrong long-term architecture,
- subprocess `vllm serve` has the right serving shape today,
- a future async in-process backend could also satisfy Harbor if it preserves the HTTP API.

### lm-eval Harness via `VllmEnvironment`

This is also an HTTP-serving consumer, even though the end user thinks of it as evaluation.

It behaves much more like Harbor than Evalchemy from an architectural standpoint because lm-eval
is talking over `local-completions` / `local-chat-completions`.

### RL

RL uses both worlds:

- sync mode uses direct `LLM.generate(...)` and direct weight sync,
- async mode already uses the async engine pattern we want for serving.

RL is not evidence that the queue-based HTTP server is good. It is evidence that Marin already has
code for async engine plus custom weight management.

---

## Status of the Fast-Loading Work

The fast-loading work should be treated as mostly successful and largely independent from the HTTP
throughput problem.

What is already established:

- dummy-load bootstrap works for supported object-store checkpoints,
- metadata staging works around `LLM(load_format="dummy")` rejecting raw `gs://` paths,
- Levanter fsspec loading is materially faster than `runai_streamer` on TPU object-store models,
- shard-streaming plus `sync_weights()` keeps host memory manageable,
- this path is already integrated into the in-process serving backend.

What is not solved by that work:

- continuous batching for HTTP workloads,
- async engine integration in serving,
- parity between fast load and `vllm serve`-style request scheduling.

---

## Recommendation

### Near-term interpretation

Use the following mental model when reasoning about Marin vLLM behavior:

- **Evalchemy**: direct batch API consumer
- **Harbor**: HTTP serving consumer
- **lm-eval harness via `VllmEnvironment`**: HTTP serving consumer
- **current in-process server**: fast-loading backend with the wrong serving architecture
- **subprocess / docker `vllm serve`**: correct serving architecture with a slower load path
- **RL async path**: design reference for the future serving backend

### Actual serving direction

The right next step is:

1. Keep the fast-loading pieces: dummy load, metadata staging, fsspec shard streaming, `sync_weights()`.
2. Replace the queue-based `LLM.generate(...)` serving core with an async-engine-based core.
3. Reuse the RL `AsyncLLM + WorkerExtension` pattern rather than inventing a third weight-update mechanism.
4. Preserve subprocess `vllm serve` as a fallback backend.

### What not to spend time on

Do not treat these as the main fix:

- merely toggling `enforce_eager`
- minor batching tweaks to the queue worker
- reasoning about Harbor as if `--served-model-name` were still the blocker
- treating direct batch `LLM.generate(...)` success as evidence that the HTTP server architecture is sound

---

## Bottom Line

Marin's vLLM story is easiest to understand if you separate **loading** from **serving**.

- Fast loading is largely in place.
- Direct batch generation is already fine for batch-style consumers like Evalchemy.
- Concurrent HTTP serving is the remaining architectural problem.
- The codebase already contains the async-engine pattern needed to solve it.

So the serving project is not "make everything use the same vLLM entrypoint." It is:

**combine Marin's fast-loading path with the RL async-engine pattern so HTTP-serving consumers get
continuous batching without giving up the TPU loading wins.**
