# Async vLLM Serving Refactor

## Problem

Marin's current "native fast path" uses `InProcessVllmServerBackend`, which builds a custom FastAPI app around blocking `vllm.LLM.generate()` calls in a single worker thread (`lib/marin/src/marin/inference/vllm_inprocess.py`).

That design has two concrete problems:

- it is not the standard vLLM async serving stack,
- it destroys throughput for concurrent HTTP workloads because requests are serialized through blocking `generate()` calls.

At the same time, Marin's fast-loading path in the same module is valuable and should not be thrown away:

- staged bootstrap metadata,
- `load_format="dummy"`,
- Levanter fsspec shard streaming,
- reshape + convert,
- `sync_weights()`.

RL already proves that Marin can use vLLM's async engine plus worker extensions on TPU in `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py`.

## Goals

- Replace the current queue-based native backend with an async-engine backend.
- Use vLLM's standard OpenAI app initialization rather than Marin's custom request queue server.
- Preserve fast TPU loading with shard streaming and low host RAM.
- Keep `VllmEnvironment` fallback to subprocess `vllm serve`.
- Allow Harbor and future scripts to use the same async-native backend.

**Non-goals**

- Reworking the docker backend.
- Solving every possible vLLM version skew.
- Replacing subprocess fallback.
- Optimizing weight transfer beyond a first working async-native implementation.

## Proposed Solution

Create a new native async backend that:

1. stages local bootstrap metadata for object-store checkpoints,
2. creates `AsyncLLM` with `load_format="dummy"`,
3. streams one safetensor shard at a time on the frontend process,
4. serializes each shard for RPC using the existing RL helper,
5. applies the shard inside the async engine using RL's `WorkerExtension`,
6. builds the standard vLLM OpenAI FastAPI app and initializes it with the created engine client,
7. serves that app from a dedicated thread / event loop.

Core idea:

```python
engine_args = AsyncEngineArgs(
    model=bootstrap_model_source,
    load_format="dummy",
    worker_extension_cls="marin.rl.environments.inference_ctx.inflight.worker.WorkerExtension",
    tensor_parallel_size=...,
    enforce_eager=...,
)

async with build_async_engine_client_from_engine_args(
    engine_args,
    disable_frontend_multiprocessing=True,
) as engine_client:
    for shard_dict in stream_remote_shards(model.path):
        reshape_attention_tensors(shard_dict, ...)
        serialized = serialize_state_dict_for_rpc(shard_dict)
        await engine_client.engine_core.collective_rpc_async(
            "update_weight",
            args=(serialized, mapping_model_name),
        )

    supported_tasks = await engine_client.get_supported_tasks()
    app = build_app(args, supported_tasks)
    await init_app_state(engine_client, app.state, args, supported_tasks)
    await serve_app(app)
```

## Implementation Outline

1. Add a new async-native runtime in Marin inference that owns engine creation, shard streaming, app init, and lifecycle.
2. Replace backend selection in `VllmEnvironment` so native mode prefers the new async-native backend instead of `InProcessVllmServerBackend`.
3. Reuse RL `WorkerExtension` and `serialize_state_dict_for_rpc` instead of maintaining separate async weight-update code.
4. Remove the queue-based request-serving path from the active backend and update tests around eligibility, fallback, and served-model-name handling.
5. Validate with targeted unit tests locally and cluster smoke tests remotely.

## Notes

- This plan intentionally separates "make serving architecture correct" from "make weight transfer maximally efficient." The first pass can use collective RPC per shard and still be a major architectural improvement.
- If collective RPC turns out to duplicate shard payloads too expensively on TPU, the follow-up is a worker-side initial-load method or a standard weight-transfer engine, not a return to the queue-based `LLM.generate()` server.
- `MARIN_VLLM_MODE=native` should remain the switch for using this path.

## Future Work

- Extract RL async worker-extension utilities into a shared inference module.
- Add remote smoke tests that compare async-native load time and HTTP throughput against subprocess `vllm serve`.
- Investigate whether upstream `serve_http` can be used directly in Marin without signal-handling problems in worker threads.
- Replace per-shard RPC with a more direct transfer mechanism if cluster profiling shows it is necessary.
