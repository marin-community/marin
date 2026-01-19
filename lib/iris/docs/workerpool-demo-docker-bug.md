# WorkerPool Docker Demo Bug Report

## Summary

The WorkerPool demo fails in Docker mode. The coordinator job crashes with exit code 1 before producing any output, suggesting an early failure in context/client initialization.

## Environment

- Running via `uv run python examples/demo_cluster.py --test-notebook --docker`
- Workers use DockerRuntime to execute jobs in containers

## Symptoms

1. The coordinator job (`workerpool-demo`) is dispatched successfully
2. It successfully submits the pool-worker job with 3 tasks (co-scheduling works)
3. All 3 pool-worker tasks are dispatched to the worker
4. The coordinator then fails with exit code 1
5. No logs are produced from inside the coordinator container

```
JobFailedError: Job workerpool-demo JOB_STATE_FAILED: Exit code: 1
```

## What Works

- All unit tests pass (102 controller tests, 18 worker pool tests)
- Co-scheduling is correctly implemented (one job with replicas=N)
- Local mode (non-Docker) works correctly
- The actor demo in the same notebook passes

## Diagnostic Findings

From controller logs:
```
Job workerpool-demo submitted with 1 task(s)
Dispatched task workerpool-demo/task-0 to worker worker-0-xxx
Job workerpool-demo/pool-worker-xxx submitted with 3 task(s)  # Co-scheduling works!
Dispatched task workerpool-demo/pool-worker-xxx/task-0 to worker worker-0-xxx
Dispatched task workerpool-demo/pool-worker-xxx/task-1 to worker worker-0-xxx
Dispatched task workerpool-demo/pool-worker-xxx/task-2 to worker worker-0-xxx
```

The coordinator crashes before printing "Coordinator starting", meaning the failure occurs in `iris_ctx()` or `create_context_from_env()`.

## Likely Root Cause

The `create_context_from_env()` function creates an `IrisClient.remote()` from environment variables. Inside a nested Docker container (coordinator submitting child jobs), this may fail due to:

1. **Controller address rewriting**: The `IRIS_CONTROLLER_ADDRESS` may not be correctly rewritten for the coordinator's context when it tries to submit child jobs

2. **Bundle inheritance**: The `IRIS_BUNDLE_GCS_PATH` inheritance chain may break for nested jobs

3. **Client initialization**: Something in `IrisClient.remote()` may fail silently when called from inside a Docker container

## Reproduction

```bash
cd lib/iris
uv run python examples/demo_cluster.py --test-notebook --docker
```

The test will fail at the WorkerPool demo cell.

## Proposed Investigation

1. Add error handling/logging to `create_context_from_env()` to capture the actual exception

2. Check if `IRIS_CONTROLLER_ADDRESS` is correctly set to `host.docker.internal:port` for the coordinator job

3. Test the coordinator job in isolation (without WorkerPool) to confirm the issue is in context creation

4. Verify bundle path inheritance works for coordinator -> pool-worker job chain

## Related Changes

This bug was discovered while implementing co-scheduling for WorkerPool:
- `worker_pool.py`: Refactored to use `replicas=N` instead of N separate jobs
- `resolver.py`: Added `_rewrite_address_for_host()` for Docker networking
- `worker.py`: Added `IRIS_ADVERTISE_HOST` environment variable

## Workaround

Use local mode (without `--docker` flag) for testing WorkerPool functionality.
