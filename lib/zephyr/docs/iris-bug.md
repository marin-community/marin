# Iris Environment Injection Bug - Status Report

**Date:** 2026-01-30
**Status:** Fix in progress - moving from os.environ to copy_context()

## Problem

When jobs are submitted via `IrisClient.remote()` to a local cluster, the `IRIS_CONTROLLER_ADDRESS` environment variable is not available in ActorServer handler threads. This breaks cross-actor communication because `iris_ctx()` cannot create an IrisClient without the controller address.

## Root Cause

Thread chain for callable entrypoints (e.g., `_host_actor`):

```
Main thread
  └─ _LocalContainer._execute (threading.Thread)
       ├─ set_job_info() sets ContextVars here
       └─ _host_actor / user callable
            └─ ActorServer.serve_background()
                 └─ uvicorn thread (threading.Thread) ← BREAKS CONTEXT
                      └─ asyncio event loop
                           └─ ActorServer.Call()
                                └─ asyncio.to_thread(method, ...) ← needs ContextVars
```

The critical break is at `ActorServer.serve_background()` which spawns a raw `threading.Thread` for uvicorn. This thread starts with an empty ContextVar context, so handler threads can't see `IRIS_CONTROLLER_ADDRESS`.

## Initial Fix (REJECTED)

**Approach:** Inject IRIS_* vars into `os.environ` in `_LocalContainer._execute()`

**Why rejected:** Local runs should NEVER touch `os.environ`. Even for testing, global state mutation is bad practice:
- Makes tests non-hermetic
- Can cause flaky tests if run in parallel
- Violates isolation principles
- Bad habit that could leak into production

## Correct Solution: copy_context()

### Key Insight

It's not a pure `copy_context()` - the internal context needs to have the **parent job set appropriately**. When we copy the context to spawn the ActorServer thread, we need to:

1. Copy the ContextVars from the container thread (where `set_job_info()` was called)
2. Ensure the parent job relationship is maintained
3. Run the uvicorn server thread within the copied context

### Implementation Location

**File:** `lib/iris/src/iris/actor/server.py`
**Method:** `ActorServer.serve_background()`

**Current code:**
```python
thread = threading.Thread(target=server.run, daemon=True)
thread.start()
```

**Fixed code:**
```python
import contextvars

ctx = contextvars.copy_context()
thread = threading.Thread(target=ctx.run, args=(server.run,), daemon=True)
thread.start()
```

This propagates ContextVars from the container thread → uvicorn thread → asyncio event loop → `asyncio.to_thread()` → handler threads.

### Why This Works

1. `contextvars.copy_context()` captures the current thread's ContextVars (including `_job_info` set by `set_job_info()`)
2. `ctx.run(server.run)` executes the uvicorn server within the copied context
3. `asyncio.run()` (called by uvicorn internally) inherits the context
4. `asyncio.to_thread()` (Python 3.11+) copies the current task's context to the thread pool thread
5. Handler threads can now call `get_job_info()` → returns the correct `IRIS_CONTROLLER_ADDRESS`

### Parent Job Relationship

The copied context contains:
- `IRIS_JOB_ID` - the actor's job ID
- `IRIS_TASK_ID` - the actor's task ID
- `IRIS_CONTROLLER_ADDRESS` - controller URL for creating IrisClient
- `IRIS_WORKER_ID` - worker ID
- Parent job info if this is a nested job

When the actor makes cross-actor calls via `iris_ctx()`, it creates a resolver scoped to its own job namespace, allowing it to resolve endpoints in the correct context.

## Files to Modify

1. **lib/iris/src/iris/actor/server.py**
   - Add `import contextvars`
   - Wrap uvicorn thread with `copy_context()` in `serve_background()`

2. **lib/iris/src/iris/cluster/vm/local_platform.py**
   - Remove `_inject_iris_env()` function
   - Remove call to `_inject_iris_env(env)` in `_LocalContainer._execute()`

3. **lib/zephyr/docs/iris-flow.md**
   - Update to document ContextVar propagation
   - Document parent job relationship in copied context
   - Explain why copy_context() is used instead of os.environ

## Verification

Tests that should pass after fix:
- `lib/zephyr/tests/test_execution.py::test_simple_map[iris]`
- Cross-actor communication tests
- Environment variable availability in handler threads

## Next Steps

1. ✅ Document current status (this file)
2. ⏳ Update control flow diagram with ContextVar propagation details
3. ⏳ Have senior engineer implement copy_context() solution
4. ⏳ Verify all tests pass
5. ⏳ Update technical documentation

## References

- Bug report: `lib/iris/BUGS/controller-address-not-injected.md`
- Control flow: `lib/zephyr/docs/iris-flow.md`
- ContextVar analysis: `lib/iris/docs/technical-notes/contextvar-propagation.md`
