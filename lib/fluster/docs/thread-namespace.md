# Thread Context Propagation in Fluster

## Problem

`WorkerPool` spawns `WorkerDispatcher` threads that need to resolve actor endpoints. The resolver uses `FlusterContext` (stored in a `ContextVar`) to determine the namespace. However, `ContextVar` values are **not automatically inherited** by child threads.

```
Main thread (has FlusterContext)
  └── WorkerDispatcher thread (no FlusterContext) ← RuntimeError!
```

## Why ContextVar Doesn't Auto-Inherit

Python's `ContextVar` is designed for async task isolation, not thread inheritance. Each thread starts with an empty context by default. This is intentional - it prevents accidental state leakage between threads.

From [PEP 567](https://peps.python.org/pep-0567/):
> "Context implements the `collections.abc.Mapping` ABC. Each thread has its own context."

## The Clean Fix: `copy_context()`

Python provides `contextvars.copy_context()` to explicitly propagate context to child threads:

```python
from contextvars import copy_context
import threading

# BEFORE (broken): child thread has no context
thread = threading.Thread(target=worker_func)
thread.start()

# AFTER (correct): child thread inherits current context
ctx = copy_context()
thread = threading.Thread(target=ctx.run, args=(worker_func,))
thread.start()
```

## Implementation

In `WorkerPool._launch_workers()`, each thread needs its own context copy because a `Context` can only be entered by one thread at a time:

```python
from contextvars import copy_context

def _launch_workers(self) -> None:
    for worker_state in self._workers.values():
        # Each thread needs its own context copy
        ctx = copy_context()
        dispatcher = WorkerDispatcher(..., context=ctx)
        dispatcher.start()
```

The `WorkerDispatcher.start()` method then runs its thread in that context:

```python
def start(self) -> None:
    if self._context is not None:
        self._thread = threading.Thread(
            target=self._context.run,
            args=(self._run,),
            daemon=True,
        )
    else:
        self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()
```

## Why Not FixedNamespaceResolver?

An alternative is to capture the namespace at creation time and pass it explicitly. This works but:

1. **Adds complexity**: New class, new API parameter
2. **Solves symptom, not cause**: Other code using ContextVar will hit the same issue
3. **Not composable**: Every new ContextVar-dependent feature needs similar workarounds

Using `copy_context()` is the canonical Python solution and handles all ContextVars uniformly.

## References

- [PEP 567 - Context Variables](https://peps.python.org/pep-0567/)
- [contextvars.copy_context()](https://docs.python.org/3/library/contextvars.html#contextvars.copy_context)
