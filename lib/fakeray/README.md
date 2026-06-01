# fakeray

A Ray-Core-compatible **shim** that executes tasks on Iris via
[Fray](../fray), instead of a real Ray cluster. Built to run
[smallpond](https://github.com/deepseek-ai/smallpond) — which drives execution
through a small slice of the Ray Core API — on Marin infrastructure.

Status: **sketch / stage 1–2 proven.** Single-node (Fray `LocalClient`) works
end to end, including an unmodified smallpond quickstart. Multi-node fan-out on
Iris is gated on the shared-filesystem story — see the design doc at
`.agents/projects/20260531_fray_smallpond_fakeray_design.md`.

## What it implements

The exact Ray surface smallpond uses, nothing more:

`init` · `shutdown` · `put` · `get` · `wait` · `remote` (`@ray.remote`,
`.options(...)`, `.remote(...)`) · `ObjectRef` · `timeline` (no-op) ·
`exceptions.{RuntimeEnvSetupError, GetTimeoutError, RayTaskError, RayError}`.

An `ObjectRef` is a driver-local `concurrent.futures.Future`. A ready-queue
scheduler dispatches each remote call to a pool of Fray actors (one worker
thread per actor slot, blocking on `ActorFuture.result()`). Values passed
between tasks are smallpond `DataSet` *descriptors* (parquet paths), not bulk
data, so nothing large crosses the shim — the data lives on `data_root`.

## Layout

```
lib/fakeray/
  src/fakeray/            # the implementation (distribution: marin-fakeray)
    __init__.py           #   public Ray-compatible API
    _object_ref.py        #   ObjectRef (Future-backed)
    _scheduler.py         #   ready-queue DAG scheduler + FakeRayConfig
    _executor.py          #   FakeRayExecutor (the Fray actor)
    exceptions.py         #   Ray-compatible exception types
  ray-shim/               # a SEPARATE distribution literally named `ray`
    src/ray/__init__.py   #   re-exports everything from fakeray
    src/ray/exceptions.py #   real submodule so `import ray.exceptions` resolves
  tests/                  # stage-1 tests over Fray LocalClient
```

## Exposing it as `ray`

Two ways to make a library's `import ray` resolve to this shim.

### 1. Runtime install (no build, no resolver changes)

```python
import fakeray
fakeray.install()      # swaps sys.modules["ray"] (and ray.exceptions)
import smallpond        # smallpond's `import ray` now binds to fakeray
```

Must run **before** the requiring library is imported. Wins even when real
`ray` is installed in the same environment (verified against ray 2.55.1).
Registers `ray.exceptions` explicitly, because dotted submodule imports resolve
through `sys.modules`, not a package `__getattr__`.

### 2. uv override (no source edits to the requirer)

Force the sibling `ray`-named distribution in, beating the transitive
`ray>=x` constraint:

```bash
( cd lib/fakeray && uv build --wheel -o dist . && uv build --wheel -o dist ray-shim )
```
```toml
# in the consuming project's pyproject.toml
[tool.uv]
override-dependencies = ["ray @ file:///abs/path/to/dist/ray-2.55.0-py3-none-any.whl"]
```

`override-dependencies` is the one uv lever that overrides *transitive*
requirements; the stub's version is cosmetic. (Mechanism per the design note;
the version is kept ≥ smallpond's `ray>=2.10` floor for any non-overridden path.)

## Tests

Not yet a uv-workspace member (deliberate, while it's a sketch). Run against the
marin venv with pytest available:

```bash
PYTHONPATH=lib/fakeray/src uv run --no-project --python .venv/bin/python \
  --with pytest --with marin-fray --with-editable lib/fakeray \
  python -m pytest lib/fakeray/tests -c lib/fakeray/pytest.ini
```

Covers: remote/get round-trip, ObjectRef auto-deref, a diamond DAG, `put` refs
as deps, ordered list-`get`, non-blocking `wait(timeout=0)`, task-failure
propagation to `get`, and descendant poisoning (a dependent of a failed task
raises rather than hanging).

## Known limitations (v1)

- Slot-based scheduling (N actors × 1 task); `num_cpus`/`memory` per-task are
  recorded but not bin-packed.
- `timeline` is a no-op (use the Iris dashboard).
- Multi-node requires a shared `data_root` (gcsfuse/Filestore, or fsspec-native
  smallpond I/O). The shim is inert without it. See the design doc §10.
