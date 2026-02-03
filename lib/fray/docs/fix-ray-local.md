# Diagnosis: Ray local tests trigger full package installation

## 1. Files that set up `runtime_env` and how

### `lib/fray/src/fray/v2/ray/backend.py` — `build_runtime_env()`

This is the central function (line 168). It decides between two paths:

```python
if environment.pip_packages or extras:
    # HEAVY PATH: calls build_runtime_env_for_packages(), sets working_dir
    runtime_env = build_runtime_env_for_packages(extra=extras, pip_packages=..., env_vars=...)
    runtime_env["working_dir"] = environment.workspace   # <-- zips entire workspace
    runtime_env["excludes"] = [".git", "tests/", "docs/", "**/*.pack"]
    runtime_env["config"] = {"setup_timeout_seconds": 1800}
else:
    # LIGHT PATH: just env_vars + PYTHONPATH
    runtime_env = {"env_vars": env_vars}
```

The heavy path is triggered whenever `environment.pip_packages` or `extras` is non-empty.

### `lib/fray/src/fray/v2/ray/deps.py` — `build_runtime_env_for_packages()`

Runs `uv export --package marin ...` to compute frozen dependencies, writes a `requirements.txt` to `/tmp`, and returns a `RuntimeEnv` dict with `pip: {"packages": "/tmp/ray_reqs_xxx.txt"}`. This is what causes Ray to create a fresh virtualenv and install all packages from scratch in worker processes.

### `lib/fray/src/fray/v2/types.py` — `create_environment()`

Called as the default when `JobRequest.environment` is `None`. It always sets `workspace = os.getcwd()`. It does **not** set `extras` or `pip_packages` by default, so `create_environment()` alone does not trigger the heavy path.

However, `build_runtime_env()` (in `backend.py`) appends extras based on the device type:

- TPU jobs: appends `"tpu"` to extras
- GPU jobs: appends `"gpu"` to extras
- CPU jobs: no extras added, but the `EnvironmentConfig.__post_init__` validation means `workspace` is always set

So **any GPU or TPU job** always hits the heavy path. CPU-only jobs hit the heavy path only if `pip_packages` or `extras` were explicitly set.

### `lib/zephyr/tests/conftest.py`

The Zephyr test fixtures call `ray.init(ignore_reinit_error=True)` with no `runtime_env` — this is fine. The `create_backend_context("ray")` creates a `RayBackendContext` which uses `ray.remote()` with no `runtime_env` at all. **Zephyr tests themselves are NOT the problem.**

### `tests/integration_test.py`

This file does the right thing for local mode:
```python
ray.init(resources={"head_node": 1}, runtime_env={"working_dir": None}, num_cpus=os.cpu_count())
```
The `{"working_dir": None}` explicitly disables workspace zipping.

## 2. Why this triggers package installation

When `build_runtime_env()` produces a dict with:
- `"pip": {"packages": "/tmp/ray_reqs_xxx.txt"}` — Ray creates an isolated virtualenv and runs `pip install -r` with hundreds of frozen packages
- `"working_dir": environment.workspace` — Ray zips up the entire workspace directory and ships it to workers

On a local single-machine Ray cluster, all workers share the same filesystem and Python environment as the driver. The `runtime_env` is completely unnecessary — workers can import packages directly from the parent process's environment.

The cost is:
1. `uv export` subprocess (~seconds)
2. Ray zipping the workspace (~seconds for a large monorepo)
3. Ray creating a new virtualenv and installing all packages (~minutes)

This happens on every `RayClient.submit()` call, which means every test that submits a job through `RayClient` pays this cost.

## 3. Concrete code changes to fix it

### Option A: Skip `runtime_env` entirely for local Ray (recommended)

Add a `local` flag to `RayClient` and `build_runtime_env` that bypasses the heavy path.

**File: `lib/fray/src/fray/v2/ray/backend.py`**

Change `build_runtime_env` signature:

```python
def build_runtime_env(request: JobRequest, cluster_spec: str, *, local: bool = False) -> dict:
```

When `local=True`, skip the pip/working_dir setup entirely:

```python
def build_runtime_env(request: JobRequest, cluster_spec: str, *, local: bool = False) -> dict:
    environment = request.environment if request.environment else create_environment()
    env_vars = dict(environment.env_vars)
    extras = list(environment.extras)

    # ... existing JAX_PLATFORMS logic ...

    # Note: FRAY_CLIENT_SPEC removed - auto-detection via ray.is_initialized()

    if local:
        return {"env_vars": env_vars}

    # ... existing pip_packages/extras logic ...
```

Add a `local` parameter to `RayClient.__init__`:

```python
class RayClient:
    def __init__(self, address: str = "auto", namespace: str | None = None, local: bool = False):
        self._local = local
        # ...
```

Thread it through `_launch_binary_job`, `_launch_callable_job`, `_launch_tpu_job`:

```python
runtime_env = build_runtime_env(request, self._get_client_spec(), local=self._local)
```

**Detection heuristic**: `local=True` when `address` resolves to a local Ray instance. The simplest approach: check if `ray.is_initialized()` was called without a remote address, or let callers pass it explicitly. The integration test already does `ray.init(...)` locally, so `RayClient(local=True)` is natural.

### Option B: Use `MARIN_CI_DISABLE_RUNTIME_ENVS` env var

There is already a reference to this env var in `create_environment()` (line 407 of types.py), but it is only captured into `env_vars` — it is never checked to skip runtime env construction.

Wire it up in `build_runtime_env`:

```python
def build_runtime_env(request: JobRequest, cluster_spec: str) -> dict:
    environment = request.environment if request.environment else create_environment()
    env_vars = dict(environment.env_vars)

    # ... existing logic ...

    # Note: FRAY_CLIENT_SPEC removed - auto-detection via ray.is_initialized()

    disable_runtime_envs = os.environ.get("MARIN_CI_DISABLE_RUNTIME_ENVS", "").lower() in ("1", "true")
    if disable_runtime_envs:
        return {"env_vars": env_vars}

    # ... rest of existing logic ...
```

This is simpler but less explicit than Option A.

### Option C: Combine both

Add the `local` flag (Option A) AND check the env var (Option B) as a fallback. The `local` flag is the primary mechanism; the env var is a belt-and-suspenders escape hatch for CI.

### Test fixture changes

For fray tests that submit jobs via `RayClient`, use:

```python
@pytest.fixture(scope="module")
def ray_client():
    ray.init(ignore_reinit_error=True)
    client = RayClient(local=True)
    yield client
```

For the integration test, change:
```python
set_current_client(RayClient(local=True))
```

## 4. Summary

| Path | Role | Problem? |
|------|------|----------|
| `backend.py:build_runtime_env` | Builds runtime_env dict | Yes - always builds pip env for GPU/TPU jobs |
| `deps.py:build_runtime_env_for_packages` | Runs `uv export`, writes requirements.txt | Yes - called by above |
| `deps.py:build_python_path` | Builds PYTHONPATH for light path | No |
| `context.py:RayBackendContext` | Zephyr's ray.remote wrapper | No - no runtime_env |
| `zephyr/tests/conftest.py` | Zephyr test fixtures | No - plain ray.init() |
| `types.py:create_environment` | Default env config | No directly, but captures unused `MARIN_CI_DISABLE_RUNTIME_ENVS` |

The fix is straightforward: add a `local` mode to `RayClient` that skips `runtime_env` construction entirely, since local Ray workers inherit the driver's filesystem and packages.
