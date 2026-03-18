# Iris Env Var Refactor: Shared `build_common_iris_env()` + `PodConfig`

**Status:** Planned — not yet implemented
**Filed:** 2026-03-18
**Branch context:** `multi/KUA9Ot3s` (KubernetesProvider feature parity)

## Context

The KubernetesProvider feature parity implementation (10 tasks) exposed a structural problem: **the Docker/worker path and K8s path independently interpret `RunTaskRequest` to build env vars**. A code review and codebase audit found 5 bugs that 116 tests didn't catch — all traceable to duplicated, divergent logic:

- `IRIS_TASK_ID` missing `:attempt_id` suffix (fixed)
- `/dev/shm` sized for GPU but not TPU (fixed)
- `SYS_RESOURCE` capability missing for TPU (fixed)
- `IRIS_WORKER_REGION` never set in k8s path (still missing)
- Device env vars reimplemented inline instead of reusing `build_device_env_vars()` (still duplicated)

**Root cause:** Two functions that should be one:
- `kubernetes_provider.py:_build_iris_env()` — 55 lines
- `task_attempt.py:build_iris_env()` — 71 lines
- ~70% identical logic, divergences are silent and untested

**Additionally**, `_build_pod_manifest()` has **8 parameters** — all passed through from `self` in `_apply_pod()`. This will only grow as we add features (image pull secrets, priority class, sidecars, etc.).

**Goal:** Make env-var drift *structurally impossible* and collapse the parameter explosion.

---

## Execution Flow

```
                    ┌──────────────────────────────┐
                    │       RunTaskRequest          │
                    │  (protobuf from controller)   │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │   build_common_iris_env()     │
                    │    (runtime/env.py)           │
                    │                               │
                    │  SINGLE SOURCE OF TRUTH       │
                    │  ~50 lines, keyword-only args │
                    │  Produces: IRIS_TASK_ID,      │
                    │  IRIS_NUM_TASKS, IRIS_BUNDLE_ID│
                    │  IRIS_CONTROLLER_*, IRIS_JOB_*│
                    │  JAX_PLATFORMS, PJRT_DEVICE,  │
                    │  UV_PYTHON_INSTALL_DIR, etc.  │
                    └──────────┬───────────────────┘
                               │
               ┌───────────────┼───────────────────┐
               │                                   │
    ┌──────────▼──────────┐           ┌────────────▼──────────┐
    │   Worker Path       │           │    K8s Path           │
    │  (task_attempt.py)  │           │ (kubernetes_provider) │
    │                     │           │                       │
    │  Adds:              │           │  Adds:                │
    │  • IRIS_WORKER_ID   │           │  • Downward API for   │
    │  • IRIS_ADVERTISE_  │           │    IRIS_ADVERTISE_HOST│
    │    HOST (socket IP) │           │    (status.podIP)     │
    │  • IRIS_WORKER_     │           │                       │
    │    REGION            │           │  Serializes to:       │
    │  • Real port values │           │  k8s pod manifest     │
    │                     │           │  via PodConfig        │
    │  Passes to:         │           │                       │
    │  ContainerConfig    │           │                       │
    └─────────────────────┘           └───────────────────────┘
```

## Why This Reduces Bugs

| Scenario | Before (duplicated) | After (shared) |
|----------|-------------------|----------------|
| New env var `IRIS_FOO` added | Must update 2 functions; forgetting one = silent bug | Update 1 function; both paths get it automatically |
| TPU device vars change | Must update `env.py:build_device_env_vars` AND `kubernetes_provider.py` inline block | Update 1 block in `build_common_iris_env` |
| Constraint serialization logic changes | Must update 2 identical blocks | Update 1 block |
| New developer reads code | Sees "mirrors build_iris_env()" comment, hopes it's true | Reads one function |

## Why This Reduces Code Size

| Component | Lines Before | Lines After | Delta |
|-----------|-------------|-------------|-------|
| `kubernetes_provider.py:_build_iris_env()` | 55 | **0** (deleted) | **-55** |
| `task_attempt.py:build_iris_env()` | 71 | **~12** (thin wrapper) | **-59** |
| `task_attempt.py:_create_container()` UV/CARGO lines | 2 | **0** (in shared fn) | **-2** |
| `env.py:build_device_env_vars()` | 27 | **0** (absorbed) | **-27** |
| `env.py:build_common_iris_env()` | 0 | **~55** (new) | **+55** |
| `kubernetes_provider.py:PodConfig` dataclass | 0 | **~10** | **+10** |
| `_build_pod_manifest` signature cleanup | N/A | **~-5** | **-5** |
| **Production code net** | **155** | **~72** | **-83 lines** |

Plus: **+~80 lines of new conformance tests** that catch cross-path drift.

---

## Implementation Tasks

### Task 1: Add `build_common_iris_env()` to `runtime/env.py`

Pure addition — no existing behavior changes. Keyword-only args accepting primitive fields from `RunTaskRequest` (not the full proto). Includes device env vars (TPU/GPU) to eliminate `build_device_env_vars()` duplication.

### Task 2: Add frozen `PodConfig` dataclass to `kubernetes_provider.py`

Pure addition. Bundles the 7 non-request params of `_build_pod_manifest()` into a frozen dataclass.

### Task 3: Wire k8s path to use shared function + PodConfig

Delete `_build_iris_env()`, refactor `_build_pod_manifest(run_req, config: PodConfig)` to 2 params, add `pod_config` property to `KubernetesProvider`.

### Task 4: Wire worker path to use shared function

Rewrite `build_iris_env()` as a thin wrapper (71 → 12 lines) that calls `build_common_iris_env()` then adds `IRIS_WORKER_ID` and `IRIS_ADVERTISE_HOST`.

### Task 5: Delete `build_device_env_vars()` from `env.py`

Remove from `env.py`, update callers in `docker.py` and `process.py`.

### Task 6: Add conformance tests

Cross-path env var parity test + unit tests for `build_common_iris_env` edge cases.

### Migration Order

```
Task 1 + Task 2 (parallel, pure additions)
    │
    ├── Task 3 (k8s path, depends on 1+2)
    ├── Task 4 (worker path, depends on 1)
    │
    └── Task 5 (delete old code, depends on 3+4)
         └── Task 6 (tests, alongside 3-5)
```

## Critical Files

| File | Change |
|------|--------|
| `lib/iris/src/iris/cluster/runtime/env.py` | Add `build_common_iris_env()`, delete `build_device_env_vars()` |
| `lib/iris/src/iris/cluster/controller/kubernetes_provider.py` | Add `PodConfig`, delete `_build_iris_env()`, refactor `_build_pod_manifest` to 2 params |
| `lib/iris/src/iris/cluster/worker/task_attempt.py` | Rewrite `build_iris_env()` as thin wrapper |
| `lib/iris/src/iris/cluster/runtime/docker.py` | Remove `build_device_env_vars()` call |
| `lib/iris/src/iris/cluster/runtime/process.py` | Remove `build_device_env_vars()` call |
| `lib/iris/tests/cluster/controller/test_kubernetes_provider.py` | Update ~25 `_build_pod_manifest` calls |
| `lib/iris/tests/cluster/runtime/test_env_parity.py` | **New**: conformance + unit tests |

## Verification

1. `uv run pytest lib/iris/tests/cluster/controller/test_kubernetes_provider.py -x -v`
2. `uv run pytest lib/iris/tests/cluster/controller/ -x`
3. `uv run pytest lib/iris/tests/cluster/runtime/ -x`
4. `uv run pytest lib/iris/tests/cluster/worker/ -x`
5. `./infra/pre-commit.py --all-files --fix`
