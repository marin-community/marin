# K8s/GCP Service Boundary Architecture

## Problem

K8s LOCAL mode (`k8s_service_impl.py:263-749`) is identical to DRY_RUN: both store manifests in-memory and run a fake scheduler. The `_mode` field is stored but never branched on. LOCAL mode should spawn real processes for pod manifests, enabling actual task execution without a K8s cluster.

GCP service (`gcp_service_impl.py:186-1086`) handles all three modes in one 900-line class with `if self._mode == ServiceMode.CLOUD` scattered throughout every method. It has no factory function — consumers construct `GcpServiceImpl` directly (12+ call sites in tests alone).

## Proposed Solution

### K8s: Split into DryRunK8sService + LocalK8sService

Split `K8sServiceImpl` into two classes sharing validation/scheduling via composition:

```python
# k8s_service_impl.py — keep validation, scheduling helpers, and DryRunK8sService

class K8sStore:
    """Shared in-memory manifest store with validation and scheduling."""
    # Contains: _resources, _nodes, _node_pools, _injected_failures,
    # _logs, _events, _exec_responses, _file_contents, etc.
    # All current validation/scheduling methods move here.

class DryRunK8sService:
    """Pure in-memory K8s fake. No side effects."""
    def __init__(self, namespace: str = "iris", ...):
        self._store = K8sStore(namespace, available_node_pools)
    # Delegates all protocol methods to self._store
    # Exposes test helpers (inject_failure, set_logs, etc.) via store

class LocalK8sService:
    """Spawns real subprocesses for pods. Scheduling + validation via shared store."""
    def __init__(self, namespace: str = "iris", ...):
        self._store = K8sStore(namespace, available_node_pools)
        self._processes: dict[str, subprocess.Popen] = {}

    def apply_json(self, manifest: dict) -> None:
        self._store.apply_json(manifest)  # validate + schedule
        if manifest["kind"].lower() == "pod":
            self._spawn_pod(manifest)

    def _spawn_pod(self, manifest: dict) -> None:
        """Extract container command from pod spec, run as subprocess."""
        name = manifest["metadata"]["name"]
        spec = manifest["spec"]
        container = spec["containers"][0]
        cmd = container.get("command", []) + container.get("args", [])
        env = {e["name"]: e["value"] for e in container.get("env", []) if "value" in e}
        proc = subprocess.Popen(cmd, env={**os.environ, **env},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self._processes[name] = proc

    def logs(self, pod_name: str, ...) -> str:
        if pod_name in self._processes:
            return self._processes[pod_name].stdout.read().decode()
        return self._store.logs(pod_name, ...)

    def delete(self, resource: str, name: str, ...) -> None:
        self._store.delete(resource, name, ...)
        if name in self._processes:
            self._processes.pop(name).kill()
```

**Why composition over inheritance**: Both classes need the same store mechanics but differ in side effects. A shared `K8sStore` avoids diamond inheritance and keeps each class focused.

**Why not Docker**: GCP LOCAL mode already uses `ProcessRuntime` for local workers (`gcp_service_impl.py:952-982`). K8s LOCAL should match — subprocesses, not containers. Docker is for e2e tests (`_docker_cluster.py`).

### K8s Factory Update

```python
# k8s/__init__.py
def create_k8s_service(mode: ServiceMode, namespace: str = "iris", ...) -> K8sService:
    if mode == ServiceMode.CLOUD:
        from iris.cluster.k8s.kubectl import Kubectl
        return Kubectl(namespace=namespace, ...)
    if mode == ServiceMode.LOCAL:
        return LocalK8sService(namespace=namespace, available_node_pools=available_node_pools)
    return DryRunK8sService(namespace=namespace, available_node_pools=available_node_pools)
```

### GCP: Split into DryRunGcpService + LocalGcpService + CloudGcpService, add factory

The current `GcpServiceImpl` has three distinct personalities:
- **CLOUD** (`_tpu_create_cloud`, `_vm_create_cloud`, etc.): shells out to `gcloud`
- **DRY_RUN**: validates + stores in-memory
- **LOCAL**: validates + stores in-memory + spawns workers via `create_local_slice`

```python
# gcp_service_impl.py — shared validation stays as module-level functions (already is)

class DryRunGcpService:
    """In-memory GCP fake with validation and failure injection."""
    # Current DRY_RUN paths from GcpServiceImpl. No worker spawning.

class LocalGcpService(DryRunGcpService):
    """Extends DryRunGcpService with real local worker spawning."""
    # Adds create_local_slice, _create_slice_with_workers, shutdown
    # Overrides tpu_create/tpu_delete to also manage local slices

class CloudGcpService:
    """Shells out to gcloud CLI."""
    # Current _*_cloud methods become the primary methods
```

**Why LocalGcpService extends DryRunGcpService**: LOCAL mode does everything DRY_RUN does (validation, in-memory state, failure injection) plus worker spawning. This is true extension, not just shared code.

```python
# New: gcp_service_factory.py or in gcp_service_impl.py
def create_gcp_service(mode: ServiceMode, project_id: str = "", **kwargs) -> GcpService:
    if mode == ServiceMode.CLOUD:
        return CloudGcpService(project_id=project_id)
    if mode == ServiceMode.LOCAL:
        return LocalGcpService(project_id=project_id, **kwargs)
    return DryRunGcpService(project_id=project_id)
```

## Implementation Plan

### Task 1: Extract K8sStore from K8sServiceImpl (no behavior change)

- **Files**: `lib/iris/src/iris/cluster/k8s/k8s_service_impl.py`
- **What**: Move all state (`_resources`, `_nodes`, etc.) and internal methods (`_validate_manifest`, `_schedule_pod`, `_release_pod_resources`, etc.) into a `K8sStore` class. `K8sServiceImpl` becomes a thin wrapper delegating to `K8sStore`.
- **Tests**: Existing `test_k8s_service_impl.py` must pass unchanged. No new tests needed.
- **Depends on**: nothing

### Task 2: Split K8sServiceImpl → DryRunK8sService + LocalK8sService

- **Files**: `lib/iris/src/iris/cluster/k8s/k8s_service_impl.py`, `lib/iris/src/iris/cluster/k8s/__init__.py`
- **What**: Rename `K8sServiceImpl` → `DryRunK8sService`. Create `LocalK8sService` that delegates to `K8sStore` but also spawns subprocesses on `apply_json` for pods. Update `create_k8s_service` factory. Update `__init__.py` exports.
- **Tests**: Update `test_k8s_service_impl.py` — rename references. Add new tests for `LocalK8sService`: pod subprocess spawning, logs from subprocess, delete kills subprocess, status reflects process exit code.
- **Depends on**: Task 1

### Task 3: Split GcpServiceImpl → DryRunGcpService + LocalGcpService + CloudGcpService

- **Files**: `lib/iris/src/iris/cluster/platform/gcp_service_impl.py`
- **What**: Extract `CloudGcpService` (all `_*_cloud` methods become primary). Extract `DryRunGcpService` (current DRY_RUN paths). Create `LocalGcpService(DryRunGcpService)` with worker spawning. Add `create_gcp_service()` factory.
- **Tests**: Update `test_gcp_service_impl.py` to use `DryRunGcpService` directly. Add test for factory routing. Existing test_platform.py tests should pass (they construct `GcpServiceImpl` — provide a compatibility alias or update all 15+ call sites).
- **Depends on**: nothing (parallel with Tasks 1-2)

### Task 4: Update all GcpServiceImpl consumers to use factory

- **Files**: `local_cluster.py:100`, `platform/factory.py:76`, `platform/gcp.py:536`, all test files constructing `GcpServiceImpl` directly
- **What**: Replace `GcpServiceImpl(mode=X, ...)` with `create_gcp_service(X, ...)` or the specific class. ~15 call sites in test files, 3 in production code.
- **Tests**: All existing tests must pass.
- **Depends on**: Task 3

## Risks and Open Questions

1. **K8s LOCAL subprocess lifecycle**: Pod containers often specify images, not local commands. `LocalK8sService._spawn_pod` needs to extract a runnable command from the manifest. If the manifest uses an image with no explicit `command`, what happens? Likely: raise an error in LOCAL mode requiring explicit command.

2. **exec/read_file/rm_files in LOCAL mode**: These need subprocess-aware implementations. `exec` could run a command in the subprocess's context (tricky). Initial implementation can fall back to store behavior and iterate.

3. **Thread safety**: `LocalK8sService._processes` dict will be accessed from multiple threads. Needs a lock, same pattern as `GcpServiceImpl._local_slices`.

4. **Backward compatibility**: `K8sServiceImpl` is imported directly in tests and `__init__.py`. Keep it as an alias for `DryRunK8sService` during transition, then remove.
