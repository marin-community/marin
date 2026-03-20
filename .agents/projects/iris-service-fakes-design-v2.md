# K8sServiceImpl Rebuild & GcpServiceImpl Enhancement — Design v2

## Problem

The K8sServiceImpl fake (`lib/iris/src/iris/cluster/k8s/k8s_service_impl.py`) is a flat key-value store with no scheduling semantics. `apply_json` stores manifests verbatim; there are no nodes, no resource fitting, no toleration matching, and no pod status transitions. This means:

1. **`_query_capacity` is broken** — `provider.py:1096` calls `list_json("nodes", cluster_scoped=True)` which returns `[]` because no node objects exist. Capacity is always `None`.

2. **Pods never go Pending** — A pod requesting 8× A100 GPUs on a cluster with zero GPU nodes silently gets stored as-is. Real K8s would leave it `Pending` with a `FailedScheduling` event. Tests can't verify scheduling failure paths.

3. **`_poll_pods` gets no status** — `provider.py:997` reads `status.phase` and `status.containerStatuses`. The fake stores raw manifests with no `status` block, so tests must manually inject status dicts.

4. **Node pool model is just names** — `add_node_pool("gpu-pool")` at `k8s_service_impl.py:126` records a string. No GPU count, labels, taints, or resources.

5. **Kubectl types leak into the protocol layer** — `KubectlError`, `KubectlLogLine`, `KubectlLogResult` are defined in `kubectl.py` but used by `k8s_service.py:12` and `k8s_service_impl.py:15`. The `coreweave.py:48` and `config.py:1131` import `Kubectl` directly.

6. **GcpServiceImpl gaps** — No duplicate detection (`tpu_create` overwrites at `gcp_service_impl.py:357`), no `CREATING→READY` transition (hardcoded `"READY"` at line 349), no per-type-per-zone availability.

## Proposed Solution

### 1. Node & NodePool Data Model

```python
# k8s_service_impl.py — new dataclasses

@dataclass
class FakeNodeResources:
    """Resources a node has available (allocatable in K8s terms)."""
    cpu_millicores: int = 4000
    memory_bytes: int = 16 * 1024**3
    gpu_count: int = 0
    ephemeral_storage_bytes: int = 100 * 1024**3

    def to_allocatable(self) -> dict[str, str]:
        """Convert to K8s-style allocatable dict for node status."""
        d: dict[str, str] = {
            "cpu": str(self.cpu_millicores) + "m",
            "memory": str(self.memory_bytes),
            "ephemeral-storage": str(self.ephemeral_storage_bytes),
        }
        if self.gpu_count > 0:
            d["nvidia.com/gpu"] = str(self.gpu_count)
        return d


@dataclass
class FakeNode:
    """A schedulable node in the fake cluster."""
    name: str
    labels: dict[str, str]
    taints: list[dict[str, str]]  # [{"key": ..., "operator": ..., "effect": ...}]
    allocatable: FakeNodeResources
    committed: FakeNodeResources = field(default_factory=lambda: FakeNodeResources(
        cpu_millicores=0, memory_bytes=0, gpu_count=0, ephemeral_storage_bytes=0
    ))

    def available(self) -> FakeNodeResources:
        return FakeNodeResources(
            cpu_millicores=self.allocatable.cpu_millicores - self.committed.cpu_millicores,
            memory_bytes=self.allocatable.memory_bytes - self.committed.memory_bytes,
            gpu_count=self.allocatable.gpu_count - self.committed.gpu_count,
            ephemeral_storage_bytes=self.allocatable.ephemeral_storage_bytes - self.committed.ephemeral_storage_bytes,
        )

    def to_k8s_dict(self) -> dict:
        """Render as a K8s Node JSON object for list_json("nodes")."""
        return {
            "apiVersion": "v1",
            "kind": "Node",
            "metadata": {"name": self.name, "labels": dict(self.labels)},
            "spec": {"taints": list(self.taints)} if self.taints else {},
            "status": {"allocatable": self.allocatable.to_allocatable()},
        }


@dataclass
class NodePoolConfig:
    """Configuration for a pool of identical nodes."""
    name: str
    instance_type: str
    node_count: int
    labels: dict[str, str] = field(default_factory=dict)
    taints: list[dict[str, str]] = field(default_factory=list)
    per_node_resources: FakeNodeResources = field(default_factory=FakeNodeResources)
```

**Why dataclasses over dicts**: The provider reads structured fields (`allocatable.cpu`, `spec.taints`). Typed dataclasses catch mismatches at write time. The `to_k8s_dict()` method produces the exact shape `_query_capacity` expects at `provider.py:1107-1120`.

### 2. Scheduling Algorithm

When `apply_json` receives a Pod (or Job/Deployment template), after validation:

```python
def _schedule_pod(self, manifest: dict) -> None:
    spec = _pod_spec(manifest)
    if spec is None:
        return  # ConfigMap, NodePool, etc — no scheduling

    node_selector = spec.get("nodeSelector", {})
    tolerations = spec.get("tolerations", [])
    requests = _extract_resource_requests(spec)

    for node in self._nodes.values():
        if not _node_selector_matches(node, node_selector):
            continue
        if not _tolerations_satisfy_taints(tolerations, node.taints):
            continue
        if not _resources_fit(node, requests):
            continue
        # Schedule here
        _commit_resources(node, requests)
        manifest.setdefault("spec", {})["nodeName"] = node.name
        manifest["status"] = {"phase": "Running", "containerStatuses": [
            {"name": "task", "state": {"running": {}}}
        ]}
        return

    # No node fits → Pending
    manifest["status"] = {"phase": "Pending", "conditions": [
        {"type": "PodScheduled", "status": "False", "reason": "Unschedulable",
         "message": _unschedulable_reason(node_selector, tolerations, requests)}
    ]}
    self._auto_event(manifest, "FailedScheduling",
                     _unschedulable_reason(node_selector, tolerations, requests))
```

Helper functions (all module-level, no methods):

```python
def _extract_resource_requests(spec: dict) -> dict[str, int]:
    """Sum resource requests across all containers. Returns {resource_key: quantity}."""
    totals: dict[str, int] = {}
    for container in spec.get("containers", []):
        reqs = container.get("resources", {}).get("requests", {})
        limits = container.get("resources", {}).get("limits", {})
        for key in {"cpu", "memory", "nvidia.com/gpu", "ephemeral-storage"}:
            val = reqs.get(key) or limits.get(key)
            if val:
                totals[key] = totals.get(key, 0) + _parse_k8s_quantity(val)
    return totals

def _node_selector_matches(node: FakeNode, selector: dict[str, str]) -> bool:
    return all(node.labels.get(k) == v for k, v in selector.items())

def _tolerations_satisfy_taints(tolerations: list[dict], taints: list[dict]) -> bool:
    """Every NoSchedule taint must be tolerated. Mirrors k8s TaintToleration plugin."""
    for taint in taints:
        if taint.get("effect") != "NoSchedule":
            continue
        if not any(_toleration_matches(t, taint) for t in tolerations):
            return False
    return True

def _toleration_matches(toleration: dict, taint: dict) -> bool:
    if toleration.get("operator") == "Exists":
        return toleration.get("key", "") == "" or toleration.get("key") == taint.get("key")
    return (toleration.get("key") == taint.get("key")
            and toleration.get("value", "") == taint.get("value", "")
            and toleration.get("effect", "") == taint.get("effect", ""))

def _resources_fit(node: FakeNode, requests: dict[str, int]) -> bool:
    avail = node.available()
    if requests.get("cpu", 0) > avail.cpu_millicores:
        return False
    if requests.get("memory", 0) > avail.memory_bytes:
        return False
    if requests.get("nvidia.com/gpu", 0) > avail.gpu_count:
        return False
    return True
```

**Why this approach over a simpler stub**: The provider already builds real pod manifests with `nodeSelector`, `tolerations`, and `resources.limits` (`provider.py:301-459`). If the fake doesn't evaluate these, tests can't catch misconfigurations like missing tolerations or wrong resource requests. The scheduling logic is ~60 lines of pure functions — trivial compared to real kube-scheduler but enough to catch the bugs that matter.

### 3. K8s Types Extraction

Move `KubectlError`, `KubectlLogLine`, `KubectlLogResult` from `kubectl.py` to a new `k8s_types.py`:

```
lib/iris/src/iris/cluster/k8s/k8s_types.py  ← new file, receives the 3 types
lib/iris/src/iris/cluster/k8s/kubectl.py     ← imports from k8s_types.py
lib/iris/src/iris/cluster/k8s/k8s_service.py ← imports from k8s_types.py
lib/iris/src/iris/cluster/k8s/k8s_service_impl.py ← imports from k8s_types.py
lib/iris/src/iris/cluster/k8s/provider.py    ← imports KubectlLogLine from k8s_types.py
lib/iris/src/iris/cluster/k8s/__init__.py    ← re-exports from k8s_types.py
lib/iris/tests/cluster/k8s/test_k8s_service_impl.py ← imports from k8s_types.py
lib/iris/tests/kubernetes/test_kubectl.py     ← imports from k8s_types.py
```

### 4. Kubectl Removal from Non-CLOUD Code

**Current state**: `coreweave.py:48` and `config.py:1131` import `Kubectl` directly.

**Target**: The only code that should import `Kubectl` is:
- `config.py:make_provider` (constructs it, types as `K8sService`)
- `coreweave.py.__init__` (constructs it, types as `K8sService`)
- `kubectl.py` itself

**The `popen()` problem**: `_coreweave_tunnel` at `coreweave.py:881` takes `kubectl: Kubectl` and calls `kubectl.popen()`. This is the only method on `Kubectl` not on the `K8sService` protocol.

**Solution**: Add a `SubprocessK8s` protocol that extends `K8sService` with `popen()`:

```python
# k8s_service.py

class SubprocessK8s(K8sService, Protocol):
    """K8sService plus subprocess escape hatch for port-forwarding."""
    def popen(self, args: list[str], *, namespaced: bool = False, **kwargs) -> subprocess.Popen: ...
```

`CoreWeavePlatform.__init__` types `self._kubectl` as `SubprocessK8s`. `_coreweave_tunnel` takes `SubprocessK8s`. The `K8sServiceImpl` fake does NOT implement `popen()` — it's only needed by `CoreWeavePlatform` in CLOUD mode. Tests that use `K8sServiceImpl` never call `popen()`.

### 5. GcpServiceImpl Enhancements

```python
# gcp_service_impl.py — additions to GcpServiceImpl.__init__

self._available_types_by_zone: dict[str, set[str]] | None = None  # None = all types in all zones

# In tpu_create, after validation:
key = (request.name, request.zone)
if key in self._tpus:
    raise PlatformError(f"TPU {request.name!r} already exists in {request.zone}")

# Per-type-per-zone check:
if self._available_types_by_zone is not None:
    zone_types = self._available_types_by_zone.get(request.zone, set())
    if request.accelerator_type not in zone_types:
        raise ResourceNotFoundError(
            f"Accelerator type {request.accelerator_type!r} not available in {request.zone}"
        )

# State transitions: create as CREATING, test helper advances
info = TpuInfo(..., state="CREATING", ...)
```

Test helpers:

```python
def set_available_types_by_zone(self, mapping: dict[str, set[str]]) -> None:
    """Restrict which TPU types are available in which zones."""
    self._available_types_by_zone = mapping

def advance_tpu_state(self, name: str, zone: str, state: str = "READY") -> None:
    """Advance a TPU's state (e.g. CREATING → READY). For tests."""
    key = (name, zone)
    if key not in self._tpus:
        raise ValueError(f"TPU {name!r} not found in {zone}")
    self._tpus[key] = dataclasses.replace(self._tpus[key], state=state)
```

### 6. Updated `add_node_pool` API

The existing `add_node_pool(pool_name: str)` signature must change. New signature:

```python
def add_node_pool(
    self,
    pool_name: str,
    *,
    node_count: int = 1,
    labels: dict[str, str] | None = None,
    taints: list[dict[str, str]] | None = None,
    resources: FakeNodeResources | None = None,
) -> None:
```

This creates `node_count` `FakeNode` objects, each with the pool's labels (plus `cloud.google.com/gke-nodepool: pool_name`), taints, and resources. The pool name set check in `_validate_manifest` remains.

### 7. `list_json("nodes")` Support

`list_json` already filters by `(kind, _)`. Nodes are stored in a separate `_nodes: dict[str, FakeNode]` dict (keyed by node name). When `list_json("nodes", cluster_scoped=True)` is called, return `[node.to_k8s_dict() for node in self._nodes.values()]` with label filtering.

This means `list_json` needs a small branch:

```python
def list_json(self, resource, *, labels=None, cluster_scoped=False):
    normalized = _normalize_resource(resource)
    if normalized == "node":
        return self._list_nodes(labels)
    # ... existing logic
```

### 8. Pod Status Transition Helpers

```python
def transition_pod(self, name: str, phase: str, *,
                   exit_code: int | None = None,
                   reason: str | None = None) -> None:
    """Manually set pod phase and container status. For tests."""
    key = ("pod", name)
    manifest = self._resources.get(key)
    if manifest is None:
        raise ValueError(f"Pod {name!r} not found")
    status: dict = {"phase": phase}
    if phase == "Running":
        status["containerStatuses"] = [{"name": "task", "state": {"running": {}}}]
    elif phase in ("Succeeded", "Failed"):
        terminated: dict = {"exitCode": exit_code or (0 if phase == "Succeeded" else 1)}
        if reason:
            terminated["reason"] = reason
        status["containerStatuses"] = [{"name": "task", "state": {"terminated": terminated}}]
    manifest["status"] = status
```

---

## Implementation Plan

### Task 1: Extract K8s types to `k8s_types.py` (no deps)

**Files to modify:**
- Create `lib/iris/src/iris/cluster/k8s/k8s_types.py` — move `KubectlError`, `KubectlLogLine`, `KubectlLogResult` from `kubectl.py:39-57`
- `lib/iris/src/iris/cluster/k8s/kubectl.py:39-57` — remove classes, import from `k8s_types.py`
- `lib/iris/src/iris/cluster/k8s/k8s_service.py:12` — change import
- `lib/iris/src/iris/cluster/k8s/k8s_service_impl.py:15` — change import
- `lib/iris/src/iris/cluster/k8s/provider.py:26` — change import
- `lib/iris/src/iris/cluster/k8s/__init__.py:9` — change import source
- `lib/iris/tests/cluster/k8s/test_k8s_service_impl.py:12` — change import
- `lib/iris/tests/kubernetes/test_kubectl.py` — change imports

**Behavior:** Pure move refactor. All existing tests must pass unchanged.

**Tests:** Run existing `test_k8s_service_impl.py` and `test_kubectl.py`. No new tests needed.

### Task 2: Add `SubprocessK8s` protocol and retype Kubectl usages (depends on Task 1)

**Files to modify:**
- `lib/iris/src/iris/cluster/k8s/k8s_service.py` — add `SubprocessK8s` protocol with `popen()` method
- `lib/iris/src/iris/cluster/platform/coreweave.py:48` — change import from `Kubectl` to `SubprocessK8s`; change `self._kubectl` type annotation; change `_coreweave_tunnel` param type from `Kubectl` to `SubprocessK8s`
- `lib/iris/src/iris/cluster/config.py:1131` — import `Kubectl` stays (it's the constructor), but the `KubernetesProvider(kubectl=...)` call already types to `K8sService`
- `lib/iris/src/iris/cluster/k8s/__init__.py` — keep `Kubectl` in exports (CLOUD code needs it), add `SubprocessK8s`

**Behavior:** `CoreWeavePlatform._kubectl` typed as `SubprocessK8s` instead of `Kubectl`. Everything else unchanged.

**Tests:** Type-check with `uv run pyrefly`. Run coreweave-related tests if any exist.

### Task 3: Rebuild K8sServiceImpl with node model and scheduling (no deps)

This is the largest task. **Files to modify:**

- `lib/iris/src/iris/cluster/k8s/k8s_service_impl.py` — full rebuild:
  - Add `FakeNodeResources`, `FakeNode`, `NodePoolConfig` dataclasses
  - Add `_nodes: dict[str, FakeNode]` storage
  - Rewrite `add_node_pool()` to accept full attributes and create `FakeNode` objects
  - Add `remove_node_pool()` updates to also remove associated nodes
  - Add scheduling logic in `apply_json` → `_schedule_pod()`
  - Add module-level helpers: `_extract_resource_requests`, `_node_selector_matches`, `_tolerations_satisfy_taints`, `_toleration_matches`, `_resources_fit`, `_parse_k8s_quantity`
  - Make `list_json("nodes")` return fake nodes
  - Add `transition_pod()` test helper
  - Add `set_node_count()` to change node count in existing pool
  - Auto-generate `FailedScheduling` events when pods can't be scheduled
  - Add `_auto_event()` helper that creates a K8s event dict and appends to `self._events`

- `lib/iris/tests/cluster/k8s/test_k8s_service_impl.py` — add tests:
  - `test_pod_scheduled_on_matching_node` — pod with nodeSelector lands on node with matching labels
  - `test_pod_pending_no_matching_node` — pod goes Pending, FailedScheduling event generated
  - `test_pod_pending_insufficient_resources` — pod requesting 8 GPUs on 4-GPU node
  - `test_toleration_matching` — pod without GPU toleration can't land on tainted node
  - `test_resource_commitment` — scheduling two pods reduces available resources
  - `test_list_nodes_returns_fake_nodes` — `list_json("nodes", cluster_scoped=True)` returns node dicts
  - `test_add_node_pool_creates_nodes` — `add_node_pool` with attributes creates N nodes
  - `test_transition_pod` — manual phase transition helper
  - `test_delete_pod_releases_resources` — deleting a pod frees committed resources on its node

**Key detail:** The existing `_validate_manifest` check for `cloud.google.com/gke-nodepool` at line 101 should be folded into the scheduling logic — if no node has the label, the pod goes Pending. The explicit pool name set check can remain as a fast-fail for unknown pool names.

### Task 4: GcpServiceImpl enhancements (no deps)

**Files to modify:**

- `lib/iris/src/iris/cluster/platform/gcp_service_impl.py`:
  - Add `_available_types_by_zone: dict[str, set[str]] | None = None` to `__init__` (line ~210)
  - Add duplicate detection in `tpu_create` before line 347: raise `PlatformError` if `(name, zone)` in `_tpus`
  - Change TPU initial state from `"READY"` to `"CREATING"` at line 349
  - Add per-type-per-zone check after quota check at line ~333
  - Add `set_available_types_by_zone()` test helper
  - Add `advance_tpu_state()` test helper
  - Add duplicate detection in `vm_create` for `(name, zone)` in `_vms`
  - Add VM quota enforcement matching TPU pattern (add `_vm_zone_quotas`)

- `lib/iris/tests/cluster/platform/test_gcp_service_impl.py` — add tests:
  - `test_tpu_create_duplicate_raises` — second create with same (name, zone) raises
  - `test_tpu_create_state_creating` — new TPU starts in CREATING state
  - `test_advance_tpu_state` — CREATING → READY via helper
  - `test_tpu_type_zone_restriction` — type not available in specific zone
  - `test_vm_create_duplicate_raises`
  - `test_vm_quota_enforcement`

### Task 5: Update existing tests for state transition change (depends on Task 4)

When TPUs start in `CREATING` instead of `READY`, existing tests that assert `state == "READY"` after create will break.

**Files to modify:**
- `lib/iris/tests/cluster/platform/test_gcp_service_impl.py` — update assertions to expect `"CREATING"` and call `advance_tpu_state("READY")` where needed
- Any integration tests that create TPUs and immediately check state
- `lib/iris/tests/cluster/platform/` — grep for `state.*READY` after `tpu_create`

**Tests:** All existing GCP service tests must pass.

---

## Dependency Graph

```
Task 1 (k8s_types extraction)
    └── Task 2 (SubprocessK8s protocol + retype)

Task 3 (K8sServiceImpl rebuild)  ← independent of Tasks 1-2

Task 4 (GcpServiceImpl enhancements)
    └── Task 5 (update tests for state change)
```

Tasks 1, 3, 4 can all run in parallel. Task 2 depends on Task 1. Task 5 depends on Task 4.

After all merge, run the full test suite: `uv run pytest lib/iris/tests/ -x`

---

## Risks and Open Questions

1. **Breaking existing test setup patterns.** Many tests call `add_node_pool("pool-name")` with no attributes. The new signature needs backward compat via defaults (`node_count=1`, default CPU-only resources). Existing callers that don't pass attributes get a single CPU node — which may cause pods requesting GPUs to go Pending when they previously succeeded silently. This is *correct behavior* but will require updating those tests.

2. **`_query_capacity` skips tainted nodes.** `provider.py:1110` skips nodes with `NoSchedule`/`NoExecute` taints when computing capacity. GPU nodes have `nvidia.com/gpu` NoSchedule taint. This means GPU capacity is excluded from `ClusterCapacity`. The fake must replicate this — GPU nodes created by `add_node_pool(taints=[NVIDIA_GPU_TOLERATION])` should be skipped by `_query_capacity`. This is correct K8s behavior but may surprise test authors.

3. **Resource release on delete.** When `delete("pod", name)` is called, committed resources on the node must be released. This requires tracking which node a pod was scheduled on (via `spec.nodeName` set during scheduling).

4. **TPU CREATING→READY migration.** Changing TPU initial state to `CREATING` will break any test or production code path that creates a TPU and immediately treats it as usable. The `GcpPlatform` likely polls for `READY` state, but we need to verify every call site.

5. **`popen()` in tests.** If any test creates a `CoreWeavePlatform` with `K8sServiceImpl`, it will fail because `K8sServiceImpl` doesn't implement `popen()`. This is correct — `CoreWeavePlatform` should only be used with `Kubectl` (CLOUD) or a test double that implements `SubprocessK8s`. We need to verify no test does this.

6. **Pod affinity.** `_build_pod_manifest` adds `podAffinity` for multi-task jobs at `provider.py:436`. The fake does NOT implement affinity evaluation (it's a soft preference, `preferredDuringScheduling`). This is acceptable — affinity is best-effort in real K8s too, and the fake can ignore it without causing false test failures.
