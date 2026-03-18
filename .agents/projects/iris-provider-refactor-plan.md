# Iris Provider Refactor — Execution Plan

**Branch**: `multi/KUA9Ot3s`
**PR**: #3806 — iris: DirectTaskProvider k8s integration

This plan supersedes previous work. It addresses feedback from Codex PR review,
user feedback on platform/provider distinction, test structure issues, and missing
constraint mapping.

---

## Context

The PR introduced `DirectTaskProvider` (a Protocol) and `KubernetesProvider`
(the concrete implementation). Issues found:

1. **`isinstance` dispatching on Protocols** — `main.py` and `controller.py` use
   `isinstance(provider, KubernetesProvider)` which defeats Protocol structural
   typing. Replace with `is_direct_provider: bool` property on both `TaskProvider`
   and `DirectTaskProvider` protocols.

2. **`test_direct_provider.py` tests DB transitions, not a Protocol** — Protocols
   cannot be usefully tested in isolation. The DB transition tests
   (`drain_for_direct_provider`, `apply_direct_provider_updates`) belong in
   `test_transitions.py`. Delete `test_direct_provider.py`.

3. **Constraint mapping missing** — `_build_pod_manifest` ignores
   `run_req.constraints`. On CoreWeave, nodes are labeled with `pool=<name>` and
   `region=<region>`. Iris constraints like `pool=h100-8x` must map to k8s
   `nodeSelector` (and GPU tolerations). Without this, tasks run on wrong nodes.

4. **CoreWeave platform vestigial code** — `platform/coreweave.py` is 1558 lines.
   The bulk (~1200 lines) is slice/worker management (NodePools, worker Pods,
   bootstrapping). With `KubernetesProvider`, there are no Iris worker nodes.
   This dead code must be deleted.

5. **Codex bugs to fix**:
   - P1: `GetKubernetesClusterStatus` reads `self._controller.provider` which
     asserts when using DirectTaskProvider (should use `_direct_provider`)
   - P2: Worker-status log fetch lost `tail=True` (regression)
   - P2: `log_substring`/`min_log_level` dropped when forwarding
     `GetProcessStatusRequest`
   - P2: Cursor semantics inconsistent between primary/fallback log paths

---

## Architecture After This Plan

### Protocol dispatch (no isinstance)

```python
# provider.py — TaskProvider gets is_direct_provider = False
class TaskProvider(Protocol):
    @property
    def is_direct_provider(self) -> bool:
        return False
    ...

# direct_provider.py — DirectTaskProvider gets is_direct_provider = True
class DirectTaskProvider(Protocol):
    @property
    def is_direct_provider(self) -> bool:
        return True
    ...
```

`Controller.__init__` accepts `provider: TaskProvider | DirectTaskProvider`
(single field, not two). Dispatches via `self._provider.is_direct_provider`.

### Constraint mapping in KubernetesProvider

`_build_pod_manifest` maps Iris constraints to k8s scheduling fields:

```python
# Constraint(key="pool", op=EQ, value="h100-8x")
# → spec.nodeSelector["iris.pool"] = "h100-8x"
# Constraint(key="region", op=EQ, value="US-WEST-04A")
# → spec.nodeSelector["iris.region"] = "US-WEST-04A"
# GPU nodes on CoreWeave → toleration for qos.coreweave.cloud/interruptable
```

The mapping from Iris constraint keys to k8s label keys must be documented in
config (or derived from `platform.label_prefix`).

### CoreWeave platform — stripped to controller lifecycle only

`CoreweavePlatform` keeps only:
- `start_controller()` / `stop_controller()` / `restart_controller()`
- `stop_all()` (deletes controller resources only)
- `discover_controller()`
- `tunnel()` / `resolve_image()` / `shutdown()`
- `ensure_rbac()` / `_ensure_s3_credentials_secret()`

**Deleted** from CoreWeave platform:
- `CoreweaveWorkerHandle` class (entire class, ~110 lines)
- `CoreweaveSliceHandle` class (entire class, ~90 lines)
- `create_vm()` (returns NotImplemented)
- `create_slice()` / `_monitor_slice()` / `_create_worker_pod()` (~400 lines)
- `_wait_for_pod_ready()` / `_get_pod_ip()`
- `list_slices()` / `list_all_slices()` / `_list_slices_by_labels()` / `list_vms()`
- `ensure_nodepools()` / `_delete_stale_nodepools()` / `_ensure_one_nodepool()`
- Worker-specific helpers: `_worker_pod_name`, `_worker_config_cm_name`,
  `_resource_labels`, `_nodepool_name`

Scale groups in coreweave.yaml are used only for constraint labeling, not
NodePool provisioning. The `stop_all()` on CoreWeave deletes controller
resources only (no slices).

---

## Work Breakdown

### Stage 1: Protocol + isinstance cleanup

**File**: `lib/iris/src/iris/cluster/controller/provider.py`
- Add `is_direct_provider: bool = False` property to `TaskProvider` protocol

**File**: `lib/iris/src/iris/cluster/controller/direct_provider.py`
- Add `is_direct_provider: bool = True` property to `DirectTaskProvider` protocol

**File**: `lib/iris/src/iris/cluster/controller/controller.py`
- Change `_provider: TaskProvider | None` + `_direct_provider: DirectTaskProvider | None`
  to a single `_provider: TaskProvider | DirectTaskProvider` field
- All dispatch on `isinstance(provider, KubernetesProvider)` → `provider.is_direct_provider`
- All access to `self._direct_provider` → `self._provider` (guarded by `is_direct_provider`)
- All access to `self._provider` (cast to TaskProvider) → explicit narrowing

**File**: `lib/iris/src/iris/cluster/controller/main.py`
- Remove `isinstance(provider, KubernetesProvider)` — use `provider.is_direct_provider`
- Remove import of `KubernetesProvider` at module level (keep in `make_provider`)

**File**: `lib/iris/src/iris/cluster/controller/service.py`
- `GetKubernetesClusterStatus` reads from `_direct_provider` (fix P1 assertion bug)
- Restore `tail=True` for worker-status log fetch (fix P2 regression)
- Forward `log_substring`/`min_log_level` in `GetProcessStatusRequest` (fix P2)
- Use `provider.is_direct_provider` instead of `isinstance`

**Tests**: Add/update tests in `test_service.py` to cover has_direct_provider dispatch.

### Stage 2: Move transition tests, delete test_direct_provider.py

**Action**: Move all test functions from `test_direct_provider.py` to
`test_transitions.py` (they test `ControllerTransitions` methods). Delete
`test_direct_provider.py`.

**Verification**: All tests in `test_transitions.py` pass.

### Stage 3: Constraint mapping in KubernetesProvider

**File**: `lib/iris/src/iris/cluster/controller/kubernetes_provider.py`

Add `_constraints_to_node_selector` helper:

```python
# Well-known Iris constraint keys → k8s node label keys
# Labels on CoreWeave nodes are set by the NodePool (iris.pool, iris.region)
_CONSTRAINT_KEY_TO_NODE_LABEL: dict[str, str] = {
    "pool": "iris.pool",      # matches nodepool label applied to nodes
    "region": "iris.region",  # set on nodes by CoreWeave NodePool
}

def _constraints_to_node_selector(
    constraints: list[cluster_pb2.Constraint],
) -> dict[str, str]:
    """Map Iris constraints to k8s nodeSelector entries.

    Only EQ constraints with known label keys are mapped. IN/NE/GT constraints
    are logged and ignored (affinity rules would be needed, not implemented here).
    """
    node_selector: dict[str, str] = {}
    for c in constraints:
        label_key = _CONSTRAINT_KEY_TO_NODE_LABEL.get(c.key)
        if label_key is None:
            logger.debug("Ignoring unmapped constraint key: %s", c.key)
            continue
        if c.op == cluster_pb2.Constraint.OP_EQ:
            node_selector[label_key] = c.value
        else:
            logger.warning("Unsupported constraint op %s for key %s; ignoring", c.op, c.key)
    return node_selector
```

In `_build_pod_manifest`:
```python
node_selector = _constraints_to_node_selector(list(run_req.constraints))
if node_selector:
    spec["nodeSelector"] = node_selector

# Add GPU toleration for CoreWeave when GPU requested
if gpu_count > 0:
    spec.setdefault("tolerations", []).append(_CW_INTERRUPTABLE_TOLERATION)
```

The `_CW_INTERRUPTABLE_TOLERATION` constant must move from `coreweave.py` to
`kubernetes_provider.py` or a shared `k8s/constants.py`.

**Tests**: `test_kubernetes_provider.py` must test:
1. Pod manifest with EQ constraint → nodeSelector populated
2. Pod manifest with pool=h100-8x (from coreweave.yaml config) → correct label
3. GPU request → interruptable toleration added
4. Unknown constraint key → ignored (no crash)
5. Non-EQ constraint → logged, not added to nodeSelector

### Stage 4: Integration test with coreweave.yaml config

**File**: `lib/iris/tests/cluster/controller/test_kubernetes_provider.py`

Add `test_coreweave_full_lifecycle` using mock kubectl:

```python
def test_coreweave_full_lifecycle(tmp_path):
    """Full task lifecycle using KubernetesProvider configured from coreweave.yaml.

    Verifies that constraints from scale group config (pool, region) are
    translated to k8s nodeSelector entries in the pod manifest.
    """
    config_path = Path(__file__).parents[5] / "examples/coreweave.yaml"
    cluster_config = load_config(config_path)

    # Build provider from config
    kubectl = FakeKubectl()  # records apply_json calls
    provider = KubernetesProvider(
        kubectl=kubectl,
        namespace="iris",
        default_image=cluster_config.kubernetes_provider.default_image,
    )

    # Simulate submitting a task with h100-8x constraints
    run_req = _make_run_req("/my-job/task-0", attempt_id=1)
    run_req.constraints.extend([
        cluster_pb2.Constraint(key="pool", op=cluster_pb2.Constraint.OP_EQ, value="h100-8x"),
        cluster_pb2.Constraint(key="region", op=cluster_pb2.Constraint.OP_EQ, value="US-WEST-04A"),
    ])
    run_req.resources.device.gpu.count = 8

    batch = DirectProviderBatch(tasks_to_run=[run_req], running_tasks=[], tasks_to_kill=[])
    result = provider.sync(batch)

    # Verify pod was submitted with correct nodeSelector
    assert len(kubectl.applied_manifests) == 1
    spec = kubectl.applied_manifests[0]["spec"]
    assert spec["nodeSelector"]["iris.pool"] == "h100-8x"
    assert spec["nodeSelector"]["iris.region"] == "US-WEST-04A"
    assert any(t["key"] == "qos.coreweave.cloud/interruptable" for t in spec.get("tolerations", []))
    assert not result.updates  # no running tasks to poll
```

Use a `FakeKubectl` that records calls (no MagicMock), similar to existing fake
patterns in the codebase.

### Stage 5: Strip CoreWeave platform

**File**: `lib/iris/src/iris/cluster/platform/coreweave.py`

Delete:
- `CoreweaveWorkerHandle` class (lines ~129–238)
- `CoreweaveSliceHandle` class (lines ~240–328)
- `_worker_pod_name`, `_worker_config_cm_name` helper functions
- `CoreweavePlatform.create_vm` (stub returning NotImplementedError)
- `CoreweavePlatform.create_slice` and `_monitor_slice`, `_create_worker_pod`
- `CoreweavePlatform._wait_for_pod_ready`, `_get_pod_ip`
- `CoreweavePlatform.list_slices`, `list_all_slices`, `_list_slices_by_labels`
- `CoreweavePlatform.list_vms`
- `CoreweavePlatform.ensure_nodepools`, `_delete_stale_nodepools`, `_ensure_one_nodepool`
- `CoreweavePlatform._resource_labels`, `_nodepool_name`
- `_POD_READY_TIMEOUT` constant (only used in worker pod wait)

Update `stop_all` to delete only controller resources (no NodePool enumeration).

Update `Platform` protocol in `base.py`:
- `create_slice` / `list_slices` / `list_all_slices` / `list_vms` / `create_vm`
  should have default `NotImplementedError` bodies, or CoreWeave must explicitly
  raise `PlatformUnsupportedError` from them.

Update any call sites in `cli/`, `cluster/manager.py`, `local_cluster.py` that
call `platform.create_slice()` / `platform.list_all_slices()` — guard with
`hasattr` or dispatch via config type check.

Actually: the correct approach is to add `supports_slices: bool` property to
Platform protocol (returns True for GCP/Manual/Local, False for CoreWeave).
Callers check this before calling slice ops.

**Tests**: Update/remove `tests/cluster/platform/` CoreWeave platform tests that
tested worker slice creation.

---

## Stage ordering and dependencies

```
Stage 1: Protocol cleanup + isinstance removal (no external deps)
Stage 2: Move tests (depends on Stage 1 being clean)
Stage 3: Constraint mapping (independent, can parallel with Stage 2)
Stage 4: Integration test (depends on Stage 3)
Stage 5: CoreWeave strip (independent, can parallel with Stage 1-4)
```

Stages 1+3 can start in parallel.
Stage 5 can start immediately.
Stage 2 after Stage 1.
Stage 4 after Stage 3.

---

## Files to change

| File | Change |
|------|--------|
| `controller/provider.py` | Add `is_direct_provider = False` to TaskProvider |
| `controller/direct_provider.py` | Add `is_direct_provider = True` to DirectTaskProvider |
| `controller/controller.py` | Single `_provider` field, dispatch via `is_direct_provider` |
| `controller/main.py` | Remove isinstance, use `is_direct_provider` |
| `controller/service.py` | Fix P1/P2 bugs, use `is_direct_provider` |
| `controller/kubernetes_provider.py` | Add constraint→nodeSelector mapping |
| `tests/cluster/controller/test_direct_provider.py` | DELETE, merge to test_transitions.py |
| `tests/cluster/controller/test_transitions.py` | Add direct provider transition tests |
| `tests/cluster/controller/test_kubernetes_provider.py` | Add constraint + coreweave lifecycle tests |
| `platform/coreweave.py` | Delete ~1200 lines of slice/worker code |
| `platform/base.py` | Add `supports_slices: bool` to Platform protocol |

---

## Tests to run

```bash
# After each stage
uv run pytest lib/iris/tests/ -m "not e2e" -o "addopts=" -x
```
