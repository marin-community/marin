# Service Implementation Gaps Report

## 1. K8sServiceImpl: Current Capabilities and Gaps

### What it does today

The `K8sServiceImpl` (`lib/iris/src/iris/cluster/k8s/k8s_service_impl.py`) is a simple
in-memory key-value store keyed by `(kind, name)`:

**Validation on `apply_json`:**
- Manifest must have `kind` and `metadata.name`
- Node pool selector (`cloud.google.com/gke-nodepool`) checked against `_available_node_pools` set
- Container resource request/limit keys checked against `VALID_RESOURCE_TYPES` allowlist

**State tracking:**
- `_resources: dict[tuple[str, str], dict]` — stores raw manifest dicts
- `list_json` supports label filtering
- Resource type normalization (plural → singular, case-insensitive)
- Failure injection (one-shot per operation)
- Pre-populated logs, events, exec responses, file contents, top_pod overrides

**Test helpers:**
- `add_node_pool` / `remove_node_pool` — manipulate the pool name set (just strings, no attributes)
- `set_logs`, `add_event`, `set_exec_response`, `set_file_content`, `set_top_pod`

### What's missing for a "replica of K8s"

1. **No scheduling semantics.** `apply_json` stores the manifest and returns. Real K8s scheduling involves:
   - **Resource fitting**: Do any nodes have enough CPU/memory/GPU to satisfy `resources.requests`?
   - **nodeSelector matching**: Does a node exist with matching labels?
   - **Toleration matching**: Does the pod tolerate node taints?
   - **Affinity/anti-affinity**: Are `podAffinity`/`podAntiAffinity` rules satisfied?
   - **Pod status transitions**: Pods should go Pending → Running (or stay Pending if unschedulable)

2. **No node model.** There are no nodes at all — just a set of pool name strings. Real scheduling needs:
   - Node objects with `metadata.labels`, `spec.taints`, `status.allocatable`
   - Per-node tracking of committed resources (what's already scheduled)
   - Ability to query nodes (KubernetesProvider._query_capacity does `list_json("nodes")`)

3. **No pod lifecycle.** Pods are stored as raw dicts with no status. Real K8s returns:
   - `status.phase` (Pending/Running/Succeeded/Failed)
   - `status.containerStatuses[].state` (waiting/running/terminated)
   - `status.conditions` (PodScheduled, Initialized, Ready)

   KubernetesProvider._poll_pods reads `status.phase` and `status.containerStatuses` —
   the fake returns whatever was stored, meaning tests must manually set status fields.

4. **No events generation.** Real K8s generates events when scheduling fails ("FailedScheduling"),
   when pods start, etc. The fake only returns manually pre-populated events.

5. **No capacity computation.** `KubernetesProvider._query_capacity` lists nodes and sums
   allocatable resources minus running pod requests. The fake has no nodes to query.

6. **Node pool attributes are just names.** `add_node_pool("gpu-pool")` records a string.
   No GPU count, instance type, region, or resource capacity is associated with it.

## 2. Kubectl Usage Audit

### Files that import/use `Kubectl` directly

| File | Usage |
|------|-------|
| `lib/iris/src/iris/cluster/k8s/kubectl.py` | Definition of `Kubectl` class |
| `lib/iris/src/iris/cluster/k8s/__init__.py` | Re-exports `Kubectl` in `__all__` |
| `lib/iris/src/iris/cluster/k8s/k8s_service.py` | Imports `KubectlLogResult` (not `Kubectl` class itself) |
| `lib/iris/src/iris/cluster/k8s/k8s_service_impl.py` | Imports `KubectlError`, `KubectlLogLine`, `KubectlLogResult` |
| `lib/iris/src/iris/cluster/k8s/provider.py` | `KubernetesProvider.kubectl` typed as `K8sService` ✅, imports `KubectlLogLine` |
| `lib/iris/src/iris/cluster/platform/coreweave.py:48` | `from iris.cluster.k8s.kubectl import Kubectl` — **instantiates `Kubectl` directly** |
| `lib/iris/src/iris/cluster/config.py:1131` | `from iris.cluster.k8s.kubectl import Kubectl` — **instantiates `Kubectl` in `make_provider`** |
| `lib/iris/tests/kubernetes/test_kubectl.py` | Tests for the `Kubectl` class, imports `_parse_k8s_cpu`, `_parse_k8s_memory` |
| `lib/iris/tests/cluster/k8s/test_k8s_service_impl.py` | Imports `KubectlError` only |

### `Kubectl` methods NOT on `K8sService` protocol

| Method | Used by | Notes |
|--------|---------|-------|
| `popen()` | `coreweave.py:918` (_coreweave_tunnel) | Escape hatch for streaming/port-forward. Returns `subprocess.Popen`. |
| `run()` | Internal to Kubectl | Low-level subprocess runner. Not needed on protocol. |
| `prefix` property | Internal | Returns command prefix. Not needed. |

**Key finding**: `popen()` is the only `Kubectl` method used in production code that isn't on `K8sService`. The `_coreweave_tunnel` function takes `kubectl: Kubectl` (concrete type) and calls `kubectl.popen()`.

### Items shared between Kubectl and K8sServiceImpl via types

Both import from `kubectl.py`:
- `KubectlError` — used as the error type across both
- `KubectlLogLine`, `KubectlLogResult` — used for streaming logs

These types must remain in `kubectl.py` or be moved to a shared location.

### CoreWeave's direct `Kubectl` usage

`CoreWeavePlatform.__init__` creates `self._kubectl = Kubectl(...)` and uses it throughout
(37+ method calls via `self._kubectl.*`). All methods called are on the `K8sService` protocol
**except** `popen()` used in `_coreweave_tunnel`.

## 3. GcpServiceImpl: Current Capabilities and Gaps

### What it validates (all modes)

- **Resource name format**: lowercase alphanumeric/hyphens, starts with letter, max 63 chars
- **Zone**: checked against `KNOWN_GCP_ZONES` (16 zones) — except LOCAL mode skips zone validation for TPUs
- **Accelerator type**: checked against `KNOWN_TPU_TYPES` (derived from `TPU_TOPOLOGIES`)
- **Runtime version**: must be non-empty
- **Label format**: key matches `[a-z][a-z0-9_-]{0,62}`, value matches `[a-z0-9_-]{0,63}`
- **VM disk size**: must be positive

### DRY_RUN/LOCAL in-memory state

- `_tpus: dict[tuple[str, str], TpuInfo]` — keyed by (name, zone)
- `_vms: dict[tuple[str, str], VmInfo]` — keyed by (name, zone)
- Zone-level TPU quota via `set_zone_quota` / `_zone_quotas`
- Failure injection (one-shot per operation)
- Accelerator type management (`set_tpu_type_unavailable` / `add_tpu_type`)
- Synthetic network endpoints based on TPU topology VM count

### What's missing

1. **No per-accelerator-type quota.** Real GCP has quotas per accelerator type per zone/region,
   not just per zone. E.g., you might have quota for 8x v4-8 but 0x v5e-256.

2. **No region-level quota.** GCP TPU quotas are often region-level, not just zone-level.

3. **No duplicate name detection.** Creating two TPUs with the same (name, zone) silently
   overwrites the first. Real GCP returns ALREADY_EXISTS.

4. **No CREATING → READY state transition.** TPUs are created in "READY" state immediately.
   Real GCP starts in "CREATING" and transitions to "READY" over minutes.

5. **No error codes.** Real GCP returns structured errors (RESOURCE_EXHAUSTED, NOT_FOUND,
   ALREADY_EXISTS, PERMISSION_DENIED). The fake uses generic `PlatformError`.

6. **No VM quota enforcement.** Only TPU zone quotas are enforced; VMs have no quota checking.

7. **No accelerator type → zone availability mapping.** Not all TPU types are available in all
   zones. The fake allows any known type in any known zone.

## 4. Node Pool Model Requirements

### Current `add_node_pool` implementation

```python
def add_node_pool(self, pool_name: str) -> None:
    if self._available_node_pools is None:
        self._available_node_pools = set()
    self._available_node_pools.add(pool_name)
```

A node pool is **just a name string**. No attributes whatsoever.

### What real K8s/GKE node pools have

From `coreweave.py` NodePool manifests (`_nodepool_spec`, line ~800):
- `spec.instanceType` (e.g., "H100_80GB_SXM_NVLINK")
- `spec.targetNodes` / `spec.minNodes` / `spec.maxNodes`
- `spec.autoscaling` (bool)
- `metadata.labels` including `iris.pool`, `iris.region`, `iris.scale-group`
- Per-node resources: CPU, memory, GPU count (derived from instance type)

From `KubernetesProvider._build_pod_manifest` (provider.py):
- Pods get `nodeSelector` based on constraints (e.g., `iris.pool`, `iris.region`)
- Pods get `tolerations` for GPU nodes (`nvidia.com/gpu`, `qos.coreweave.cloud/interruptable`)
- Pods get `resources.limits` for `nvidia.com/gpu`, `rdma/ib`, CPU, memory, ephemeral-storage
- Pods get `podAffinity` for multi-task job colocation (`coreweave.cloud/spine` topology key)

### Node pool attributes needed for realistic scheduling

```
NodePool:
  name: str
  instance_type: str
  labels: dict[str, str]           # e.g., {"iris.pool": "gpu-a100", "iris.region": "us-east"}
  taints: list[dict]               # e.g., [{"key": "nvidia.com/gpu", "effect": "NoSchedule"}]
  node_count: int                  # current count of nodes
  per_node_resources:
    cpu_millicores: int
    memory_bytes: int
    gpu_count: int                 # 0 for CPU pools
    gpu_type: str                  # e.g., "nvidia.com/gpu"
    tpu_count: int                 # for TPU pools
    ephemeral_storage_bytes: int
```

## 5. Real K8s Scheduling Semantics the Fake Needs

### What KubernetesProvider actually puts on pods (provider.py)

1. **nodeSelector** (line 411-415): Constraints mapped to labels like `iris.pool`, `iris.region`.
   Plus `managed_label` if set.

2. **tolerations** (line 417-423): GPU pods get `CW_INTERRUPTABLE_TOLERATION` and
   `NVIDIA_GPU_TOLERATION`.

3. **resource requests/limits** (line 341-364):
   - `cpu` (millicores)
   - `memory` (bytes)
   - `nvidia.com/gpu` (count)
   - `rdma/ib` (count, for host_network + GPU)
   - `ephemeral-storage` (GiB)

4. **podAffinity** (line 436-452): Multi-task jobs prefer colocation on same
   `coreweave.cloud/spine`.

5. **activeDeadlineSeconds** (line 431-432): From task timeout.

### Minimum scheduling the fake should implement

**Must have (to catch real bugs):**
- Pods targeting a non-existent nodeSelector label should go Pending (not silently succeed)
- Pods requesting more resources than available should go Pending
- Pods without tolerations for tainted nodes should not be scheduled there
- Pod status should transition: applied → Pending, then scheduled → Running (or stay Pending)

**Nice to have:**
- Pod affinity preference (soft, used for colocation)
- activeDeadlineSeconds → eventual failure

### How the controller uses pod status

`KubernetesProvider._poll_pods` (provider.py:997-1085) reads:
- `status.phase`: Pending, Running, Succeeded, Failed
- `status.containerStatuses[0].state.terminated.reason`: OOMKilled, Evicted, etc.
- `status.containerStatuses[0].state.terminated.exitCode`
- `status.conditions[].reason/message`
- `spec.nodeName` (for cluster status reporting)

## 6. Real GCP Semantics the Fake Needs

### TPU provisioning

- **Zone/type availability**: Not all TPU types are in all zones. Could be modeled as
  `available_types_by_zone: dict[str, set[str]]`.
- **Quota**: Per-type per-region quotas (e.g., "v4-8 quota: 4 in us-central2").
- **State transitions**: CREATING → READY (with configurable delay for testing).
  DELETING state after delete (with eventual removal).
- **Error codes**: RESOURCE_EXHAUSTED (quota), NOT_FOUND (describe/delete nonexistent),
  ALREADY_EXISTS (duplicate create), PERMISSION_DENIED.
- **Network endpoints**: Already modeled correctly (based on TPU topology VM count).

### VM provisioning

- **Zone availability**: Already validated.
- **State transitions**: PROVISIONING → STAGING → RUNNING.
- **Duplicate detection**: ALREADY_EXISTS on duplicate name.

## 7. Specific Recommendations

### 7.1 K8sServiceImpl: Add node/scheduling model

**Add a Node data model:**
```python
@dataclass
class FakeNode:
    name: str
    labels: dict[str, str]
    taints: list[dict]
    allocatable: dict[str, str]  # cpu, memory, nvidia.com/gpu, etc.
```

**Add node pool → node creation:**
```python
def add_node_pool(self, pool_name: str, *,
                  node_count: int = 1,
                  labels: dict[str, str] | None = None,
                  taints: list[dict] | None = None,
                  allocatable: dict[str, str] | None = None) -> None:
```

**Add scheduling logic in `apply_json`:**
1. Extract pod spec (already done via `_pod_spec`)
2. Find candidate nodes matching `nodeSelector`
3. Filter by toleration matching
4. Check resource fit
5. If no node fits → pod gets `status.phase = "Pending"` with appropriate condition
6. If a node fits → pod gets `status.phase = "Running"`, `spec.nodeName = node.name`

**Add pod status management:**
- `apply_json` should set initial status based on scheduling result
- Add `transition_pod(name, phase, ...)` test helper for manual overrides
- Store `status` alongside the raw manifest

### 7.2 Kubectl removal plan

1. Move `KubectlError`, `KubectlLogLine`, `KubectlLogResult` to a shared module
   (e.g., `k8s_types.py`)
2. Add `popen()` to `K8sService` protocol (needed by CoreWeave tunnel), or extract
   tunnel to use a separate `PortForwarder` protocol
3. Change `coreweave.py` to type `self._kubectl` as `K8sService` (add `popen` to protocol
   or refactor tunnel)
4. Change `config.py:make_provider` to construct `Kubectl` but assign to `K8sService`-typed var
5. Keep `Kubectl` class as the CLOUD implementation but stop exporting from `__init__.py`

### 7.3 GcpServiceImpl enhancements

1. **Per-type-per-zone availability**: `set_available_types(zone, types)` or
   `available_types_by_zone: dict[str, set[str]]`
2. **Duplicate detection**: Check if (name, zone) already exists before creating
3. **ALREADY_EXISTS error**: New error subclass
4. **Optional state transitions**: `tpu_create` returns CREATING; test helper to advance to READY

### 7.4 What NOT to build

- Don't model DaemonSets, StatefulSets, Services — Iris only creates Pods and ConfigMaps
- Don't model inter-pod networking or DNS
- Don't model PersistentVolumeClaims
- Don't model full node autoscaling (that's CoreWeave/GKE's job)
- Don't model pod preemption priority (Iris doesn't use PriorityClasses)

## Appendix: File References

| File | Lines | Purpose |
|------|-------|---------|
| `k8s_service.py` | 89 | K8sService protocol (15 methods) |
| `k8s_service_impl.py` | 326 | In-memory fake (simple key-value store) |
| `kubectl.py` | 446 | Real kubectl wrapper (subprocess-based) |
| `provider.py` | 1215 | KubernetesProvider (pod lifecycle, capacity) |
| `gcp_service.py` | 118 | GcpService protocol (14 methods) |
| `gcp_service_impl.py` | 1038 | GCP fake (DRY_RUN/LOCAL with validation) |
| `coreweave.py` | 1296 | CoreWeave platform (37+ kubectl calls, uses `popen`) |
| `config.py:1131` | - | `make_provider` instantiates `Kubectl` |
| `constants.py` | 21 | GPU/CW toleration dicts |
| `test_k8s_service_impl.py` | 336 | K8s fake tests |
| `test_gcp_service_impl.py` | 370 | GCP fake tests |
