# CoreWeave: Shared NodePool Model

## Context

Currently, Iris creates **one NodePool per slice** on CoreWeave. Each `create_slice()` call provisions a new bare-metal node via a new NodePool CRD, taking 20-30 minutes. When the slice is terminated, the NodePool is deleted. This means every scale-up event pays the full bare-metal boot penalty.

The goal is to switch to a **single pre-allocated NodePool** (per scale group / instance type) with CoreWeave autoscaling enabled. Iris creates/deletes Pods; CoreWeave handles node provisioning.

## Current Architecture (what needs to change)

**Slice = NodePool + Pod** (1:1:1 mapping):
- `create_slice()` → create NodePool (`targetNodes: 1`, `autoscaling: false`) → wait 20-30 min for node → create Pod with `nodeSelector: {iris-slice-id: <unique>}` → wait for Pod ready
- `terminate()` → delete Pod → delete NodePool
- `list_slices()` → list NodePools by label → discover Pods per NodePool

**Controller** also gets its own NodePool (`iris-controller-pool`).

## Target Architecture

**Slice = Pod** (on a shared NodePool):
- NodePool is pre-allocated (operator-managed or created once by `start_controller`)
- `create_slice()` → create Pod with resource requests (GPU count triggers CoreWeave autoscaler to add nodes) → wait for Pod ready
- `terminate()` → delete Pod (CoreWeave autoscaler may scale down idle nodes)
- `list_slices()` → list Pods by label (not NodePools)

One shared NodePool per instance type (i.e., per scale group that differs in `instance_type`).

## Changes Required

### 1. Proto/Config Changes (`lib/iris/src/iris/rpc/config.proto`)

Add a `nodepool_name` field to `CoreweaveSliceConfig`:

```protobuf
message CoreweaveSliceConfig {
  string region = 1;
  string instance_type = 2;
  int32 gpus_per_node = 3;
  string gpu_class = 4;
  bool infiniband = 5;
  string nodepool_name = 6;  // Pre-allocated NodePool name. If set, skip NodePool creation.
}
```

Optionally, add to `CoreweavePlatformConfig`:

```protobuf
message CoreweavePlatformConfig {
  string region = 1;
  string namespace = 2;
  string kubeconfig_path = 3;
  string controller_nodepool_name = 4;  // Pre-allocated controller NodePool
}
```

### 2. CoreWeave Platform (`lib/iris/src/iris/cluster/platform/coreweave.py`)

This is the bulk of the work.

#### a) `CoreweaveSliceHandle` — decouple from NodePool

Currently wraps a NodePool. Change to wrap a **Pod**:
- `slice_id` stays (but is now a Pod-based identifier, not a NodePool name)
- `terminate()`: delete Pod only, **not** the NodePool
- Remove NodePool-specific logic from the handle

#### b) `create_slice()` (lines 341-401) — skip NodePool creation

```python
def create_slice(self, config, bootstrap_config):
    cw = config.coreweave

    # If nodepool_name is set, use shared pool model
    if cw.nodepool_name:
        # Generate slice_id from Pod naming (not NodePool)
        slice_id = f"{self._label_prefix}-{config.name_prefix}-{Timestamp.now().epoch_ms()}"
        labels = self._resource_labels(config.name_prefix, slice_id)
        handle = CoreweaveSliceHandle(...)

        # Skip NodePool creation entirely
        # Go straight to Pod creation + readiness wait
        self._executor.submit(self._monitor_slice_pod_only, handle, config, bootstrap_config)
        return handle

    # ... existing NodePool-per-slice logic (keep for backward compat during transition)
```

#### c) `_monitor_slice()` — new Pod-only path

New method `_monitor_slice_pod_only()`:
1. Skip `_wait_for_nodepool_ready()` entirely
2. Call `_create_worker_pod()` directly
3. Wait for Pod ready (Pod may pend while CoreWeave autoscaler adds a node)
4. Increase `_POD_READY_TIMEOUT` for this path (Pod pending = node provisioning)

#### d) `_create_worker_pod()` (lines 470-563) — change nodeSelector

Currently:
```python
"nodeSelector": {
    self._slice_id_label_key(): handle.slice_id,  # pins to specific NodePool node
}
```

Change to:
```python
"nodeSelector": {
    # Target any node in the shared NodePool (uses NodePool's nodeLabels)
    f"{self._label_prefix}-scale-group": handle.scale_group,
}
```

The shared NodePool's `nodeLabels` must include the scale-group label so its nodes match this selector.

GPU resource requests (`nvidia.com/gpu: 8`) naturally ensure one Pod per node.

#### e) `terminate()` on CoreweaveSliceHandle (line 269-277)

Change from:
```python
def terminate(self):
    self._kubectl.delete("pod", pod_name, force=True)
    self._kubectl.delete("nodepool", self._slice_id, cluster_scoped=True)  # REMOVE THIS
```

To just:
```python
def terminate(self):
    self._kubectl.delete("pod", pod_name, force=True)
    # NodePool is shared; don't delete it
```

#### f) `list_slices()` / `_list_slices_by_labels()` (lines 592-667)

Currently lists NodePools and discovers Pods per NodePool. Change to:
- List **Pods** by label (the Pod IS the slice now)
- Derive slice state from Pod phase/readiness directly
- No NodePool query needed per slice

#### g) `start_controller()` (lines 722-808)

Two options:
1. **Keep controller NodePool separate** (simpler, controller is CPU-only)
2. **Use pre-allocated controller NodePool** via `controller_nodepool_name` config

For option 1, keep existing logic. For option 2, skip NodePool creation when `controller_nodepool_name` is set (similar pattern to slice changes).

The controller Deployment's `nodeSelector` would target the pre-allocated pool's labels instead of `iris-role: controller`.

#### h) `stop_all()` (lines 821-856)

Currently deletes all NodePools + controller resources. Change to:
- Delete all managed **Pods** (not NodePools)
- Delete controller Deployment/Service/ConfigMap
- **Do NOT delete pre-allocated NodePools** — they are operator-managed
- May still need to delete dynamically-created NodePools if backward compat is needed

#### i) `reload()` (lines 858-935)

Reload stays mostly the same (delete Pod, recreate Pod). No NodePool changes needed since NodePool is persistent.

### 3. Config Example (`lib/iris/examples/coreweave.yaml`)

Add `nodepool_name` to each scale group's slice template:

```yaml
scale_groups:
  h100_8x:
    slice_template:
      coreweave:
        region: US-WEST-04A
        instance_type: gd-8xh100ib-i128
        nodepool_name: iris-h100-pool  # pre-allocated
```

### 4. K8s Manifests (`infra/coreweave/k8s/`)

Add a NodePool manifest for operator pre-allocation:

```yaml
# infra/coreweave/k8s/nodepool-h100.yaml
apiVersion: compute.coreweave.com/v1alpha1
kind: NodePool
metadata:
  name: iris-h100-pool
  labels:
    iris-managed: "true"
    iris-scale-group: h100_8x
spec:
  computeClass: default
  instanceType: gd-8xh100ib-i128
  targetNodes: 0        # start with 0; CoreWeave autoscaler grows on demand
  autoscaling: true     # let CoreWeave manage node count
  nodeLabels:
    iris-managed: "true"
    iris-scale-group: h100_8x
```

### 5. Smoke Test (`lib/iris/scripts/smoke-test.py`)

- **Timeouts**: Pod scheduling on existing nodes is fast (seconds vs 30 min). But if CoreWeave autoscaler needs to add a node, it still takes 20-30 min. Keep generous timeouts.
- **Cleanup**: `_cleanup_existing()` should not delete the pre-allocated NodePool. May need to just delete Pods.
- **Redeploy mode**: Works naturally — Pods are recreated, NodePool persists.

### 6. Debug Loop (`lib/iris/scripts/debug-coreweave-loop.py`)

- `cleanup_kubernetes()` (lines 107-166): Currently deletes all NodePools with `iris-managed=true`. Change to only delete Pods, or skip NodePool deletion for pre-allocated pools.
- Need to distinguish "pre-allocated" vs "dynamically created" NodePools. The `nodepool_name` in config is the signal.

### 7. ScalingGroup (`lib/iris/src/iris/cluster/controller/scaling_group.py`)

- `reconcile()`: Currently discovers slices via `platform.list_slices()` which lists NodePools. With the new model, `list_slices()` lists Pods, so reconcile works unchanged (it just calls `list_slices`).
- `scale_up()` / `scale_down()`: No changes — they call `platform.create_slice()` and `handle.terminate()`.

### 8. Autoscaler (`lib/iris/src/iris/cluster/controller/autoscaler.py`)

No changes needed. The autoscaler interacts with slices through the Platform/SliceHandle abstractions. The fact that a "slice" is now a Pod instead of a NodePool+Pod is transparent.

## Key Design Decisions

1. **One NodePool per instance type**: Different GPU types (H100, A100, CPU) each need their own NodePool since `instanceType` is immutable per NodePool.

2. **Pod resource requests drive autoscaling**: When a Pod requests `nvidia.com/gpu: 8`, and no node has capacity, the CoreWeave cluster autoscaler provisions a new node in the NodePool. This is the standard Kubernetes autoscaler pattern.

3. **Backward compatibility**: Keep the old NodePool-per-slice path gated on `if not cw.nodepool_name`. Configs without `nodepool_name` continue to work as before. This allows incremental migration.

4. **Node-to-Pod affinity**: GPU resource requests (8 GPUs per node) naturally ensure one worker Pod per node. No explicit anti-affinity needed if nodes have exactly 8 GPUs.

## File Summary

| File | Change Scope |
|------|-------------|
| `lib/iris/src/iris/rpc/config.proto` | Add `nodepool_name` field |
| `lib/iris/src/iris/cluster/platform/coreweave.py` | Major: slice model, create/terminate/list |
| `lib/iris/examples/coreweave.yaml` | Add `nodepool_name` to scale groups |
| `infra/coreweave/k8s/` | Add pre-allocated NodePool manifests |
| `lib/iris/scripts/smoke-test.py` | Cleanup changes (don't delete NodePool) |
| `lib/iris/scripts/debug-coreweave-loop.py` | Cleanup changes |
| `lib/iris/src/iris/cluster/controller/scaling_group.py` | No changes |
| `lib/iris/src/iris/cluster/controller/autoscaler.py` | No changes |

## Verification

1. Run existing CoreWeave unit tests: `uv run pytest lib/iris/tests/ -x -k coreweave`
2. Smoke test with pre-allocated NodePool: `uv run python lib/iris/scripts/smoke-test.py --config lib/iris/examples/coreweave.yaml`
3. Verify scale-up: submit a job, confirm Pod is created on shared pool, CoreWeave autoscaler adds node if needed
4. Verify scale-down: terminate slice, confirm only Pod is deleted, NodePool persists
5. Verify `stop_all`: confirm NodePool is preserved, only Pods + controller resources are cleaned up
