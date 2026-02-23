# CoreWeave Platform Integration

**Issue**: [#2822 -- Iris: Implement CoreWeave platform](https://github.com/marin-community/marin/issues/2822)

## 1. Overview

Iris runs on CoreWeave CKS (bare-metal Kubernetes) using a shared NodePool model.
Each Iris scale group maps to one CoreWeave NodePool with autoscaling enabled.
CoreWeave manages node provisioning and deprovisioning; Iris manages only Pods.
Tasks execute as independent Kubernetes Pods via `KubernetesRuntime` (Pod-per-task),
which replaced an originally-planned containerd/crictl approach during implementation.

Example config: `lib/iris/examples/coreweave.yaml`

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  CoreWeave CKS Cluster                                              │
│                                                                     │
│  ┌──────────────────────────────────┐                               │
│  │  Controller Deployment           │  <-- created by               │
│  │  (iris-controller)               │      start_controller()       │
│  │                                  │                               │
│  │  ghcr.io/.../iris-controller     │                               │
│  │  port 10000                      │                               │
│  │  in-cluster K8s auth             │  <-- ServiceAccount           │
│  │  /etc/iris/config.json           │  <-- ConfigMap                │
│  └────────┬─────────────────────────┘                               │
│           │                                                         │
│  Service: iris-controller-svc (ClusterIP:10000)                     │
│           │                                                         │
│  ┌────────▼─────────────────────────┐  ┌──────────────────────────┐ │
│  │  Shared NodePool: iris-h100-8x   │  │ Shared NodePool: ...     │ │
│  │  (one per scale group)           │  │ (one per scale group)    │ │
│  │  instanceType: gd-8xh100ib-i128 │  │                          │ │
│  │  autoscaling: true               │  │                          │ │
│  │  minNodes: 0, maxNodes: N        │  │                          │ │
│  │                                  │  │                          │ │
│  │  Pod: iris-worker-{slice-id}     │  │  Pod: iris-worker-...    │ │
│  │  (light: no GPU/RDMA requests)   │  │                          │ │
│  │    ↓                             │  │                          │ │
│  │  Pod: iris-task-{uuid}           │  │                          │ │
│  │  (claims GPU/RDMA from device    │  │                          │ │
│  │   plugin, hostNetwork: true)     │  │                          │ │
│  └──────────────────────────────────┘  └──────────────────────────┘ │
│                                                                     │
│  Operator-managed (one-time):                                       │
│    Namespace, ServiceAccount, ClusterRole, ClusterRoleBinding       │
│                                                                     │
│  Platform-managed (runtime, via start_controller / create_slice):   │
│    ConfigMap, NodePools, Controller Deployment+Service, Worker Pods  │
└─────────────────────────────────────────────────────────────────────┘
```

Key architectural properties:

- **Shared NodePool model**: One NodePool per scale group (not per slice). CoreWeave
  autoscaling is enabled (`autoscaling: true`). NodePool names follow
  `{label_prefix}-{scale_group_name}`. NodePools scale to zero when idle.
- **Controller as K8s Deployment**: Created by `start_controller()`, discovered by
  workers via in-cluster DNS (`iris-controller-svc.iris.svc.cluster.local:10000`).
- **KubernetesRuntime (Pod-per-task)**: Task Pods claim GPU/RDMA resources directly
  from the kubelet device plugin. Worker Pods are "light" (no GPU/RDMA requests).
- **hostNetwork**: Both worker and task Pods use `hostNetwork: true` for RDMA/GPU
  performance and flat-network endpoint registration.
- **In-cluster auth**: The controller uses the `iris-controller` ServiceAccount.
  No kubeconfig needed inside the cluster.
- **Public images**: All images on `ghcr.io/marin-community/` are public. No
  `imagePullSecrets` required.

## 3. Operator Setup Guide

### Prerequisites

- A CoreWeave CKS cluster (created via Console or Terraform)
- A kubeconfig downloaded from CoreWeave Console > Tokens
- Images pushed to `ghcr.io/marin-community/`

### Step 1: Save kubeconfig

```bash
mkdir -p ~/.kube
mv ~/Downloads/kubeconfig.yaml ~/.kube/coreweave-iris
export KUBECONFIG=~/.kube/coreweave-iris
kubectl cluster-info
```

### Step 2: Apply RBAC prerequisites (one-time)

```bash
kubectl apply -f infra/coreweave/k8s/
```

This creates the `iris` namespace, `iris-controller` ServiceAccount, ClusterRole
(with permissions for NodePools, Pods, Nodes, ConfigMaps, Events, Secrets), and
ClusterRoleBinding. See `infra/coreweave/k8s/` for the individual manifests.

### Step 3: Set S3 credentials (if using S3 storage)

```bash
export R2_ACCESS_KEY_ID=<your-r2-access-key-id>
export R2_SECRET_ACCESS_KEY=<your-r2-secret-access-key>
```

`iris cluster start` creates a K8s Secret (`iris-s3-credentials`) from these
environment variables automatically.

### Step 4: Start the cluster

```bash
iris --config=lib/iris/examples/coreweave.yaml cluster start
```

This is fully idempotent. It creates/reconciles:
1. S3 credentials Secret (if S3 storage URIs are configured)
2. ConfigMap (`iris-cluster-config`) with the cluster config as JSON
3. Shared NodePools (one per scale group, in parallel)
4. Controller Deployment (`iris-controller`)
5. Controller Service (`iris-controller-svc`, ClusterIP)

### Step 5: Use the cluster

```bash
iris --config=lib/iris/examples/coreweave.yaml cluster status
iris --config=lib/iris/examples/coreweave.yaml cluster dashboard
```

### Step 6: Stop

```bash
iris --config=lib/iris/examples/coreweave.yaml cluster stop
```

Deletes worker Pods and controller resources. NodePools are left in place (they
scale to zero when idle).

## 4. Key Design Decisions

### Shared NodePools with CoreWeave autoscaling

Each scale group maps to one shared NodePool with `autoscaling: true`. CoreWeave
provisions bare-metal nodes on demand when Pods are scheduled and deprovisions
them when idle. Iris does not manage node lifecycle directly.

NodePools are created idempotently by `ensure_nodepools()` during `start_controller()`.
Stale NodePools (from renamed/removed scale groups) are garbage-collected automatically.
For existing pools, `targetNodes` is clamped to `min(currentNodes, 1)` to prevent
runaway autoscaling from system pods.

### Controller as a Kubernetes Deployment

The controller runs as a single-replica Deployment scheduled onto the configured
`scale_group` NodePool. Workers discover it via K8s Service DNS. The controller
Pod uses in-cluster ServiceAccount auth for all kubectl operations.

Cost note: the smallest CoreWeave CPU instance (`cd-gp-i64-erapids`, 64 vCPU,
256 GB RAM) is overprovisioned for the controller. CoreWeave does not offer
smaller bare-metal nodes.

### Bootstrap via Platform.create_slice() with async state model

`create_slice()` returns a `SliceHandle` immediately in `CREATING` state. A
background thread drives the handle through `CREATING -> BOOTSTRAPPING -> READY`
(or `FAILED`). The autoscaler observes transitions via `handle.describe()` and
does not drive bootstrap logic.

On failure, the platform cleans up its own resources (deletes the worker Pod) and
marks the handle as `FAILED`. The autoscaler calls `handle.terminate()` as a
safety net.

### KubernetesRuntime for task execution (Pod-per-task)

Each task attempt is a separate Kubernetes Pod created by `KubernetesRuntime`.
Task Pods:
- Claim GPU/RDMA resources from the kubelet device plugin
- Use `hostNetwork: true` with `dnsPolicy: ClusterFirstWithHostNet`
- Get S3 credentials via `secretKeyRef` from the platform-managed Secret
- Use `emptyDir` for `/app` (workdir) so tasks can run on any node
- Materialize code bundles in-pod via fsspec
- Have `ownerReferences` pointing to the worker Pod for GC

The worker Pod intentionally does **not** request GPU/RDMA resources when
`runtime: kubernetes` is configured, so task Pods can claim them instead.

### Task networking via hostNetwork

All Pods use `hostNetwork: true`, bypassing the Kubernetes overlay network.
This preserves Iris's flat-network assumptions for endpoint registration,
peer-to-peer task communication, and RDMA performance.

### Reconcile-driven recovery

Correctness does not depend on in-memory thread state. After a controller restart,
`list_all_slices()` discovers existing worker Pods by labels and reconstructs
slice handles with the correct state based on Pod phase and readiness conditions.

## 5. Control Flow

### Cluster startup (`iris cluster start`)

`CoreweavePlatform.start_controller()` orchestrates the full startup sequence.
See `lib/iris/src/iris/cluster/platform/coreweave.py`.

1. Create S3 credentials Secret (if S3 storage configured)
2. Apply ConfigMap with cluster config
3. Create/reconcile all shared NodePools in parallel via `ensure_nodepools()`
4. Apply controller Deployment (with rollout restart)
5. Apply controller Service (ClusterIP)
6. Wait for Deployment availability (polls with early failure detection for
   image pull errors, crash loops, and volume mount failures)
7. Return controller address (K8s Service DNS)

### Scale-up (autoscaler creates a worker slice)

1. Autoscaler calls `create_slice(config, bootstrap_config)`
2. Platform generates slice ID: `{label_prefix}-{scale_group}-{timestamp_ms}`
3. Platform applies worker Pod to the scale group's shared NodePool via
   `nodeSelector` matching the scale group label
4. Platform returns `CoreweaveSliceHandle` immediately (state: CREATING)
5. Background thread:
   a. Transitions to BOOTSTRAPPING
   b. Creates worker Pod (image, ports, env from bootstrap_config)
   c. Polls Pod readiness (with early failure detection)
   d. On ready: extracts Pod IP, creates `CoreweaveWorkerHandle`, marks READY
   e. On failure: deletes Pod, marks FAILED

### Worker registration

Worker Pod runs `iris.cluster.worker.main serve --runtime=kubernetes`. It:
1. Reads config from ConfigMap mount (`/etc/iris/config.json`)
2. Discovers controller via `iris-controller-svc.iris.svc.cluster.local:10000`
3. Creates `KubernetesRuntime` (reads `IRIS_SERVICE_ACCOUNT_NAME`,
   `IRIS_S3_SECRET_NAME` from environment)
4. Registers with controller, enters heartbeat loop

### Task execution

Standard Iris flow. Controller assigns task via heartbeat RPC. Worker calls
`KubernetesRuntime.create_container()` which creates a task Pod. See
`lib/iris/src/iris/cluster/runtime/kubernetes.py`.

### Scale-down

1. Autoscaler selects the idle slice
2. `handle.terminate()` force-deletes the worker Pod
3. CoreWeave autoscaler deprovisions the bare-metal node when no Pods remain

## 6. Credentials Summary

### Operator-managed (one-time)

| Resource | Purpose | How to Obtain |
|----------|---------|---------------|
| CoreWeave API token | kubeconfig auth | Console > Tokens > Create Token |
| Kubeconfig file | Operator's kubectl access | Console > Tokens > Download Kubeconfig |
| `iris-controller` ServiceAccount | In-cluster K8s API auth | `kubectl apply -f infra/coreweave/k8s/` |

### Platform-managed (runtime, created by `iris cluster start`)

| Resource | Purpose | Created By |
|----------|---------|------------|
| `iris-s3-credentials` Secret | S3 object storage auth for controller/workers/tasks | `start_controller()`, from `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` env vars |
| `iris-cluster-config` ConfigMap | Cluster config for controller and workers | `start_controller()` |
| In-cluster ServiceAccount token | kubectl calls from controller Pod | Auto-mounted by Kubernetes |

The `kubeconfig_path` config field is only needed when running the CLI
**outside** the cluster (e.g., `iris cluster start` from a laptop). Inside the
cluster, Pods use in-cluster auth automatically.

## 7. Open Questions / Known Limitations

1. **Multi-node slices**: `num_vms > 1` is not supported and raises `ValueError`.
   InfiniBand co-scheduling for multi-node training needs investigation.

2. **NodePool rate limits**: Creating many NodePools at scale has not been
   validated with CoreWeave.

3. **Disk layout**: CoreWeave bare-metal nodes have a 15 GB RAM disk as root
   filesystem and multi-TB NVMe at `/mnt/local`. The `cache_dir` must point to
   NVMe (e.g., `/mnt/local/iris-cache`). Using the default root path will fill
   the RAM disk immediately.

4. **Task Pod GC**: `ownerReferences` on task Pods only trigger GC when the
   worker Pod object is deleted. If the worker crash-loops in place, stale task
   Pods can accumulate. See TODO in `kubernetes.py`.

## 8. References

- [CoreWeave CKS Introduction](https://docs.coreweave.com/docs/products/cks)
- [CKS Cluster Creation](https://docs.coreweave.com/docs/products/cks/clusters/create)
- [API Access Tokens and Kubeconfig](https://docs.coreweave.com/docs/products/cks/auth-access/manage-api-access-tokens)
- [CoreWeave Node Pools](https://docs.coreweave.com/docs/products/cks/nodes/nodes-and-node-pools)
- [CoreWeave Autoscaling](https://docs.coreweave.com/docs/products/cks/nodes/autoscaling)
- [CoreWeave GPU Instances](https://docs.coreweave.com/docs/products/instances/gpu-instances)
- [CoreWeave Terraform Provider](https://docs.coreweave.com/docs/products/cks/terraform/about)

### Source files

| File | Description |
|------|-------------|
| `lib/iris/src/iris/cluster/platform/coreweave.py` | CoreWeave platform implementation |
| `lib/iris/src/iris/cluster/runtime/kubernetes.py` | KubernetesRuntime (Pod-per-task) |
| `lib/iris/src/iris/cluster/k8s/kubectl.py` | Kubectl CLI wrapper |
| `lib/iris/examples/coreweave.yaml` | Example cluster config |
| `infra/coreweave/k8s/` | RBAC/namespace operator manifests |
| `lib/iris/AGENTS.md` | CoreWeave integration notes for agents |
