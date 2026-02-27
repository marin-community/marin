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
│  All resources auto-created by `iris cluster start`:                │
│    Namespace, ServiceAccount, ClusterRole, ClusterRoleBinding,      │
│    ConfigMap, NodePools, Controller Deployment+Service, S3 Secret   │
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
  Task Pods request `nvidia.com/gpu: N` and optionally `rdma/ib: 1`. They also
  receive tolerations for the `nvidia.com/gpu` NoSchedule taint on GPU nodes.
- **hostNetwork**: Both worker and task Pods use `hostNetwork: true` for RDMA/GPU
  performance and flat-network endpoint registration. `dnsPolicy` is set to
  `ClusterFirstWithHostNet` to preserve in-cluster DNS resolution.
- **In-cluster auth**: The controller uses the `iris-controller` ServiceAccount.
  No kubeconfig needed inside the cluster.
- **Public images**: All images on `ghcr.io/marin-community/` are public. No
  `imagePullSecrets` required.

## 3. Tools

### CoreWeave Intelligent CLI (`cwic`)

CoreWeave provides `cwic` for cluster-level operations beyond standard `kubectl`:

- `cwic auth login` — Authenticate to CoreWeave
- NodePool upgrades and rollback (`cwic rollback`)
- Object storage bucket management

See [CoreWeave CLI docs](https://docs.coreweave.com) for installation.

### kubectl

Standard Kubernetes operations. CoreWeave adds the `NodePool` CRD
(`compute.coreweave.com/v1alpha1`):

```bash
kubectl get nodepool                    # List pools (TARGET vs CURRENT)
kubectl describe nodepool <name>        # Check conditions (Valid, AtTarget)
kubectl get pods -n iris                # List Iris Pods
kubectl describe pod <name> -n iris     # Check scheduling / pull events
kubectl logs <pod> -n iris              # Read Pod logs
kubectl get nodes --show-labels         # Verify GPU node labels
```

### CoreWeave Observe (Managed Grafana)

Free, fully-managed Grafana included with every CKS cluster. Pre-configured
dashboards for CKS (control plane, Pods), Fleet (node/resource trends),
and Network (traffic, latency). No setup required.

## 4. Operator Setup Guide

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

### Step 2: Set S3 credentials (if using S3 storage)

```bash
export R2_ACCESS_KEY_ID=<your-r2-access-key-id>
export R2_SECRET_ACCESS_KEY=<your-r2-secret-access-key>
```

`iris cluster start` creates a K8s Secret (`iris-s3-credentials`) from these
environment variables automatically.

> **Note**: CoreWeave AI Object Storage (`cwobject.com`, `cwlota.com`) uses
> virtual-hosted-style S3 addressing, which is auto-detected and configured.
> However, this addressing style is incompatible with JAX's GCS/S3 backend.
> Use Cloudflare R2 or another path-style-compatible endpoint for JAX workloads.

### Step 3: Start the cluster

```bash
iris --config=lib/iris/examples/coreweave.yaml cluster start
```

This is fully idempotent. It creates/reconciles:
1. Namespace (`iris`) and RBAC (ServiceAccount, ClusterRole, ClusterRoleBinding)
2. S3 credentials Secret (if S3 storage URIs are configured)
3. ConfigMap (`iris-cluster-config`) with the cluster config as JSON
4. Shared NodePools (one per scale group, in parallel)
5. Controller Deployment (`iris-controller`) — images are built and pushed automatically
6. Controller Service (`iris-controller-svc`, ClusterIP)

### Step 4: Use the cluster

```bash
iris --config=lib/iris/examples/coreweave.yaml cluster status
iris --config=lib/iris/examples/coreweave.yaml cluster dashboard
```

### Step 5: Stop

```bash
iris --config=lib/iris/examples/coreweave.yaml cluster stop
```

Deletes worker Pods and controller resources. NodePools are left in place (they
scale to zero when idle).

## 5. RBAC Permissions

`iris cluster start` auto-applies these resources via `ensure_rbac()` (defined
in `CoreweavePlatform`):

| Resource | Purpose |
|----------|---------|
| `iris` Namespace | Isolation for all Iris resources |
| `iris-controller` ServiceAccount | In-cluster K8s API auth for controller and worker Pods |
| `iris-controller` ClusterRole | API permissions (see below) |
| `iris-controller` ClusterRoleBinding | Binds ServiceAccount to ClusterRole |

**ClusterRole permissions**:

| API Group | Resources | Verbs |
|-----------|-----------|-------|
| `compute.coreweave.com` | `nodepools` | get, list, watch, create, update, patch, delete |
| core (`""`) | `pods`, `pods/exec`, `pods/log` | get, list, watch, create, update, patch, delete |
| core (`""`) | `nodes` | get, list, watch |
| core (`""`) | `configmaps` | get, list, watch, create, update, patch, delete |

## 6. Configuration Reference

### CoreweavePlatformConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `region` | string | — | CoreWeave region (e.g. `US-WEST-04A`) |
| `namespace` | string | `iris` | Kubernetes namespace for all resources |
| `kubeconfig_path` | string | — | Only needed when running CLI outside the cluster |
| `object_storage_endpoint` | string | — | S3-compatible endpoint URL |

### CoreweaveControllerConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `port` | int | `10000` | Controller listening port |
| `service_name` | string | `iris-controller-svc` | K8s Service name |
| `scale_group` | string | **required** | Scale group to schedule the controller onto |

### CoreweaveSliceConfig (per scale group)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `region` | string | — | Scale group region |
| `instance_type` | string | — | CoreWeave instance type (e.g. `gd-8xh100ib-i128`) |
| `gpu_class` | string | — | GPU model (e.g. `H100`) |
| `infiniband` | bool | `false` | Request `rdma/ib: 1` resource on task Pods |

### Bootstrap config

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `docker_image` | string | — | Worker image |
| `worker_port` | int | — | Worker listening port |
| `cache_dir` | string | — | **Must point to NVMe** (see warning below) |
| `runtime` | string | — | Set to `kubernetes` for CoreWeave (enables Pod-per-task) |

> **Warning — Disk layout**: CoreWeave bare-metal nodes have a **15 GB RAM disk**
> as the root filesystem and multi-TB NVMe at `/mnt/local`. The `cache_dir` must
> point to NVMe (e.g. `/mnt/local/iris-cache`). Using the default root path will
> fill the RAM disk immediately and cause Pod eviction.

### Startup grace period

The default `startup_grace_period` is 2400s (40 minutes). This covers CoreWeave
bare-metal node provisioning (20-30 min) plus Pod image pull and startup time.

## 7. Instance Type Naming

CoreWeave instance types follow the pattern `{prefix}-{count}x{model}{networking}-i{cpu}`:

| Component | Meaning | Example |
|-----------|---------|---------|
| `gd` | GPU device | `gd-8xh100ib-i128` |
| `cd` | CPU device | `cd-gp-i64-erapids` |
| `8x` | GPU count | 8 GPUs |
| `h100` | GPU model | NVIDIA H100 |
| `ib` | InfiniBand | High-bandwidth interconnect |
| `i128` | vCPU count | 128 vCPUs |

**Known-good instance types**:

| Instance Type | GPUs | vCPUs | RAM | Use Case |
|---------------|------|-------|-----|----------|
| `gd-8xh100ib-i128` | 8x H100 | 128 | 2 TB | GPU training (primary) |
| `cd-gp-i64-erapids` | none | 64 | 256 GB | Controller / CPU tasks |

Full list: [CoreWeave GPU Instances](https://docs.coreweave.com/docs/platform/instances/gpu-instances)

## 8. Key Design Decisions

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
Pod uses in-cluster ServiceAccount auth for all kubectl operations and requests
dedicated `cpu: 2` and `memory: 4Gi` (with matching limits) so it runs with
Guaranteed QoS instead of BestEffort.

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
- Claim GPU/RDMA resources from the kubelet device plugin (`nvidia.com/gpu: N`,
  `rdma/ib: 1` when `infiniband: true`)
- Receive tolerations for `nvidia.com/gpu` NoSchedule taints automatically
- Use `hostNetwork: true` with `dnsPolicy: ClusterFirstWithHostNet`
- Get S3 credentials via `secretKeyRef` from the platform-managed Secret
- Use `emptyDir` for `/app` (workdir) so tasks can run on any node
- Materialize code bundles in-pod via fsspec
- Have `ownerReferences` pointing to the worker Pod for GC

The worker Pod intentionally does **not** request GPU/RDMA resources when
`runtime: kubernetes` is configured, so task Pods can claim them instead.

### Reconcile-driven recovery

Correctness does not depend on in-memory thread state. After a controller restart,
`list_all_slices()` discovers existing worker Pods by labels and reconstructs
slice handles with the correct state based on Pod phase and readiness conditions.

## 9. Early Failure Detection

The platform detects fatal errors before the full timeout expires:

| Error | Detection | Behavior |
|-------|-----------|----------|
| `ErrImagePull`, `ImagePullBackOff`, `InvalidImageName` | Container waiting reason | Immediate failure with error message |
| `CreateContainerConfigError` | Container waiting reason | Immediate failure (usually missing Secret/ConfigMap) |
| `CrashLoopBackOff` | Waiting reason + `restartCount >= 2` | Fail with last 30 lines of logs |
| `FailedMount`, `FailedAttachVolume` | Pod events, `count >= 3`, after 90s grace | Immediate failure |

## 10. Environment Variables

### Operator (outside cluster)

| Variable | Purpose |
|----------|---------|
| `KUBECONFIG` | Path to kubeconfig (alternative to `kubeconfig_path` in config) |
| `R2_ACCESS_KEY_ID` | S3/R2 access key (required if storage uses `s3://`) |
| `R2_SECRET_ACCESS_KEY` | S3/R2 secret key |

### Auto-injected into worker and task Pods

| Variable | Source | Description |
|----------|--------|-------------|
| `IRIS_WORKER_NODE_NAME` | Downward API (`spec.nodeName`) | Kubernetes node name |
| `IRIS_POD_NAMESPACE` | Downward API (`metadata.namespace`) | Pod's namespace |
| `IRIS_POD_NAME` | Downward API (`metadata.name`) | Pod's name |
| `IRIS_POD_UID` | Downward API (`metadata.uid`) | Pod's UID |
| `IRIS_SERVICE_ACCOUNT_NAME` | Platform | ServiceAccount for task Pods (set when `runtime: kubernetes`) |
| `IRIS_S3_SECRET_NAME` | Platform | K8s Secret name for S3 credentials |
| `AWS_ACCESS_KEY_ID` | Secret ref | From `iris-s3-credentials` Secret |
| `AWS_SECRET_ACCESS_KEY` | Secret ref | From `iris-s3-credentials` Secret |
| `AWS_ENDPOINT_URL` | Config | S3 endpoint URL |
| `FSSPEC_S3` | Platform | JSON-encoded fsspec S3 config (includes endpoint and addressing style) |

## 11. Timeouts

| Timeout | Default | Description |
|---------|---------|-------------|
| Pod readiness | 2400s (40 min) | Max wait for worker Pod to pass readiness probe |
| Deployment readiness | 2400s (40 min) | Max wait for controller Deployment availability |
| kubectl commands | 1800s (30 min) | Default subprocess timeout for kubectl calls |
| Mount failure grace | 90s | Grace period before treating FailedMount as fatal |

## 12. Control Flow

### Cluster startup (`iris cluster start`)

`CoreweavePlatform.start_controller()` orchestrates the full startup sequence.
See `lib/iris/src/iris/cluster/platform/coreweave.py`.

1. Apply RBAC prerequisites (Namespace, ServiceAccount, ClusterRole, ClusterRoleBinding)
2. Create S3 credentials Secret (if S3 storage configured)
3. Apply ConfigMap with cluster config
4. Create/reconcile all shared NodePools in parallel via `ensure_nodepools()`
5. Apply controller Deployment (with rollout restart)
6. Apply controller Service (ClusterIP)
7. Wait for Deployment availability (polls with early failure detection for
   image pull errors, crash loops, and volume mount failures)
8. Return controller address (K8s Service DNS)

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

## 13. Credentials Summary

### Platform-managed (all created by `iris cluster start`)

| Resource | Purpose | Created By |
|----------|---------|------------|
| `iris` Namespace + RBAC | K8s API auth and permissions | `start_controller()` via `ensure_rbac()` |
| `iris-s3-credentials` Secret | S3 object storage auth | `start_controller()`, from `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` env vars |
| `iris-cluster-config` ConfigMap | Cluster config for controller and workers | `start_controller()` |
| In-cluster ServiceAccount token | kubectl calls from controller Pod | Auto-mounted by Kubernetes |

### Operator-managed

| Resource | Purpose | How to Obtain |
|----------|---------|---------------|
| CoreWeave API token | kubeconfig auth | Console > Tokens > Create Token |
| Kubeconfig file | Operator's kubectl access | Console > Tokens > Download Kubeconfig |

The `kubeconfig_path` config field is only needed when running the CLI
**outside** the cluster (e.g., `iris cluster start` from a laptop). Inside the
cluster, Pods use in-cluster auth automatically.

## 14. Open Questions / Known Limitations

1. **Multi-node slices**: `num_vms > 1` is not supported and raises `ValueError`.
   InfiniBand co-scheduling for multi-node training needs investigation.

2. **NodePool rate limits**: Creating many NodePools at scale has not been
   validated with CoreWeave.

3. **Task Pod GC**: `ownerReferences` on task Pods only trigger GC when the
   worker Pod object is deleted. If the worker crash-loops in place, stale task
   Pods can accumulate. See TODO in `kubernetes.py`.

## 15. Troubleshooting

### NodePool not scaling up

```bash
kubectl get nodepool                     # Check TARGET vs CURRENT
kubectl describe nodepool <name>         # Check conditions: Valid, AtTarget
```

If `Valid` is `False`, the instance type or configuration is rejected.

### Pod stuck in Pending

```bash
kubectl describe pod <name> -n iris      # Check Events section
kubectl get events -n iris --sort-by='.lastTimestamp'
```

Common causes: node not yet provisioned (wait for autoscaler), resource limits
exceeded, or missing tolerations.

### Image pull errors

The platform detects `ErrImagePull` / `ImagePullBackOff` and fails immediately.
Verify the image exists and is public:

```bash
docker pull ghcr.io/marin-community/iris-worker:latest
```

### CrashLoopBackOff

The platform detects crash loops after 2+ restarts and reports the last 30 log
lines. To inspect manually:

```bash
kubectl logs <pod> -n iris --previous    # Logs from the last crash
```

### Disk full / Pod eviction

If `cache_dir` is not set to `/mnt/local/...`, the 15 GB root RAM disk fills
instantly. Fix in config and redeploy.

## 16. References

- [CoreWeave CKS Introduction](https://docs.coreweave.com/docs/products/cks)
- [CKS Cluster Creation](https://docs.coreweave.com/docs/products/cks/clusters/create)
- [API Access Tokens and Kubeconfig](https://docs.coreweave.com/docs/products/cks/auth-access/manage-api-access-tokens)
- [CoreWeave Node Pools](https://docs.coreweave.com/docs/products/cks/nodes/nodes-and-node-pools)
- [CoreWeave Autoscaling](https://docs.coreweave.com/docs/products/cks/nodes/autoscaling)
- [CoreWeave GPU Instances](https://docs.coreweave.com/docs/platform/instances/gpu-instances)
- [CoreWeave Observe (Managed Grafana)](https://docs.coreweave.com/docs/observability/managed-grafana)
- [CoreWeave Terraform Provider](https://docs.coreweave.com/docs/products/cks/terraform/about)

### Source files

| File | Description |
|------|-------------|
| `lib/iris/src/iris/cluster/platform/coreweave.py` | CoreWeave platform implementation (includes `ensure_rbac()`) |
| `lib/iris/src/iris/cluster/runtime/kubernetes.py` | KubernetesRuntime (Pod-per-task) |
| `lib/iris/src/iris/cluster/k8s/kubectl.py` | Kubectl CLI wrapper |
| `lib/iris/examples/coreweave.yaml` | Example cluster config |
| `lib/iris/AGENTS.md` | CoreWeave integration notes for agents |
