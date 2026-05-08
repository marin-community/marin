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
- Controller extras in the local Iris venv:
  `uv pip install 'marin-iris[controller]'`

This document is the canonical runbook for day-to-day CoreWeave operations.

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
iris --cluster=coreweave cluster start
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
iris --cluster=coreweave cluster status
iris --cluster=coreweave cluster dashboard
```

### Step 5: Stop

```bash
iris --cluster=coreweave cluster stop
```

Deletes worker Pods and controller resources. NodePools are left in place (they
scale to zero when idle).

### Connecting

Preferred: use `--cluster=NAME` so Iris opens and closes the controller tunnel:

```bash
iris --cluster=coreweave-ci job logs /runner/my-job
iris cluster list
```

`--cluster=NAME` resolves to a config under `lib/iris/examples/` and opens a
`kubectl port-forward` to the controller service. This path requires the
`iris[controller]` extras (`duckdb`, `pyarrow`, `kubernetes`). Without them,
auto-tunneled CoreWeave commands fail before connecting:
`ImportError: Install iris[controller] to use CloudK8sService`.

Fallback: manual port-forward if you need a long-lived tunnel:

```bash
kubectl --kubeconfig ~/.kube/coreweave-iris \
  port-forward -n <namespace> svc/<service_name> 10000:10000 &
iris --controller-url=http://localhost:10000 ...
```

| Cluster name | Namespace | Service | Config file |
|--------------|-----------|---------|-------------|
| `coreweave` | `iris` | `iris-controller-svc` | `coreweave.yaml` |
| `coreweave-ci` | `iris-ci` | `iris-ci-controller-svc` | `coreweave-ci.yaml` |

### GPU Configs

| Target | Iris config | `--gpu` request | `nvidia-smi` GPU name |
|--------|-------------|-----------------|-----------------------|
| H100 | `lib/iris/examples/coreweave-ci.yaml` | `H100x1` | `NVIDIA H100 80GB HBM3` |
| GH200 | `lib/iris/examples/coreweave-rno2a.yaml` | `GH200x1` | `NVIDIA GH200 480GB` |
| B200 | `lib/iris/examples/coreweave-usw09b.yaml` | `B200x1` | `NVIDIA B200` |

Use `GH200x1` for RNO2A. `H200x1` also schedules there today; both land on
CoreWeave `gd-1xgh200` nodes labeled `gpu.nvidia.com/model=GH200_480GB` and
report `NVIDIA GH200 480GB`.

Before the full GPU canary, run one tiny direct JAX job for each row. It should
prove `nvidia-smi`, GPU-backed JAX, and a tiny matmul.

Marin's `gpu` extra installs the JAX CUDA 13 wheel stack from PyPI. CoreWeave
GPU nodes must expose NVIDIA driver 580 or newer; `nvidia-smi` should report
CUDA 13.x.

### Grug MoE Canary Warm-Node Multinode Smoke

For a realistic Grug MoE multinode smoke, use the GPU path in
`experiments.ferries.canary_ferry`. This is a temporary warm-node validation
while cold-start/gang scheduling is tracked in #5480: before submitting the job,
verify that the required GPU nodes are already `CURRENT`, free, and schedulable.

Warm-node preflight:

```bash
# Confirm the target pool is not already occupied by another Iris workload.
uv run iris --cluster=<cluster> job list --state running

# Confirm the requested nodes are already warm, not still provisioning.
kubectl --kubeconfig <kubeconfig> get nodepool.compute.coreweave.com <nodepool> \
  -o custom-columns=NAME:.metadata.name,TARGET:.spec.targetNodes,CURRENT:.status.currentNodes,INPROGRESS:.status.inProgress,QUEUED:.status.queuedNodes

# Confirm the scheduler-visible pods are not already consuming the target nodes.
kubectl --kubeconfig <kubeconfig> -n <namespace> get pods -o wide
```

Starting smoke settings:

| Target | Cluster | Namespace | Kubeconfig | NodePool | `CANARY_GPU_*` | Batch | `NCCL_SOCKET_IFNAME` |
|--------|---------|-----------|------------|----------|-----------------|-------|----------------------|
| H100x8 x 2 | `coreweave-ci` | `iris-ci` | `~/.kube/coreweave-iris` | `iris-ci-h100-8x` | `TYPE=H100`, `COUNT=8`, `REPLICAS=2` | 64 | `=enp157s0np0` |
| B200x8 x 2 | `coreweave-usw09b` | `iris` | `~/.kube/cw-usw09b.yaml` | `iris-usw09b-b200-8x` | `TYPE=B200`, `COUNT=8`, `REPLICAS=2` | 128 | `=enp44s0np0` |
| GH200x1 x 2 | `coreweave-rno2a` | `iris` | `~/.kube/cw-rno2a.yaml` | `iris-rno2a-gh200-1x` | `TYPE=GH200`, `COUNT=1`, `REPLICAS=2` | 16 | `=eth0` |

For H100, manually warm the second node before launch and restore the pool after
the run:

```bash
kubectl --kubeconfig ~/.kube/coreweave-iris patch nodepool.compute.coreweave.com iris-ci-h100-8x \
  --type merge -p '{"spec":{"targetNodes":2}}'

# After the smoke:
kubectl --kubeconfig ~/.kube/coreweave-iris patch nodepool.compute.coreweave.com iris-ci-h100-8x \
  --type merge -p '{"spec":{"targetNodes":1}}'
```

The GitHub CoreWeave canary workflow uses the same H100 pool; run it and manual
H100 validation sequentially. If the controller restarts after warming the H100
pool, recheck `targetNodes`; startup may reconcile the pool back toward the
single-node target.

Submit with explicit `-e` environment variables. Iris job containers do not
inherit arbitrary shell variables from the submitter. Because this canary uses
real SlimPajama data, use a shared durable `MARIN_PREFIX` plus the credentials
needed to read/write that prefix. `CANARY_TRACKER=json_logger` avoids requiring
W&B for this smoke.

| Use | Prefix | Endpoint | Credentials |
|-----|--------|----------|-------------|
| H100 CI state and canary data | `s3://marin-na/...` | Cloudflare R2 | R2 credentials |
| B200/GH200 controller state | `s3://marin-poc/iris/state/...` | `https://cwobject.com` | CoreWeave object storage credentials |
| B200/GH200 canary data with `MARIN_PREFIX=s3://marin-na/marin/` | `s3://marin-na/marin/` | Cloudflare R2 | R2 credentials |

Set `AWS_ENDPOINT_URL` to the endpoint that matches the prefix and credentials.
For R2/CoreWeave S3-compatible endpoints, leave `AWS_REGION` and
`AWS_DEFAULT_REGION` as `auto`; use a real AWS region only for AWS S3.

```bash
RUN_ID="cw-grug-mn-warm-<target>-$(date -u +%Y%m%d-%H%M%S)"
LOG="/tmp/marin-cw-grug-moe/${RUN_ID}.log"
mkdir -p "$(dirname "$LOG")"

# TODO(#5524): remove CANARY_PROFILER_NUM_STEPS once Levanter profiler stop is
# idempotent. The shorter window keeps 20-step smokes from stopping the
# profiler on the final forced callback.
uv run iris --cluster=<cluster> job run \
  --job-name "$RUN_ID" \
  --cpu 1 --memory 2GB --disk 8GB --extra cpu \
  -e MARIN_PREFIX <shared-marin-prefix> \
  -e RUN_ID "$RUN_ID" \
  -e CANARY_ACCELERATOR gpu \
  -e CANARY_GPU_TYPE <H100|B200|GH200> \
  -e CANARY_GPU_COUNT <8|1> \
  -e CANARY_GPU_REPLICAS 2 \
  -e CANARY_STEPS 20 \
  -e CANARY_BATCH_SIZE <64|128|16> \
  -e CANARY_PROFILER_ENABLED true \
  -e CANARY_PROFILER_NUM_STEPS 10 \
  -e CANARY_TRACKER json_logger \
  -e NCCL_SOCKET_IFNAME '<interface>' \
  -e HF_TOKEN "$HF_TOKEN" \
  -e AWS_ACCESS_KEY_ID "$AWS_ACCESS_KEY_ID" \
  -e AWS_SECRET_ACCESS_KEY "$AWS_SECRET_ACCESS_KEY" \
  -e AWS_ENDPOINT_URL "$AWS_ENDPOINT_URL" \
  -e AWS_REGION "${AWS_REGION:-auto}" \
  -e AWS_DEFAULT_REGION "${AWS_DEFAULT_REGION:-auto}" \
  -- python -m experiments.ferries.canary_ferry 2>&1 | tee "$LOG"
```

Expected success signals: both replicas report JAX 0.10.0, both enter
`initialize_jax` with `IRIS_NUM_TASKS=2`, both emit tracker summaries, the
profiler starts and records a JAX profile artifact, the step reaches 20/20, a
checkpoint is committed, and the parent job exits `JOB_STATE_SUCCEEDED`. H100
batch 128 has OOMed with the current Grug MoE model; use batch 64 for functional
validation.

### KubernetesProvider Operations

On CoreWeave, there are no persistent worker daemons. The controller dispatches
tasks directly as Kubernetes Pods, `list-workers` returns empty, and the
`workers` SQL table is empty. Use:

```bash
kci get pods -n iris -l iris.managed=true
kci get nodepools
kci get events -n iris --sort-by=.lastTimestamp | tail -30
kci logs -n iris deployment/iris-controller -f
iris rpc controller get-kubernetes-cluster-status
```

(`kci` = `kubectl --kubeconfig ~/.kube/coreweave-iris`)

### NodePool Operations

```bash
kci get nodepools
kci patch nodepool <name> --type=merge -p '{"spec":{"targetNodes":N}}'
kci delete nodepool <name>
```

Do not use `kubectl scale --replicas` for NodePools; patch
`spec.targetNodes`.

If deletion is stuck because the autoscaler fights deletion or the node is
mid-delivery:

```bash
kci scale deployment iris-controller -n iris --replicas=0
kci patch nodepool <name> --type=merge -p '{"spec":{"autoscaling":false,"targetNodes":0}}'
kci patch nodepool <name> --type=json -p '[{"op":"remove","path":"/metadata/finalizers"}]'
kci delete nodepool <name>
```

`iris cluster stop` deletes pods but NodePools survive. Delete managed NodePools
explicitly to avoid lingering GPU costs:

```bash
iris cluster stop
kci delete nodepool -l iris-<label_prefix>-managed=true
```

### Gotchas

- **NodePools survive `cluster stop`.** Delete explicitly to avoid lingering GPU costs.
- **`list-workers` returns empty.** KubernetesProvider dispatches pods directly.
- **`list-tasks` requires `job_id`.** Calling without it throws `ConnectError: job_id is required`.
- **`cluster start` always rebuilds+pushes images.** Needs `docker login ghcr.io` with `write:packages` PAT.
- **Konnectivity agent.** `kubectl port-forward` returns 500 until `konnectivity-agent` pods are running (~18-30s after node provisions).
- **H100 quota is account-wide.** If a canary pod is stuck with `NotTriggerScaleUp: 2 max node group size reached`, check `kci get nodepools -A`; another H100 pool can consume the shared US-WEST-04A cap.

Cold-start timings:

| Resource | Time |
|----------|------|
| CW CPU node | ~14 min |
| CW H100 bare-metal | ~20 min |
| CW first training step (from zero) | ~25-30 min |

## 5. RBAC Permissions

`iris cluster start` auto-applies these resources via `ensure_rbac()` (defined
in `CoreweavePlatform`):

| Resource | Purpose |
|----------|---------|
| `iris` Namespace | Isolation for all Iris resources |
| `iris-controller` ServiceAccount | In-cluster K8s API auth for controller and worker Pods |
| `iris-controller-{namespace}` ClusterRole | API permissions (see below). Namespace-qualified to support multiple Iris instances on the same CKS cluster. |
| `iris-controller-{namespace}` ClusterRoleBinding | Binds ServiceAccount to ClusterRole. Namespace-qualified to avoid collisions. |

**ClusterRole permissions**:

| API Group | Resources | Verbs |
|-----------|-----------|-------|
| `compute.coreweave.com` | `nodepools` | get, list, watch, create, update, patch, delete |
| core (`""`) | `pods`, `pods/exec`, `pods/log` | get, list, watch, create, update, patch, delete |
| core (`""`) | `nodes` | get, list, watch |
| core (`""`) | `configmaps` | get |

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
See `lib/iris/src/iris/providers/k8s/coreweave.py`.

1. Apply RBAC prerequisites (Namespace, ServiceAccount, ClusterRole `iris-controller-{ns}`, ClusterRoleBinding `iris-controller-{ns}`)
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

## 13. Multi-VM Jobs

Multi-VM scale groups allow training across multiple nodes. Each slice in a
multi-VM group provisions N worker Pods (one per VM) that share a single
ConfigMap. All Pods in a slice must reach Ready before the slice is usable.

### Configuration

Define a scale group with `num_vms > 1` in the cluster config. The
`slice_template.num_vms` must match the top-level `num_vms`. For CoreWeave GPU
groups, define at least one topology label in `worker.attributes`; use
`same-slice` to discover the leader pod's node label value and pin follower
pods to that same topology domain:

```yaml
scale_groups:
  h100-16x:
    num_vms: 2
    resources:
      cpu: 128
      ram: 2048GB
      disk: 1TB
      device_type: gpu
      device_variant: H100
      device_count: 8
    worker:
      attributes:
        region: US-WEST-04A
        pool: h100-16x
        backend.coreweave.cloud/superpod: same-slice
    buffer_slices: 0
    max_slices: 1
    priority: 50
    slice_template:
      num_vms: 2
      coreweave:
        region: US-WEST-04A
        instance_type: gd-8xh100ib-i128
```

### Submitting multi-replica jobs

Jobs targeting a multi-VM CoreWeave GPU group should use coscheduling so all
replicas are launched together. Include `ports=["jax"]` so Iris allocates a
named port for JAX coordinator discovery:

```python
from iris.sdk import IrisClient, CoschedulingConfig

client = IrisClient()
client.submit(
    name="multi-node-training",
    image="ghcr.io/marin-community/iris-task:latest",
    command=["python", "train.py"],
    replicas=2,
    ports=["jax"],
    coscheduling=CoschedulingConfig(group_by="pool"),
    resources={"gpu": 8},
)
```

Each replica receives `IRIS_TASK_ID` (0 or 1), `IRIS_NUM_TASKS` (2), and
`IRIS_PORT_JAX` (the allocated coordinator port). Task code calls
`iris.runtime.jax_init.initialize_jax()` to bootstrap JAX distributed — task 0
registers its coordinator address via the endpoint API, and task 1 discovers it
by polling.

### Requirements

- **Coscheduling is mandatory for multi-host GPU groups**: replicas must
  launch together on workers from the same CoreWeave pool.
- **Topology labels are mandatory for multi-host GPU groups**: set at least one
  CoreWeave topology key in `worker.attributes`, such as
  `backend.coreweave.cloud/superpod: same-slice`.
- **hostNetwork anti-affinity**: Because worker Pods use `hostNetwork: true`,
  two Pods binding the same port cannot schedule on the same node. This
  provides implicit anti-affinity — no explicit `podAntiAffinity` rule needed.
- **Gang semantics**: If any task in a coscheduled group fails terminally, all
  siblings are killed and the entire group retries together.

## 14. Credentials Summary

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

## 15. Open Questions / Known Limitations

1. **NodePool rate limits**: Creating many NodePools at scale has not been
   validated with CoreWeave.

2. **Task Pod GC**: `ownerReferences` on task Pods only trigger GC when the
   worker Pod object is deleted. If the worker crash-loops in place, stale task
   Pods can accumulate. See TODO in `kubernetes.py`.

## 16. Troubleshooting

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

## 17. References

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
| `lib/iris/src/iris/providers/k8s/coreweave.py` | CoreWeave platform implementation (includes `ensure_rbac()`) |
| `lib/iris/src/iris/cluster/runtime/kubernetes.py` | KubernetesRuntime (Pod-per-task) |
| `lib/iris/src/iris/providers/k8s/service.py` | Kubectl CLI wrapper |
| `lib/iris/examples/coreweave.yaml` | Example cluster config |
| `lib/iris/AGENTS.md` | CoreWeave integration notes for agents |
