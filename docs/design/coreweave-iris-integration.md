# Research Report: CoreWeave Platform Integration for Iris

**Issue**: [#2822 — Iris: Implement CoreWeave platform](https://github.com/marin-community/marin/issues/2822)

## Executive Summary

CoreWeave is entirely Kubernetes-native and runs on bare metal (no hypervisor). All compute provisioning happens through **Kubernetes Node Pool Custom Resources** (`compute.coreweave.com/v1alpha1`). CoreWeave uses the upstream **Kubernetes Cluster Autoscaler** with a custom cloud provider. The Iris codebase already has a `CoreweavePlatform` stub and proto definitions — the integration requires implementing the `Platform` protocol methods by mapping Iris scaling groups to CoreWeave Node Pools and using `kubectl`/Kubernetes API calls where GCP currently uses `gcloud`.

---

## 1. CoreWeave Architecture (Key Findings)

**Bare metal, not VMs**: CoreWeave runs Kubernetes directly on bare metal GPU nodes (CKS = CoreWeave Kubernetes Service). There is no hypervisor — nodes are stateless, rebooted from clean images, and isolated via NVIDIA BlueField DPUs. This means there is no "create a VM" API equivalent; instead, you provision **Node Pools** and schedule **Pods**.

**Node Pools as the provisioning primitive**: Node Pools are Kubernetes CRDs:

```yaml
apiVersion: compute.coreweave.com/v1alpha1
kind: NodePool
metadata:
  name: iris-h100-pool
spec:
  computeClass: default
  instanceType: gd-8xh100ib-i128  # immutable once set
  targetNodes: 2
  minNodes: 0
  maxNodes: 10
  autoscaling: true
  lifecycle:
    scaleDownStrategy: PreferIdle
  nodeLabels:
    iris-scale-group: "h100_8x"
  nodeTaints:
    - key: iris.marin.community/dedicated
      value: "true"
      effect: NoSchedule
```

Key constraints:
- `instanceType` is immutable after creation (one Node Pool per GPU config)
- Scale-up takes **20-30 minutes** (bare metal boot, not VM spin-up)
- Quota must cover `maxNodes`; if `targetNodes` > quota, creation fails entirely

**Autoscaling**: CoreWeave uses the upstream Kubernetes Cluster Autoscaler (enabled by default on CKS 1.32+). Scale-up is triggered when Pods can't be scheduled. Scale-down removes idle nodes based on `scaleDownStrategy` (`IdleOnly` or `PreferIdle`). Scale-to-zero is supported but requires a secondary node pool for system services.

**GPU selection via Kubernetes labels**:

| Label | Example |
|---|---|
| `gpu.nvidia.com/class` | `H100`, `A100`, `L40S` |
| `gpu.nvidia.com/model` | `H100`, `GH200_480GB` |
| `node.kubernetes.io/instance-type` | `gd-8xh100ib-i128` |
| `topology.kubernetes.io/region` | `LGA1`, `ORD1` |

Workloads use `nodeSelector` or `nodeAffinity` + `nvidia.com/gpu` resource requests. InfiniBand is requested via `rdma/ib: 1`.

**Available GPU instance types**: GB300 NVL72, GB200 NVL72, B200 (IB), H200 (IB), H100 (IB), RTX Pro 6000, L40S, L40, GH200, A100 — all 8-GPU (or 4-GPU for Grace-based) configurations.

---

## 2. Iris Architecture (Current State)

**Platform abstraction** (`lib/iris/src/iris/cluster/platform/base.py`): Defines `Platform`, `SliceHandle`, `VmHandle`, `StandaloneVmHandle` protocols. GCP implements these by shelling out to `gcloud` CLI.

**Existing CoreWeave stub** (`lib/iris/src/iris/cluster/platform/coreweave.py`): All 8 Platform methods raise `NotImplementedError`. The factory (`factory.py`) already dispatches to it.

**Proto definitions** already exist:
- `CoreweavePlatformConfig` with `region` field (`config.proto:37-39`)
- `CoreweaveSliceConfig` with `region` and `instance_type` fields (`config.proto:100-103`)

**Scaling flow**: Autoscaler routes demand -> `ScalingGroup.scale_up()` -> `Platform.create_slice(config)` -> returns `SliceHandle` -> `bootstrap_slice_vms()` discovers VMs via `handle.describe()` and runs bootstrap script on each.

**Bootstrap** (`bootstrap.py`): Installs Docker, pulls images from GCP Artifact Registry (`gcloud auth configure-docker`), starts worker container with `--network=host`, mounts Docker socket + cache dir. Currently assumes `gcloud` is available on worker nodes.

---

## 3. Mapping Iris Concepts to CoreWeave Kubernetes

| Iris Concept | GCP Implementation | CoreWeave Equivalent |
|---|---|---|
| **ScalingGroup** | Config + scaling policy | **NodePool CRD** (1:1 mapping) |
| **Slice** | TPU pod (atomic multi-VM unit) | **Set of Pods** on dedicated nodes (or a single multi-GPU node) |
| **SliceHandle** | Wraps TPU pod name + zone | Wraps NodePool name + scheduled Pod references |
| **VmHandle** | SSH to TPU worker VM | **`kubectl exec` into Pod** (or SSH to bare metal node) |
| **create_slice()** | `gcloud compute tpus tpu-vm create` | Create Pod(s) with `nodeSelector` (autoscaler provisions nodes) OR `kubectl patch nodepool` to increment `targetNodes` |
| **list_slices()** | `gcloud compute tpus list` | `kubectl get pods -l iris-scale-group=X` |
| **terminate()** | `gcloud compute tpus delete` | Delete Pod(s); autoscaler handles node scale-down |
| **describe()** | `gcloud compute tpus describe` | `kubectl get pods` + node IP lookup |
| **discover_controller()** | Label-based VM lookup | Kubernetes Service / static config / DNS |
| **Bootstrap** | SSH + Docker on VM | Pod spec with container image (no SSH needed) |
| **QuotaExhaustedError** | gcloud error parsing | NodePool status conditions (quota/capacity) |
| **Labels for discovery** | GCE labels (`gcloud ... --labels`) | Kubernetes labels on Pods |
| **tunnel()** | `gcloud compute ssh -L` | `kubectl port-forward` |

---

## 4. Proposed Integration Design

### Option A: Pod-Based Model (Recommended)

Instead of creating VMs and SSHing into them, deploy Iris workers as **Kubernetes Pods** on CoreWeave nodes. This is the idiomatic Kubernetes approach and avoids fighting the platform.

**ScalingGroup -> NodePool mapping**: Each Iris `ScaleGroupConfig` with `slice_template.coreweave` maps 1:1 to a CoreWeave NodePool CRD. The `instance_type` field in `CoreweaveSliceConfig` maps to the NodePool's `instanceType`.

**Scaling flow**:
1. `create_slice()`: Deploy a Pod (or set of Pods for multi-node training) with appropriate `nodeSelector` for the target GPU type. If autoscaling is enabled on the NodePool, the Cluster Autoscaler handles node provisioning automatically. If not, `kubectl patch nodepool` to increment `targetNodes`.
2. `describe()`: Query Pod status + node assignments to discover worker addresses.
3. `terminate()`: Delete the Pod(s). Cluster Autoscaler scales down the idle node after the configured delay.

**Worker deployment**: Instead of SSH + Docker bootstrap, define a Pod spec:
```yaml
spec:
  nodeSelector:
    node.kubernetes.io/instance-type: gd-8xh100ib-i128
  containers:
    - name: iris-worker
      image: <worker-image>
      resources:
        limits:
          nvidia.com/gpu: 8
          rdma/ib: 1
      ports:
        - containerPort: 10001
      volumeMounts:
        - name: cache
          mountPath: /var/cache/iris
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
```

**Pros**: No SSH management, native Kubernetes health checks, Pod restart policies handle failures, ConfigMaps for config distribution, container image registry auth via Kubernetes secrets.

**Cons**: Significant departure from the SSH+Docker bootstrap model used by GCP and Manual platforms. The `VmHandle` protocol (SSH-based `run_command()`, `bootstrap()`) doesn't map cleanly to Pods.

### Option B: Kubernetes-Managed Nodes with SSH (Closer to Current Architecture)

Keep the existing SSH + Docker bootstrap model but use CoreWeave Node Pools as the provisioning layer.

**Scaling flow**:
1. `create_slice()`: Create or patch a NodePool to add nodes. Poll `kubectl get nodes` until new nodes appear.
2. Worker bootstrap: SSH into the bare metal nodes and run the existing Docker-based bootstrap script (requires SSH keys deployed to nodes).
3. `describe()`: Query node IPs from `kubectl get nodes -l node-pool=X`.
4. `terminate()`: Patch NodePool to reduce `targetNodes`, wait for node drain.

**Pros**: Minimal changes to existing Iris architecture. Reuses `DirectSshConnection`, `WorkerBootstrap`, and the Docker bootstrap script.

**Cons**: SSH to Kubernetes nodes is non-idiomatic. Requires ensuring nodes have SSH keys and Docker. The 20-30 minute bare metal boot time means the existing requesting timeout (120s) and boot timeout (300s) are insufficient.

### Option C: Hybrid -- Autoscaler-Driven with Pod Workers

Use CoreWeave's Kubernetes Cluster Autoscaler as the scaling mechanism, with Iris creating Pods that trigger autoscaling.

**Scaling flow**:
1. Pre-create NodePool CRDs (one per scale group) with `autoscaling: true`, `minNodes: 0`, `maxNodes: N`.
2. `create_slice()`: Create a Pod with `nodeSelector` matching the desired GPU type. The Cluster Autoscaler scales up the NodePool if no existing node can satisfy the Pod.
3. `describe()`: Query Pod status to determine when node is provisioned and Pod is running.
4. `terminate()`: Delete the Pod. Cluster Autoscaler scales down the idle node after the configured delay.

**Pros**: Leverages CoreWeave's autoscaler (they contributed it to upstream Kubernetes). No manual NodePool patching. Aligns with how CoreWeave intends their platform to be used.

**Cons**: Iris loses fine-grained control over scaling decisions (double autoscaler problem). The CoreWeave autoscaler may conflict with Iris's autoscaler logic.

### Recommendation: Option A with Pod-Based Workers

Option A is the cleanest integration. The key insight is that on CoreWeave, the "slice" concept maps to a **Pod or set of Pods** rather than a set of VMs. The implementation would:

1. **Use the Kubernetes Python client** (`kubernetes` package) instead of `gcloud` subprocess calls
2. **Pre-provision NodePools** as infrastructure (via Terraform or `kubectl apply` at cluster setup time), similar to how GCP zones are configured upfront
3. **Deploy workers as Pods** with `nodeSelector` targeting the right instance type
4. **Use Pod IP addresses** as worker addresses (host networking on CoreWeave)
5. **Use Kubernetes labels** for discovery (replace GCE labels)
6. **Use `kubectl port-forward`** for tunneling
7. **Controller discovery** via Kubernetes Service or static config

---

## 5. Required Changes

### Proto Changes (`config.proto`)

Extend `CoreweavePlatformConfig`:
```protobuf
message CoreweavePlatformConfig {
  string region = 1;
  string kubeconfig_path = 2;       // path to kubeconfig for CKS cluster
  string namespace = 3;              // Kubernetes namespace for Iris resources
  string container_registry = 4;    // registry for worker images
}
```

Extend `CoreweaveSliceConfig`:
```protobuf
message CoreweaveSliceConfig {
  string region = 1;
  string instance_type = 2;          // e.g. "gd-8xh100ib-i128"
  int32 gpus_per_node = 3;           // e.g. 8
  string gpu_class = 4;              // e.g. "H100" (for nodeSelector)
  bool infiniband = 5;               // request rdma/ib
}
```

Add `CoreweaveControllerConfig`:
```protobuf
message CoreweaveControllerConfig {
  int32 port = 1;
  string service_name = 2;          // Kubernetes Service for controller discovery
}
```

### Platform Implementation (`coreweave.py`)

Implement using the `kubernetes` Python client:

- `create_slice()`: Create Pod(s) with appropriate `nodeSelector`, resource requests, and Iris labels
- `list_slices()`: `kubectl get pods -l iris-label-prefix-scale-group=X`
- `list_all_slices()`: `kubectl get pods -l iris-label-prefix-managed=true`
- `describe()` on SliceHandle: Query Pod status + node IP
- `terminate()` on SliceHandle: Delete Pod(s)
- `create_vm()`: Create a Pod for the controller
- `discover_controller()`: Query Kubernetes Service or labeled Pod
- `tunnel()`: `kubectl port-forward`
- `shutdown()`: No-op (clean up Kubernetes client)

### Bootstrap Changes

For Option A, the bootstrap script becomes a Pod spec rather than an SSH script. The `WorkerBootstrap` class would need a CoreWeave-specific path that creates a Pod instead of SSHing + running a bash script. Alternatively, the bootstrap could be skipped entirely since the Pod spec already defines the container image, env vars, and volume mounts.

### Config Changes

New CoreWeave scale group config example:
```yaml
platform:
  coreweave:
    region: LGA1
    kubeconfig_path: ~/.kube/coreweave-config
    namespace: iris

scale_groups:
  h100_8x:
    accelerator_type: gpu
    accelerator_variant: H100
    num_vms: 1
    resources:
      cpu: 128
      ram: 2048GB
      gpu_count: 8
    min_slices: 0
    max_slices: 10
    slice_template:
      preemptible: false
      coreweave:
        region: LGA1
        instance_type: gd-8xh100ib-i128
        gpus_per_node: 8
        gpu_class: H100
        infiniband: true
```

### Timeout Adjustments

CoreWeave bare metal boot times are significantly longer than GCP VM creation:
- Current `boot_timeout`: 300s (5 min) -- insufficient for bare metal
- Current `requesting_timeout`: 120s -- insufficient
- **Recommended**: 1800s (30 min) for both, configurable per scale group or per platform

### Data Access

Current bootstrap configures `gcloud auth configure-docker` for GCP Artifact Registry. CoreWeave would need:
- A Kubernetes `imagePullSecret` for the container registry
- Object storage via CoreWeave's S3-compatible API (fsspec `s3fs` should work with `bundle_prefix = s3://...`)

### Dependencies

- `kubernetes` Python package (official Kubernetes client)
- CoreWeave Terraform provider (`coreweave/coreweave` v0.3.0) for initial CKS cluster + VPC provisioning (out of scope for the Iris platform code, but needed for infrastructure setup)

---

## 6. Implementation Order

1. **Extend protos**: Add fields to `CoreweavePlatformConfig` and `CoreweaveSliceConfig`
2. **Implement CoreWeave SliceHandle and VmHandle**: Pod-based handles using `kubernetes` client
3. **Implement CoreweavePlatform**: `create_slice`, `list_slices`, `list_all_slices`, `terminate`, `discover_controller`, `tunnel`
4. **Adapt bootstrap**: Create Pod specs instead of SSH scripts (or add a Kubernetes-native bootstrap path)
5. **Add CoreWeave controller config**: Controller as a Kubernetes Deployment + Service
6. **Config validation**: Add CoreWeave-specific validation in `config.py`
7. **Testing**: Unit tests with mocked Kubernetes client, integration tests against a real CKS cluster

---

## 7. Open Questions

1. **Autoscaler interaction**: Should Iris manage NodePool `targetNodes` directly, or rely on CoreWeave's Cluster Autoscaler? The latter is simpler but means Iris's autoscaler and CoreWeave's autoscaler could conflict. A clean approach: Iris creates Pods (expressing demand), CoreWeave's autoscaler provisions nodes. Iris only deletes Pods, and CoreWeave handles node scale-down.

2. **Multi-node training slices**: For distributed training across multiple 8-GPU nodes (e.g., 16-GPU job on 2 nodes), the `num_vms` field in `ScaleGroupConfig` maps to multiple Pods that need InfiniBand connectivity. This requires all Pods to land on nodes in the same InfiniBand fabric (same region, same instance type). CoreWeave handles this via the `rdma/ib` resource and proper node selection.

3. **SSH vs. Pod exec**: The current `VmHandle` protocol is SSH-centric (`run_command`, `bootstrap`, `wait_for_connection`). For a Pod-based model, these could be implemented via `kubectl exec` instead of SSH. The `kubernetes` Python client supports exec via websocket.

4. **Controller deployment**: Should the controller run as a Kubernetes Deployment (with a Service for discovery), or as a separate standalone process? Kubernetes Deployment is more idiomatic and gives automatic restarts.

5. **Image registry**: CoreWeave has its own container registry, or you can use any external registry (Docker Hub, GitHub Container Registry, etc.) with `imagePullSecrets`. Need to decide where Iris images are published for CoreWeave deployments.

---

## References

- [CoreWeave CKS Introduction](https://docs.coreweave.com/docs/products/cks)
- [CoreWeave Node Pools](https://docs.coreweave.com/docs/products/cks/nodes/nodes-and-node-pools)
- [CoreWeave Node Pool Reference](https://docs.coreweave.com/docs/products/cks/reference/node-pool)
- [CoreWeave Autoscaling](https://docs.coreweave.com/docs/products/cks/nodes/autoscaling)
- [CoreWeave GPU Instances](https://docs.coreweave.com/docs/products/instances/gpu-instances)
- [CoreWeave Workload Scheduling](https://docs.coreweave.com/docs/products/cks/clusters/scheduling/workload-scheduling)
- [CoreWeave Terraform Provider](https://docs.coreweave.com/docs/products/cks/terraform/about)
- [Kubernetes Cluster Autoscaler - CoreWeave cloud provider](https://github.com/kubernetes/autoscaler)
