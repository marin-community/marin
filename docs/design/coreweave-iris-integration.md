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
| **ScalingGroup** | Config + scaling policy | Config + scaling policy (same, no CW autoscaler) |
| **Slice** | TPU pod (atomic multi-VM unit) | **One NodePool** (`targetNodes: 1`) + worker **Pod** on that node |
| **SliceHandle** | Wraps TPU pod name + zone | Wraps NodePool name + worker Pod reference |
| **VmHandle** | SSH to TPU worker VM (`GcloudSshConnection`) | **`kubectl exec` into Pod** (`KubectlExecConnection`) |
| **create_slice()** | `gcloud compute tpus tpu-vm create` | `kubectl apply` NodePool CRD + deploy worker Pod |
| **list_slices()** | `gcloud compute tpus list` | `kubectl get nodepools -l iris-scale-group=X` |
| **terminate()** | `gcloud compute tpus delete` | `kubectl delete nodepool <slice-id>` (deterministic) |
| **describe()** | `gcloud compute tpus describe` | NodePool status conditions + Pod status |
| **discover_controller()** | Label-based VM lookup | Kubernetes Service by name |
| **Bootstrap** | SSH + install Docker + pull image + run container | `kubectl exec` + simplified script (already in container) |
| **QuotaExhaustedError** | gcloud error parsing | NodePool status conditions (quota/capacity) |
| **Labels for discovery** | GCE labels (`gcloud ... --labels`) | Kubernetes labels on NodePools + Pods |
| **tunnel()** | `gcloud compute ssh -L` | `kubectl port-forward` |

---

## 4. Proposed Integration Design

### Key Design Decisions

**Decision 1: Iris owns all scaling (CoreWeave autoscaler disabled).**

Iris's autoscaler reasons about individual slices with full identity. `ScalingGroup._slices`
is a `dict[str, SliceState]` keyed by `slice_id`. Scale-down picks the *specific longest-idle
slice* via `get_idle_slices()` (sorted by `last_active`). CoreWeave's Cluster Autoscaler
doesn't give you that — it picks which node to remove based on `PreferIdle`/`IdleOnly`
heuristics that don't know about Iris-level task activity.

Running two autoscalers would be a split brain: Iris has backoff, quota tracking, cooldowns,
and waterfall routing across groups; CoreWeave's has its own independent logic. They'd fight.

**Resolution**: Set `autoscaling: false` on all NodePools. **One NodePool = one slice = one
bare metal node.** Iris creates a new NodePool (with `targetNodes: 1`) to scale up, and
deletes the specific NodePool to scale down. This preserves full per-slice addressability.

**Decision 2: `kubectl exec` as a drop-in for SSH (no bootstrap protocol changes).**

The `SshConnection` protocol (`ssh.py:35-61`) is intentionally narrow — `run(command) ->
CompletedProcess` and `run_streaming(command) -> Popen`. All existing implementations
(`GcloudSshConnection`, `DirectSshConnection`, `GceSshConnection`) build a subprocess
command list and shell out. A `KubectlExecConnection` follows the exact same pattern:

```python
@dataclass
class KubectlExecConnection:
    """SshConnection implementation backed by kubectl exec.

    Drop-in replacement for SSH-based connections when workers run as
    Kubernetes Pods. Uses kubectl exec to run commands inside the worker
    container, producing the same subprocess.CompletedProcess / Popen
    results as SSH connections.
    """
    pod_name: str
    namespace: str
    container: str = "iris-worker"
    kubeconfig: str | None = None

    @property
    def address(self) -> str:
        return self.pod_name

    @property
    def zone(self) -> str:
        return self.namespace

    def _build_cmd(self, command: str) -> list[str]:
        cmd = ["kubectl", "exec", self.pod_name,
               "-n", self.namespace, "-c", self.container, "--",
               "bash", "-c", command]
        if self.kubeconfig:
            cmd.insert(1, f"--kubeconfig={self.kubeconfig}")
        return cmd

    def run(self, command: str, timeout: Duration = ...) -> subprocess.CompletedProcess:
        return subprocess.run(
            self._build_cmd(command), capture_output=True, text=True,
            timeout=timeout.to_seconds())

    def run_streaming(self, command: str) -> subprocess.Popen:
        return subprocess.Popen(
            self._build_cmd(command), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)
```

This gives full compatibility with `SshVmBase` — `run_command()`, `bootstrap()`,
`wait_for_connection()`, and `run_streaming_with_retry()` all work unchanged. Only the
*bootstrap script content* changes (worker is already inside a container, so no Docker
install/pull needed), not the bootstrap *protocol*.

### Architecture: One NodePool Per Slice, Pods as Workers

**Scaling flow**:

1. **Scale-up** (`create_slice()`): Create a new CoreWeave NodePool CRD with a unique name
   (the slice ID), `targetNodes: 1`, `autoscaling: false`, and the desired `instanceType`.
   Then deploy a worker Pod on the new node once it's ready. The Pod uses `nodeSelector` to
   bind to the specific NodePool's node.

2. **Worker bootstrap**: The Pod starts the iris-worker container image. Config is injected
   via ConfigMap. `KubectlExecConnection` provides the `SshConnection` interface into the
   running container. The bootstrap script is simplified — no Docker install/pull since the
   container is already running.

3. **Describe** (`describe()`): Query NodePool status conditions (`queuedNodes`,
   `currentNodes`, validation/capacity conditions) + Pod status.

4. **Scale-down** (`terminate()`): `kubectl delete nodepool <slice-id>`. This deterministically
   removes exactly the node Iris chose — no ambiguity about which node is removed.

5. **List** (`list_slices()`): `kubectl get nodepools -l iris-scale-group=X`.

6. **Reconcile**: On controller restart, `list_slices()` discovers existing NodePools by label
   and rebuilds `SliceState` tracking.

**Worker Pod spec**:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: iris-worker-<slice-id>
  namespace: iris
  labels:
    iris-managed: "true"
    iris-scale-group: "h100_8x"
    iris-slice-id: "<slice-id>"
spec:
  nodeSelector:
    # Binds to the specific node from this slice's NodePool
    node.kubernetes.io/instance-type: gd-8xh100ib-i128
    iris-slice-id: "<slice-id>"  # NodePool sets this via nodeLabels
  containers:
    - name: iris-worker
      image: <worker-image>
      command: [".venv/bin/python", "-m", "iris.cluster.worker.main", "serve",
                "--host", "0.0.0.0", "--port", "10001",
                "--cache-dir", "/var/cache/iris",
                "--config", "/etc/iris/config.yaml"]
      resources:
        limits:
          nvidia.com/gpu: 8
          rdma/ib: 1
      ports:
        - containerPort: 10001
      volumeMounts:
        - name: config
          mountPath: /etc/iris
          readOnly: true
        - name: cache
          mountPath: /var/cache/iris
  volumes:
    - name: config
      configMap:
        name: iris-cluster-config
    - name: cache
      emptyDir: {}
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
  restartPolicy: Always
```

**Comparison of scaling approaches**:

| | Iris manages scaling (autoscaler off) | CoreWeave autoscaler |
|---|---|---|
| **Scale-up** | Iris creates a new NodePool (`targetNodes: 1`) | Iris creates a Pod, CW decides when/where to scale |
| **Scale-down** | Iris deletes specific NodePool (deterministic) | CW picks node to remove (non-deterministic) |
| **Slice identity** | NodePool name = slice ID (1:1) | Pod = slice, but node removal is opaque |
| **Idle tracking** | Iris tracks per-slice, picks longest-idle | CW has own `IdleOnly`/`PreferIdle` heuristic |
| **Backoff/quota** | Iris manages fully (NodePool status conditions) | Split brain between two autoscalers |
| **Connection** | `kubectl exec` into Pod (drop-in for SSH) | Same |

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

Implement using the `kubernetes` Python client (or `kubectl` subprocess calls for consistency
with the gcloud pattern):

- `create_slice()`: Create a NodePool CRD (`targetNodes: 1`, `autoscaling: false`) with a
  unique slice ID as the name. Wait for node ready. Deploy worker Pod with `nodeSelector`
  binding to the new node. Return `CoreweaveSliceHandle`.
- `list_slices()`: `kubectl get nodepools -l iris-scale-group=X` (NodePools are the slice identity)
- `list_all_slices()`: `kubectl get nodepools -l iris-managed=true`
- `describe()` on `CoreweaveSliceHandle`: Query NodePool status + Pod status → derive
  `SliceStatus` with `CoreweaveVmHandle` for each worker Pod
- `terminate()` on `CoreweaveSliceHandle`: Delete worker Pod(s), then delete NodePool
- `create_vm()`: Create a Pod (for controller) via Deployment + Service
- `discover_controller()`: Query Kubernetes Service by name
- `tunnel()`: `kubectl port-forward`
- `shutdown()`: Clean up Kubernetes client

### SSH / Connection Layer (`ssh.py`)

Add `KubectlExecConnection` implementing `SshConnection` protocol. This is a drop-in for
`DirectSshConnection` — same `run()` and `run_streaming()` interface, backed by `kubectl exec`
instead of `ssh`. The `SshVmBase` base class, `WorkerBootstrap`, and all retry/streaming
infrastructure work unchanged.

### Bootstrap Changes

The bootstrap *protocol* (`WorkerBootstrap.bootstrap_vm()`) stays the same — it calls
`vm.wait_for_connection()` then `vm.bootstrap(script)` via the `SshConnection` interface.

The bootstrap *script content* is simplified for CoreWeave since the worker is already inside
a container (the Pod). The script just needs to:
1. Write config.yaml (from ConfigMap, or embedded)
2. Start the worker process (or it's already running as the Pod's entrypoint)

No Docker install, no Docker pull, no `gcloud auth configure-docker`.

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

## 7. Resolved Decisions

1. **Autoscaler**: Iris owns all scaling. CoreWeave's Cluster Autoscaler is disabled (`autoscaling: false`). One NodePool per slice, `targetNodes: 1`. This preserves per-slice identity for idle tracking and deterministic scale-down.

2. **SSH vs. Pod exec**: `KubectlExecConnection` implements `SshConnection` protocol as a drop-in. No changes to bootstrap protocol, `SshVmBase`, or retry infrastructure. Only bootstrap script content is simplified.

## 8. Open Questions

1. **Multi-node training slices**: For distributed training across multiple 8-GPU nodes (e.g., 16-GPU job on 2 nodes), the `num_vms` field in `ScaleGroupConfig` maps to multiple Pods that need InfiniBand connectivity. This may require a single NodePool with `targetNodes: N` (breaking the 1:1 NodePool:slice assumption) or N separate NodePools that are logically grouped. CoreWeave handles IB fabric via the `rdma/ib` resource and region affinity, but co-scheduling across NodePools needs investigation.

2. **Controller deployment**: Should the controller run as a Kubernetes Deployment (with a Service for discovery), or as a separate standalone process? Kubernetes Deployment is more idiomatic and gives automatic restarts.

3. **Image registry**: CoreWeave supports any external registry with `imagePullSecrets`. Need to decide where Iris images are published for CoreWeave deployments (GCP Artifact Registry with cross-cloud pull, or a CoreWeave-local registry).

4. **NodePool creation latency**: Creating many small NodePools (one per slice) may hit API rate limits or have overhead vs. a single large pool. Need to validate with CoreWeave that this pattern is supported at scale (e.g., 50+ concurrent NodePools).

5. **Node readiness detection**: After NodePool creation, how to efficiently detect when the bare metal node is ready (20-30 min). Options: poll NodePool status conditions, watch Kubernetes node events, or poll `kubectl get nodes -l iris-slice-id=X`.

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
