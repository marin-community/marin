# CoreWeave Platform Integration for Iris

**Issue**: [#2822 — Iris: Implement CoreWeave platform](https://github.com/marin-community/marin/issues/2822)

## Executive Summary

CoreWeave is entirely Kubernetes-native, running on bare metal (no hypervisor). All compute
provisioning happens through **Kubernetes Node Pool CRDs** (`compute.coreweave.com/v1alpha1`).
Iris already has a `CoreweavePlatform` stub and proto definitions. The integration maps Iris
scaling groups to CoreWeave Node Pools and deploys both controller and workers as Kubernetes
Pods.

This document covers the full system of control: from cluster bootstrap through controller
deployment, worker scaling, task execution, and a plan for a minimal end-to-end test.

---

## 1. CoreWeave Architecture

**Bare metal Kubernetes**: CoreWeave runs Kubernetes directly on bare metal GPU nodes (CKS =
CoreWeave Kubernetes Service). No hypervisor. Nodes are stateless, rebooted from clean images,
isolated via NVIDIA BlueField DPUs.

**Node Pools** are the provisioning primitive — Kubernetes CRDs:

```yaml
apiVersion: compute.coreweave.com/v1alpha1
kind: NodePool
metadata:
  name: iris-h100-pool
spec:
  computeClass: default
  instanceType: gd-8xh100ib-i128  # immutable after creation
  targetNodes: 1
  autoscaling: false
  nodeLabels:
    iris-managed: "true"
    iris-scale-group: "h100_8x"
```

Key constraints:
- `instanceType` is immutable (one Node Pool per GPU config)
- Scale-up takes **20-30 minutes** (bare metal boot)
- Quota must cover `targetNodes`; exceeding quota fails the entire create

**GPU instance types**: GB300 NVL72, GB200 NVL72, B200, H200, H100, RTX Pro 6000, L40S, L40,
GH200, A100 — all 8-GPU (or 4-GPU for Grace) configs.

**GPU selection labels**: `gpu.nvidia.com/class` (H100), `node.kubernetes.io/instance-type`
(gd-8xh100ib-i128), `topology.kubernetes.io/region` (LGA1).

---

## 2. Design Decisions

### Decision 1: Iris owns all scaling (CoreWeave autoscaler disabled)

Iris's autoscaler tracks individual slices with full identity (`ScalingGroup._slices` is a
`dict[str, SliceState]`). Scale-down picks the *specific longest-idle slice* via
`get_idle_slices()`. CoreWeave's Cluster Autoscaler doesn't give per-slice addressability —
it picks which node to remove based on its own heuristics.

**Resolution**: `autoscaling: false` on all NodePools. **One NodePool = one slice = one bare
metal node.** Iris creates NodePool (scale-up) / deletes NodePool (scale-down).

### Decision 2: Controller runs on CoreWeave as a Kubernetes Deployment

The controller runs as a `Deployment` with a `Service` for stable DNS-based discovery.
Workers find the controller via the Service name. No label-based VM discovery needed.

### Decision 3: `kubectl exec` as drop-in for SSH

The `SshConnection` protocol (`ssh.py`) is narrow: `run(command) -> CompletedProcess`,
`run_streaming(command) -> Popen`. A `KubectlExecConnection` follows the same pattern. All
existing infrastructure (`SshVmBase`, `WorkerBootstrap`, `run_streaming_with_retry`) works
unchanged.

### Decision 4: Workers run as Kubernetes Pods (no Docker-in-Docker)

On GCP, workers are Docker containers on bare-metal VMs. On CoreWeave, workers *are* the Pod.
The bootstrap protocol stays the same (via `kubectl exec`), but the script is simplified —
no Docker install/pull. Task execution uses the host Docker socket mounted into the Pod,
same as GCP.

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  CoreWeave CKS Cluster                                           │
│                                                                  │
│  ┌─────────────────────────────────┐                             │
│  │  Controller Deployment          │                             │
│  │  (always-on, small node pool)   │                             │
│  │                                 │                             │
│  │  Pod: iris-controller           │                             │
│  │    iris-controller:latest       │                             │
│  │    port 10000                   │                             │
│  │    /etc/iris/config.yaml        │ ◄── ConfigMap               │
│  │    bundle_prefix: s3://...      │                             │
│  └────────┬────────────────────────┘                             │
│           │                                                      │
│  Service: iris-controller-svc ──────────────────────────┐        │
│  (ClusterIP, port 10000)                                │        │
│           │                                             │        │
│  ┌────────▼────────────────────────┐  ┌─────────────────▼──────┐ │
│  │  Worker NodePool: iris-slice-A  │  │ Worker NodePool: ...   │ │
│  │  instanceType: gd-8xh100ib-i128│  │ (one per active slice) │ │
│  │  targetNodes: 1                 │  │                        │ │
│  │  autoscaling: false             │  │                        │ │
│  │                                 │  │                        │ │
│  │  Pod: iris-worker-slice-A       │  │  Pod: iris-worker-...  │ │
│  │    iris-worker:latest           │  │    iris-worker:latest  │ │
│  │    port 10001                   │  │    port 10001          │ │
│  │    nvidia.com/gpu: 8            │  │    nvidia.com/gpu: 8   │ │
│  │    /var/run/docker.sock mounted │  │                        │ │
│  │    /var/cache/iris (hostPath)   │  │                        │ │
│  └─────────────────────────────────┘  └────────────────────────┘ │
│                                                                  │
│  ConfigMap: iris-cluster-config                                  │
│  Secret: iris-image-pull (imagePullSecrets)                      │
│  Secret: iris-bundle-credentials (S3/GCS access)                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Control Flow: End-to-End Lifecycle

### Phase 0: Cluster Bootstrap (one-time setup)

This happens before Iris runs. It provisions the CKS cluster and the baseline resources that
the controller needs.

**Prerequisites**:
1. CKS cluster created (via Terraform `coreweave_cks_cluster` or CoreWeave UI)
2. `kubeconfig` obtained for the cluster
3. `kubectl` configured and authenticated

**Baseline Kubernetes resources** (applied via `kubectl apply -f`):

```yaml
# 1. Namespace
apiVersion: v1
kind: Namespace
metadata:
  name: iris
---
# 2. ConfigMap with cluster config (generated from iris config YAML)
apiVersion: v1
kind: ConfigMap
metadata:
  name: iris-cluster-config
  namespace: iris
data:
  config.yaml: |
    platform:
      coreweave:
        region: LGA1
        namespace: iris
    controller:
      coreweave:
        port: 10000
        service_name: iris-controller-svc
    defaults:
      bootstrap:
        docker_image: <registry>/iris-worker:latest
        worker_port: 10001
      timeouts:
        boot_timeout: { milliseconds: 1800000 }   # 30 min (bare metal)
        init_timeout: { milliseconds: 1800000 }
      autoscaler:
        evaluation_interval: { milliseconds: 10000 }
        scale_up_delay: { milliseconds: 60000 }
        scale_down_delay: { milliseconds: 300000 }
        startup_grace_period: { milliseconds: 2400000 }  # 40 min
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
          accelerator_type: gpu
          accelerator_variant: H100
          num_vms: 1
          coreweave:
            region: LGA1
            instance_type: gd-8xh100ib-i128
---
# 3. Image pull secret (for worker/controller images)
apiVersion: v1
kind: Secret
metadata:
  name: iris-image-pull
  namespace: iris
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: <base64-encoded docker config>
---
# 4. Bundle storage credentials (S3 or GCS service account key)
apiVersion: v1
kind: Secret
metadata:
  name: iris-bundle-credentials
  namespace: iris
data:
  credentials.json: <base64-encoded service account key>
---
# 5. Controller NodePool (always-on, small instance for controller)
apiVersion: compute.coreweave.com/v1alpha1
kind: NodePool
metadata:
  name: iris-controller-pool
spec:
  computeClass: default
  instanceType: <cpu-only-instance>  # small, no GPU needed
  targetNodes: 1
  autoscaling: false
  nodeLabels:
    iris-role: controller
---
# 6. Controller Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iris-controller
  namespace: iris
spec:
  replicas: 1
  selector:
    matchLabels:
      app: iris-controller
  template:
    metadata:
      labels:
        app: iris-controller
    spec:
      nodeSelector:
        iris-role: controller
      imagePullSecrets:
        - name: iris-image-pull
      containers:
        - name: controller
          image: <registry>/iris-controller:latest
          command:
            - .venv/bin/python
            - -m
            - iris.cluster.controller.main
            - serve
            - --host=0.0.0.0
            - --port=10000
            - --config=/etc/iris/config.yaml
            - --bundle-prefix=s3://iris-bundles/jobs
          ports:
            - containerPort: 10000
          volumeMounts:
            - name: config
              mountPath: /etc/iris
              readOnly: true
            - name: bundle-credentials
              mountPath: /etc/iris/credentials
              readOnly: true
          env:
            - name: GOOGLE_APPLICATION_CREDENTIALS
              value: /etc/iris/credentials/credentials.json
            # Or for S3:
            - name: AWS_SHARED_CREDENTIALS_FILE
              value: /etc/iris/credentials/credentials
          livenessProbe:
            httpGet:
              path: /health
              port: 10000
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 10000
            initialDelaySeconds: 5
            periodSeconds: 10
      volumes:
        - name: config
          configMap:
            name: iris-cluster-config
        - name: bundle-credentials
          secret:
            secretName: iris-bundle-credentials
---
# 7. Controller Service (stable DNS endpoint for workers)
apiVersion: v1
kind: Service
metadata:
  name: iris-controller-svc
  namespace: iris
spec:
  selector:
    app: iris-controller
  ports:
    - port: 10000
      targetPort: 10000
  type: ClusterIP
```

### Phase 1: Controller Startup

The controller Pod starts and runs:
```
python -m iris.cluster.controller.main serve --config /etc/iris/config.yaml
```

Startup sequence (same as GCP, from `controller/main.py`):
1. `load_config()` reads `/etc/iris/config.yaml` (from ConfigMap)
2. `create_platform()` returns `CoreweavePlatform` (dispatched by `factory.py`)
3. `WorkerBootstrap(cluster_config)` created
4. `create_autoscaler()` creates `ScalingGroup` per config entry
5. `autoscaler.reconcile()` — discovers existing NodePools/Pods by label
6. `Controller.start()` — scheduling loop, heartbeat loop, autoscaler loop, HTTP server

### Phase 2: Scale-Up (Autoscaler Creates Worker)

When demand arrives (job submitted), the autoscaler evaluates:

1. **Autoscaler loop** (`controller.py:_run_autoscaler_once`):
   - `compute_demand_entries()` finds pending tasks
   - `route_demand()` picks scale group by device type + waterfall priority
   - `_evaluate_group()` decides scale-up needed
   - `_execute_scale_up(group)` spawns background thread

2. **Scale-up thread** (`autoscaler.py:_do_scale_up`):
   - `group.scale_up()` calls `platform.create_slice(config)`

3. **`CoreweavePlatform.create_slice()`** (new implementation):
   ```
   a. Generate unique slice ID: iris-{group}-{timestamp_ms}
   b. kubectl apply NodePool CRD:
        name: {slice_id}
        instanceType: {config.coreweave.instance_type}
        targetNodes: 1
        autoscaling: false
        nodeLabels:
          iris-managed: "true"
          iris-scale-group: {group_name}
          iris-slice-id: {slice_id}
   c. Return CoreweaveSliceHandle(slice_id, ...)
   ```

4. **Bootstrap thread** (`autoscaler.py:_bootstrap_slice`):
   - Calls `bootstrap_slice_vms(handle, worker_bootstrap)`
   - Polls `handle.describe()` until node ready (up to 30 min)
   - On node ready: `CoreweavePlatform` creates worker Pod:
     ```
     kubectl apply Pod:
       name: iris-worker-{slice_id}
       nodeSelector:
         iris-slice-id: {slice_id}
       containers:
         - name: iris-worker
           image: {docker_image}
           command: [worker serve ...]
           resources:
             limits:
               nvidia.com/gpu: 8
               rdma/ib: 1
           volumeMounts:
             /etc/iris from ConfigMap
             /var/cache/iris from hostPath
             /var/run/docker.sock from hostPath  # for task containers
     ```
   - `describe()` returns `CoreweaveVmHandle` wrapping the Pod
   - `WorkerBootstrap.bootstrap_vm()` connects via `KubectlExecConnection`
   - Bootstrap script (simplified — no Docker install):
     ```bash
     echo "[iris-init] Worker pod bootstrap"
     echo "[iris-init] Config already mounted at /etc/iris/config.yaml"
     echo "[iris-init] Worker process running as pod entrypoint"
     echo "[iris-init] Phase: registration"
     # Worker is already running — just verify health
     for i in $(seq 1 60); do
       if curl -sf http://localhost:10001/health > /dev/null 2>&1; then
         echo "[iris-init] Worker is healthy"
         echo "[iris-init] Bootstrap complete"
         exit 0
       fi
       sleep 2
     done
     echo "[iris-init] ERROR: Worker not healthy after 120s"
     exit 1
     ```

### Phase 3: Worker Registration

Worker Pod entrypoint runs:
```
python -m iris.cluster.worker.main serve --host 0.0.0.0 --port 10001 \
    --cache-dir /var/cache/iris --config /etc/iris/config.yaml
```

Worker startup:
1. `load_config()` reads config from ConfigMap mount
2. `create_platform()` returns `CoreweavePlatform`
3. **`platform.discover_controller()`** → resolves Kubernetes Service:
   ```python
   def discover_controller(self, controller_config):
       svc = controller_config.coreweave.service_name  # "iris-controller-svc"
       port = controller_config.coreweave.port or 10000
       namespace = self._config.namespace or "iris"
       return f"{svc}.{namespace}.svc.cluster.local:{port}"
   ```
4. `DefaultEnvironmentProvider.probe()`:
   - `_is_gcp_vm()` returns False (not on GCP)
   - GPU detected via `nvidia-smi` (NVIDIA GPUs are exposed to Pod)
   - IP address probed via UDP socket
   - `IRIS_VM_ADDRESS` set by Pod env or downward API
5. `Worker._register()` sends `RegisterRequest` to controller via Service DNS
6. Controller accepts, assigns `worker_id`
7. Worker enters heartbeat serve loop

### Phase 4: Task Execution

1. Client submits job via `IrisClient.submit()`
2. Controller scheduler assigns task to worker in heartbeat
3. Worker receives `RunTaskRequest` in heartbeat response
4. Worker downloads bundle from `bundle_prefix` (S3 or GCS)
5. Worker creates task container via Docker socket:
   - Uses `iris-task:latest` base image (pre-pulled by init container or hostPath)
   - Mounts workdir, UV cache
   - Runs `uv sync` + user command
6. Task completes, worker reports in next heartbeat
7. Controller marks job complete

### Phase 5: Scale-Down

1. Autoscaler `refresh()` phase detects idle slice (longest idle first)
2. `group.scale_down(slice_id)` calls `handle.terminate()`
3. **`CoreweaveSliceHandle.terminate()`**:
   ```
   a. kubectl delete pod iris-worker-{slice_id} -n iris
   b. kubectl delete nodepool {slice_id}
   ```
4. Node deprovisioned (deterministic — exactly the chosen slice)

---

## 5. Secrets and Configuration Required

### Kubernetes Secrets

| Secret | Purpose | Contents |
|--------|---------|----------|
| `iris-image-pull` | Pull controller/worker/task images | Docker config JSON (`imagePullSecrets`) |
| `iris-bundle-credentials` | Access bundle storage (GCS or S3) | Service account JSON (GCS) or AWS credentials (S3) |

### ConfigMap

| ConfigMap | Purpose | Contents |
|-----------|---------|----------|
| `iris-cluster-config` | Full Iris cluster config | `config.yaml` with platform, controller, scale_groups |

### Environment Variables

| Variable | Where | Purpose |
|----------|-------|---------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Controller + Worker Pods | GCS access for bundles (if using GCS) |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | Controller + Worker Pods | S3 access for bundles (if using CoreWeave S3) |
| `IRIS_VM_ADDRESS` | Worker Pods | Pod IP for autoscaler VM tracking (use Downward API) |
| `WANDB_API_KEY` | Worker Pods (optional) | Weights & Biases logging |
| `KUBECONFIG` | Controller Pod | Access to CKS API for NodePool/Pod management |

### RBAC

The controller Pod needs a ServiceAccount with permissions to:
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: iris-controller
rules:
  # NodePool management (CoreWeave CRD)
  - apiGroups: ["compute.coreweave.com"]
    resources: ["nodepools"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  # Pod management (worker Pods)
  - apiGroups: [""]
    resources: ["pods", "pods/exec", "pods/log"]
    verbs: ["get", "list", "watch", "create", "delete"]
  # Node status (readiness detection)
  - apiGroups: [""]
    resources: ["nodes"]
    verbs: ["get", "list", "watch"]
  # ConfigMap read
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get"]
```

### Container Images

| Image | Where | Build |
|-------|-------|-------|
| `iris-controller:latest` | Controller Deployment | `Dockerfile.controller` (remove gcloud, add kubectl) |
| `iris-worker:latest` | Worker Pods | `Dockerfile.worker` (remove gcloud, add kubectl) |
| `iris-task:latest` | Task containers on workers | `Dockerfile.task` (unchanged) |

**CoreWeave-specific Dockerfile changes**:
- Remove `google-cloud-cli` package (not needed)
- Add `kubectl` binary (for `KubectlExecConnection` and NodePool management)
- The controller image needs the `kubernetes` Python package (or uses `kubectl` subprocess)

---

## 6. Code Changes: File-by-File

### `config.proto` — Extend proto definitions

```protobuf
message CoreweavePlatformConfig {
  string region = 1;
  string namespace = 2;              // Kubernetes namespace (default: "iris")
  string kubeconfig_path = 3;        // path to kubeconfig (optional, uses in-cluster if empty)
}

message CoreweaveSliceConfig {
  string region = 1;
  string instance_type = 2;          // e.g. "gd-8xh100ib-i128"
  int32 gpus_per_node = 3;           // e.g. 8
  string gpu_class = 4;              // e.g. "H100"
  bool infiniband = 5;               // request rdma/ib: 1
}

message CoreweaveControllerConfig {
  int32 port = 1;                    // default: 10000
  string service_name = 2;           // K8s Service name for discovery
}

// Add to ControllerVmConfig oneof:
message ControllerVmConfig {
  ...
  oneof controller {
    GcpControllerConfig gcp = 1;
    ManualControllerConfig manual = 2;
    LocalControllerConfig local = 3;
    CoreweaveControllerConfig coreweave = 4;  // NEW
  }
}
```

### `ssh.py` — Add `KubectlExecConnection`

New class implementing `SshConnection` protocol. Uses `kubectl exec` subprocess calls.
Same interface as `DirectSshConnection`.

### `coreweave.py` — Implement `CoreweavePlatform`

Replace stub with full implementation:
- `CoreweavePlatform.__init__`: Load kubeconfig (in-cluster or from file)
- `create_slice()`: `kubectl apply` NodePool CRD, poll for ready, create worker Pod
- `list_slices()`: `kubectl get nodepools -l iris-scale-group=X -o json`
- `list_all_slices()`: `kubectl get nodepools -l iris-managed=true -o json`
- `create_vm()`: `kubectl apply` Pod for standalone VM (controller)
- `list_vms()`: `kubectl get pods -l iris-role=controller -o json`
- `tunnel()`: `kubectl port-forward` context manager
- `discover_controller()`: `{service_name}.{namespace}.svc.cluster.local:{port}`
- `shutdown()`: no-op

New handle classes:
- `CoreweaveSliceHandle`: Wraps NodePool name + worker Pod(s)
  - `slice_id`: NodePool name
  - `describe()`: Query NodePool status + Pod status → `SliceStatus`
  - `terminate()`: Delete Pod, then delete NodePool
- `CoreweaveVmHandle`: Wraps worker Pod with `KubectlExecConnection`
  - `vm_id`: Pod name
  - `internal_address`: Pod IP
  - `run_command()`, `bootstrap()` etc. via `SshVmBase` + `KubectlExecConnection`
  - `status()`: Map Pod phase to `CloudVmState`
  - `wait_for_connection()`: Poll `kubectl exec echo ok`

### `bootstrap.py` — Add CoreWeave bootstrap script

New function `build_coreweave_worker_bootstrap_script()` that generates a simplified script:
- Config already mounted via ConfigMap
- Worker already running as Pod entrypoint
- Script only verifies health + does any init

### `factory.py` — Pass kubeconfig to CoreweavePlatform

Minor change to pass additional config fields.

### `config.py` — Add CoreWeave defaults

Add CoreWeave-specific timeout defaults (30min boot, 40min startup grace).

### `env_probe.py` — CoreWeave GPU detection

GPU detection via `nvidia-smi` already works on CoreWeave (NVIDIA device plugin exposes
GPUs to Pods). `_is_gcp_vm()` returns False. Pod IP detected via socket probe. Add
CoreWeave-specific attribute detection if needed.

### New Dockerfiles

`Dockerfile.controller.coreweave`:
- Base: `python:3.11-slim`
- Install: `curl`, `git`, `kubectl`
- No `google-cloud-cli`
- Install iris via `uv sync`

`Dockerfile.worker.coreweave`:
- Base: `python:3.11-slim`
- Install: `curl`, `git`, `kubectl`, `docker-ce-cli`
- No `google-cloud-cli`
- Install iris via `uv sync`

---

## 7. Minimal End-to-End Test Plan

### Goal

Submit a simple CPU job to an Iris cluster running entirely on CoreWeave, verify it succeeds.
This validates the full lifecycle: controller startup, worker scaling, task execution, and
scale-down.

### Test Infrastructure

For a minimal test, we don't need GPU nodes. Use a small CPU-only instance type:

```yaml
# tests/e2e/coreweave-test-config.yaml
platform:
  coreweave:
    region: LGA1
    namespace: iris-test

controller:
  coreweave:
    port: 10000
    service_name: iris-controller-svc

defaults:
  bootstrap:
    docker_image: <registry>/iris-worker:latest
    worker_port: 10001
  timeouts:
    boot_timeout: { milliseconds: 1800000 }
    init_timeout: { milliseconds: 1800000 }
  autoscaler:
    evaluation_interval: { milliseconds: 5000 }
    scale_up_delay: { milliseconds: 5000 }
    scale_down_delay: { milliseconds: 30000 }    # fast for testing
    startup_grace_period: { milliseconds: 2400000 }

scale_groups:
  cpu_test:
    accelerator_type: cpu
    accelerator_variant: cpu
    num_vms: 1
    resources:
      cpu: 4
      ram: 16GB
      disk: 50GB
      gpu_count: 0
    min_slices: 0
    max_slices: 2
    slice_template:
      accelerator_type: cpu
      accelerator_variant: cpu
      num_vms: 1
      coreweave:
        region: LGA1
        instance_type: <cpu-only-instance>  # cheapest available
        gpus_per_node: 0
```

### Test Steps

```python
# tests/e2e/test_coreweave_smoke.py
import pytest
from iris.rpc import cluster_pb2

pytestmark = [pytest.mark.e2e, pytest.mark.coreweave]


def _hello():
    """Minimal task: just return a value."""
    return 42


@pytest.fixture
def coreweave_cluster():
    """Connect to a running CoreWeave Iris cluster.

    Assumes the cluster is already deployed via kubectl apply.
    The IRIS_COREWEAVE_URL env var points to the controller
    (e.g., via kubectl port-forward).
    """
    import os
    from iris.client import IrisClient
    url = os.environ["IRIS_COREWEAVE_URL"]
    client = IrisClient.remote(url)
    yield client


def test_submit_and_succeed(coreweave_cluster):
    """Minimal test: submit a job, wait for success."""
    job = coreweave_cluster.submit(_hello, "coreweave-smoke")
    status = coreweave_cluster.wait(job, timeout=2400)  # 40 min for bare metal boot
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
```

### Running the Test

```bash
# 1. Deploy cluster (one-time)
kubectl apply -f tests/e2e/coreweave-bootstrap/

# 2. Wait for controller to be ready
kubectl wait --for=condition=available deployment/iris-controller -n iris-test --timeout=300s

# 3. Port-forward to controller
kubectl port-forward svc/iris-controller-svc 10000:10000 -n iris-test &

# 4. Run test
IRIS_COREWEAVE_URL=http://localhost:10000 uv run pytest tests/e2e/test_coreweave_smoke.py -v -m coreweave
```

### What the Test Validates

1. Controller starts from ConfigMap and connects to CKS API
2. `CoreweavePlatform.create_slice()` creates a NodePool and worker Pod
3. Worker boots, discovers controller via K8s Service DNS, registers
4. Controller schedules task to worker
5. Worker downloads bundle, executes task, reports completion
6. (Optional) Autoscaler scales down idle slice after test

### Secrets Required for Test

| Secret | How to Obtain |
|--------|---------------|
| CKS kubeconfig | CoreWeave dashboard or `coreweave` Terraform output |
| Image registry credentials | Docker config for wherever images are pushed |
| Bundle storage credentials | GCS service account key or CoreWeave S3 credentials |
| (Optional) WANDB_API_KEY | For logging, not required for smoke test |

---

## 8. Implementation Order

### Phase 1: Core Platform (get `create_slice` + `terminate` working)

1. Extend `config.proto`: `CoreweavePlatformConfig`, `CoreweaveSliceConfig`, `CoreweaveControllerConfig`
2. Regenerate proto Python bindings
3. Implement `KubectlExecConnection` in `ssh.py`
4. Implement `CoreweaveVmHandle` (wraps Pod + `KubectlExecConnection`)
5. Implement `CoreweaveSliceHandle` (wraps NodePool + Pod)
6. Implement `CoreweavePlatform` core methods: `create_slice`, `list_slices`, `terminate`, `discover_controller`
7. Unit tests with `FakePlatform`-style mocks for `kubectl` subprocess calls

### Phase 2: Bootstrap + Controller

8. Add CoreWeave bootstrap script (simplified, no Docker install)
9. Build CoreWeave-specific Dockerfiles (controller + worker)
10. Write Kubernetes manifests for controller Deployment + Service
11. Add RBAC ServiceAccount + ClusterRole
12. Test controller starts and runs autoscaler loop in a CKS cluster

### Phase 3: End-to-End

13. Write `coreweave-test-config.yaml` and bootstrap manifests
14. Deploy to CKS, run `test_coreweave_smoke.py`
15. Validate full lifecycle: scale-up, task execution, scale-down

### Phase 4: Production Hardening

16. NodePool status condition parsing (quota errors, capacity unavailable)
17. Timeout tuning for bare metal boot times
18. Multi-node slice support (NodePool with `targetNodes: N`)
19. Monitoring: expose Prometheus metrics from controller Pod
20. CI integration: automated CoreWeave test runs

---

## 9. Resolved Decisions

1. **Autoscaler**: Iris owns all scaling. `autoscaling: false`. One NodePool per slice.
2. **SSH -> kubectl exec**: `KubectlExecConnection` as `SshConnection` drop-in. No protocol changes.
3. **Controller**: Runs as Kubernetes Deployment + Service on CoreWeave. Workers discover via Service DNS.
4. **Bootstrap**: Same protocol, simplified script. Config via ConfigMap, worker runs as Pod entrypoint.

## 10. Open Questions

1. **Multi-node slices**: `num_vms > 1` needs a single NodePool with `targetNodes: N` or N
   separate NodePools grouped logically. InfiniBand co-scheduling across NodePools needs
   investigation.

2. **NodePool creation rate limits**: Creating many 1-node NodePools may hit API limits.
   Need to validate with CoreWeave at scale (50+ concurrent pools).

3. **Task container execution model**: Current workers use Docker socket to run task containers.
   On CoreWeave Pods, this requires `hostPath` mount of `/var/run/docker.sock`. If Docker is not
   available on bare metal nodes, tasks may need to run as child Pods instead.

4. **Bundle storage**: GCS (`gs://`) works if controller/workers have GCS credentials. CoreWeave's
   native S3-compatible storage (`s3://`) may be more natural. Need to confirm fsspec/`smart_open`
   support for the chosen path.

5. **Image registry**: Where to push Iris images — GCP Artifact Registry (cross-cloud pull),
   Docker Hub, GitHub Container Registry, or CoreWeave's own registry.

---

## References

- [CoreWeave CKS Introduction](https://docs.coreweave.com/docs/products/cks)
- [CoreWeave Node Pools](https://docs.coreweave.com/docs/products/cks/nodes/nodes-and-node-pools)
- [CoreWeave Node Pool Reference](https://docs.coreweave.com/docs/products/cks/reference/node-pool)
- [CoreWeave Autoscaling](https://docs.coreweave.com/docs/products/cks/nodes/autoscaling)
- [CoreWeave GPU Instances](https://docs.coreweave.com/docs/products/instances/gpu-instances)
- [CoreWeave Workload Scheduling](https://docs.coreweave.com/docs/products/cks/clusters/scheduling/workload-scheduling)
- [CoreWeave Terraform Provider](https://docs.coreweave.com/docs/products/cks/terraform/about)
- [Kubernetes Cluster Autoscaler — CoreWeave cloud provider](https://github.com/kubernetes/autoscaler)
