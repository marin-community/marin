# CoreWeave Platform Integration for Iris

**Issue**: [#2822 — Iris: Implement CoreWeave platform](https://github.com/marin-community/marin/issues/2822)

## Executive Summary

CoreWeave is entirely Kubernetes-native, running on bare metal (no hypervisor). All compute
provisioning happens through **Kubernetes Node Pool CRDs** (`compute.coreweave.com/v1alpha1`).
Iris already has a `CoreweavePlatform` stub and proto definitions. The integration maps Iris
scaling groups to CoreWeave Node Pools and deploys both controller and workers as Kubernetes
Pods.

This document covers: the operator setup guide, the boundary between operator responsibility
and platform.py runtime, the full control flow from cluster bootstrap through task execution,
and a plan for a minimal end-to-end test.

The implementation is structured as **two spirals**. Spiral 1 refactors the platform layer
(rename `VmHandle` to `RemoteWorkerHandle`, move bootstrap into `Platform.create_slice()`,
make `create_slice()` return immediately with async state monitoring, add `BootstrapConfig`,
simplify the autoscaler) on existing platforms (GCP + Manual + Local) before writing any
CoreWeave code. Spiral 2 builds `CoreweavePlatform` and `ContainerdRuntime` on the clean
interface from Spiral 1.

Key architectural decisions vs. the GCP platform:
- **`VmHandle` renamed to `RemoteWorkerHandle`**: On CoreWeave it is a Pod, not a VM. The
  protocol name now reflects what it represents: a handle to a remote worker process. All
  platform-layer types follow suit (`CloudVmState` -> `CloudWorkerState`,
  `CoreweaveVmHandle` -> `CoreweaveWorkerHandle`, etc.).
- **Bootstrap moves into Platform with async state model**: `Platform.create_slice(config,
  bootstrap_config)` returns a `SliceHandle` **immediately** after initiating creation. The
  platform monitors bootstrap progress internally and exposes state via
  `SliceHandle.describe()` (`CREATING -> BOOTSTRAPPING -> READY | FAILED`). The autoscaler
  observes these state transitions -- it no longer runs bootstrap logic.
  `WorkerBootstrap` is removed as a separate class. If any worker fails during bootstrap,
  the platform fails the entire slice and cleans up its own resources.
- **Task containers use containerd via `crictl`**: CoreWeave bare metal nodes run containerd,
  not Docker. A new `ContainerdRuntime` implements the existing `ContainerRuntime` protocol
  using `crictl` (the CRI-compatible CLI for containerd). The single worker image ships with
  both `docker-ce-cli` and `crictl`; the `--runtime` flag selects which backend to use.
- **`SshConnection` is renamed to `RemoteExec`**: The protocol is transport-agnostic (just
  `run(command)` and `run_streaming(command)`). The rename clarifies intent. CoreWeave does
  not add a new `RemoteExec` implementation -- the protocol is only used on GCP/Manual
  platforms where SSH is the transport.
- **`bootstrap()` and `wait_for_connection()` removed from `RemoteWorkerHandle` protocol**:
  These are implementation details of SSH-based platforms. They become methods on
  `RemoteExecWorkerBase` only. CoreWeave's `CoreweaveWorkerHandle` does not implement them.
- **Public images on ghcr.io -- no auth needed for `crictl pull`**: All images are public.
  `crictl pull ghcr.io/marin-community/iris-task:latest` works without credentials.
  `imagePullSecrets` is only needed for private registry deployments.

---

## 1. Responsibility Boundary: Operator vs. `platform.py`

A clean separation between what a human operator does (one-time / infrequent) and what
`CoreweavePlatform` does at runtime (every scale-up/down cycle):

```
OPERATOR (one-time / config change)           PLATFORM.PY (runtime, every scale cycle)
─────────────────────────────────────         ─────────────────────────────────────────
1. Create CKS cluster (Terraform/UI)         1. create_slice(config, bootstrap_config):
2. Obtain kubeconfig + API token                 kubectl apply NodePool CRD
3. Push images to ghcr.io                        kubectl apply worker Pod (with bootstrap
4. kubectl apply:                                  config baked into Pod spec)
   - Namespace                                   return SliceHandle immediately (CREATING)
   - ConfigMap (iris config)                     monitor state internally:
   - Secrets (bundle creds)                        CREATING -> BOOTSTRAPPING -> READY|FAILED
   - RBAC (ServiceAccount, ClusterRole)      2. list_slices():
   - Controller NodePool                         kubectl get nodepools -l ...
   - Controller Deployment + Service          3. describe():
5. Verify controller is healthy                  kubectl get nodepool status
6. Port-forward for external access              kubectl get pod status
                                              4. terminate():
                                                  kubectl delete pod
                                                  kubectl delete nodepool
                                              5. discover_controller():
                                                  return K8s Service DNS name
```

The controller Deployment, Service, ConfigMap, Secrets, and RBAC are all **operator-managed
infrastructure**. `platform.py` never creates or modifies these. It only dynamically creates
and destroys **worker NodePools + Pods** in response to autoscaler decisions.

This mirrors the GCP pattern: the operator sets up the GCP project, configures `gcloud auth`,
and writes the config YAML. `GcpPlatform` only calls `gcloud compute tpus create/delete` at
runtime.

---

## 2. Operator Guide: Setting Up Iris on CoreWeave

### Step 1: Create a CoreWeave Account and API Token

1. Sign up at [cloud.coreweave.com](https://cloud.coreweave.com)
2. Navigate to **Tokens** in the sidebar
3. Click **Create Token**
4. Name it (e.g., `iris-admin`), set expiration, click **Create**
5. **Copy the token immediately** -- it is shown only once. Format: `CW-SECRET-XXXXXXXXXXXXX`

```bash
# Save for later steps
export COREWEAVE_API_TOKEN="CW-SECRET-XXXXXXXXXXXXX"
```

### Step 2: Create a CKS Cluster

**Option A: Terraform (recommended for reproducibility)**

```hcl
# infra/coreweave/main.tf
terraform {
  required_providers {
    coreweave = {
      source  = "coreweave/coreweave"
      version = ">= 0.3.0"
    }
  }
}

provider "coreweave" {
  # Uses COREWEAVE_API_TOKEN env var
}

resource "coreweave_cks_cluster" "iris" {
  name               = "iris-cluster"
  kubernetes_version = "1.32"
  zone               = "US-EAST-04A"

  vpc = {
    name        = "iris-vpc"
    host_prefix = "10.16.192.0/18"
    vpc_prefixes = {
      pod     = "10.17.0.0/16"
      service = "10.18.0.0/16"
      lb      = "10.19.0.0/16"
    }
  }
}
```

```bash
cd infra/coreweave
terraform init
terraform apply
```

**Option B: CoreWeave Console UI**

1. Navigate to **CKS** > **Create Cluster**
2. Choose name, Kubernetes version (1.32+), zone (e.g., `US-EAST-04A`)
3. Configure VPC (defaults are fine for testing)
4. Skip Auth (use Managed Auth defaults)
5. Click **Deploy**, wait for status to show **Healthy**

### Step 3: Obtain Kubeconfig

1. In the CoreWeave Console, go to **Tokens**
2. Click the token you created in Step 1
3. Click **Download Kubeconfig**, selecting your cluster
4. Save the file:

```bash
mkdir -p ~/.kube
mv ~/Downloads/kubeconfig.yaml ~/.kube/coreweave-iris
export KUBECONFIG=~/.kube/coreweave-iris

# Verify connectivity
kubectl cluster-info
kubectl get nodes   # empty until NodePools are created
```

The kubeconfig uses CoreWeave **Managed Auth** -- your API token (`CW-SECRET-...`) is embedded
in the kubeconfig as a bearer token. This authenticates against CoreWeave's managed auth
endpoint.

**Important**: The controller Pod running *inside* the cluster uses **in-cluster service
account authentication**, not this kubeconfig. In-cluster auth uses the standard Kubernetes
service account token mounted at `/var/run/secrets/kubernetes.io/serviceaccount/token`.
This works against the **unmanaged auth endpoint** which CKS provides automatically.

### Step 4: Build and Push Images

Iris images are built via `iris build` CLI (`lib/iris/src/iris/cli/build.py`), which currently
pushes to GCP Artifact Registry. For CoreWeave, images also need to be on
`ghcr.io/marin-community/` (public). The CLI will be extended to support ghcr.io as a
registry target (see [Code Changes](#8-code-changes-file-by-file)).

```bash
# Build all images (from marin repo root)
uv run iris build worker-image --tag iris-worker:latest
uv run iris build controller-image --tag iris-controller:latest --dockerfile lib/iris/Dockerfile.controller.coreweave
uv run iris build task-image --tag iris-task:latest

# Push to ghcr.io (new --registry ghcr flag, to be implemented)
uv run iris build push iris-worker:latest --registry ghcr --image-name iris-worker
uv run iris build push iris-controller:latest --registry ghcr --image-name iris-controller-coreweave
uv run iris build push iris-task:latest --registry ghcr --image-name iris-task
```

All images on `ghcr.io/marin-community/` are **public** -- consumers (including `crictl pull`
on CoreWeave nodes) do not need auth to pull. The image references in the ConfigMap (Step 6)
must match the published tags.

### Step 5: Create Kubernetes Secrets

Since images are public, no `imagePullSecrets` is needed for pulling from ghcr.io.
The only secret required is for bundle storage credentials:

```bash
kubectl create namespace iris

# Bundle storage credentials
# Option A: GCS (reuse existing marin credentials)
kubectl create secret generic iris-bundle-credentials \
  --namespace iris \
  --from-file=credentials.json=/path/to/gcs-service-account.json

# Option B: CoreWeave S3
kubectl create secret generic iris-bundle-credentials \
  --namespace iris \
  --from-literal=AWS_ACCESS_KEY_ID=<key> \
  --from-literal=AWS_SECRET_ACCESS_KEY=<secret>
```

For **private registry deployments** (e.g., internal forks), create an image pull secret:

```bash
# Only needed if images are NOT public
kubectl create secret docker-registry iris-image-pull \
  --namespace iris \
  --docker-server=ghcr.io \
  --docker-username=<github-username> \
  --docker-password=$GITHUB_TOKEN
```

### Step 6: Apply Iris Cluster Resources

All operator-managed resources live in `infra/coreweave/k8s/`. The operator applies them
once, and re-applies when config changes.

```bash
# Apply everything at once
kubectl apply -f infra/coreweave/k8s/

# Or step by step:
kubectl apply -f infra/coreweave/k8s/namespace.yaml
kubectl apply -f infra/coreweave/k8s/rbac.yaml
kubectl apply -f infra/coreweave/k8s/configmap.yaml
kubectl apply -f infra/coreweave/k8s/controller-nodepool.yaml
kubectl apply -f infra/coreweave/k8s/controller-deployment.yaml
kubectl apply -f infra/coreweave/k8s/controller-service.yaml
```

### Step 7: Verify Controller is Running

```bash
# Wait for controller NodePool to provision a node (up to 30 min for bare metal)
kubectl get nodepool iris-controller-pool -w

# Once node is ready, wait for controller Deployment
kubectl wait --for=condition=available deployment/iris-controller \
  --namespace iris --timeout=600s

# Check controller health
kubectl logs deployment/iris-controller -n iris --tail=20

# Port-forward for external access (e.g., submitting jobs from laptop)
kubectl port-forward svc/iris-controller-svc 10000:10000 -n iris
```

### Step 8: Submit a Test Job

```bash
# From your laptop, with port-forward running:
IRIS_COREWEAVE_URL=http://localhost:10000 \
  uv run python -c "
from iris.client import IrisClient
client = IrisClient.remote('http://localhost:10000')
job = client.submit(lambda: 42, 'smoke-test')
result = client.wait(job, timeout=2400)
print(f'Job result: {result}')
"
```

---

## 3. CoreWeave Architecture

**Bare metal Kubernetes**: CKS runs Kubernetes directly on bare metal GPU nodes. No
hypervisor. Nodes are stateless, rebooted from clean images, isolated via NVIDIA BlueField
DPUs.

**Node Pools** are the provisioning primitive -- Kubernetes CRDs:

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

**Container runtime**: CoreWeave bare metal nodes use **containerd**, not Docker. The CRI
socket is available at `/run/containerd/containerd.sock`. Task containers are managed via
`crictl`, the standard CRI-compatible CLI. Task images are pulled via `crictl pull` --
since all images on ghcr.io are public, no auth credentials are needed.

**Authentication model**: CKS has two auth endpoints:
- **Managed Auth** (default): Uses `CW-SECRET-...` API tokens. For operators using kubectl from
  their laptop. Token is embedded in the kubeconfig downloaded from Console.
- **Unmanaged Auth**: Uses standard Kubernetes service account tokens. For Pods running
  *inside* the cluster. The controller Pod uses this automatically via in-cluster config.

---

## 4. Design Decisions

### Decision 1: Iris owns all scaling (CoreWeave autoscaler disabled)

Iris's autoscaler tracks individual slices with full identity (`ScalingGroup._slices` is a
`dict[str, SliceState]`). Scale-down picks the *specific longest-idle slice* via
`get_idle_slices()`. CoreWeave's Cluster Autoscaler doesn't give per-slice addressability.

**Resolution**: `autoscaling: false` on all NodePools. **One NodePool = one slice = one bare
metal node.** Iris creates NodePool (scale-up) / deletes NodePool (scale-down).

### Decision 2: Controller runs on CoreWeave as a Kubernetes Deployment

Operator-managed Deployment + Service. Workers discover the controller via K8s Service DNS.
The controller Pod authenticates to the K8s API via in-cluster service account.

### Decision 3: Bootstrap moves into Platform with async state model (all platforms)

Currently, bootstrap is split: `Platform.create_slice()` provisions infrastructure, then the
autoscaler's bootstrap thread calls `WorkerBootstrap.bootstrap_vm()` to SSH in, install
Docker, start the worker, and poll health. This split doesn't work for CoreWeave (no SSH,
worker is a Pod), and it leaks infrastructure concerns into the autoscaler.

**New model**: `Platform.create_slice(config, bootstrap_config)` returns a `SliceHandle`
**immediately** after initiating creation. The platform is responsible for monitoring
bootstrap progress internally (background thread or polling loop). The `SliceHandle.describe()`
method returns the current state via `SliceStatus.state`:

- `CREATING` -- infrastructure provisioning (`gcloud create` on GCP, `kubectl apply NodePool`
  on CoreWeave)
- `BOOTSTRAPPING` -- infrastructure ready, workers starting (SSH bootstrap on GCP, Pod
  starting on CoreWeave)
- `READY` -- workers healthy
- `FAILED` -- something went wrong

`BOOTSTRAPPING` is added to the `CloudSliceState` enum.

Each platform implements bootstrap differently, but the interface is the same: the caller
passes a `BootstrapConfig` (image, ports, env vars) and gets back a `SliceHandle` that
progresses through states. The autoscaler holds the handle and monitors state transitions
via `describe()` -- it does NOT drive bootstrap.

If a single worker fails during bootstrap, the platform fails the entire slice and cleans up
its own resources (deletes TPU on GCP, deletes Pod and NodePool on CoreWeave). The
`SliceHandle.describe()` returns `FAILED` state, and the autoscaler observes this, calls
`handle.terminate()` as a safety net, and records failure.

`WorkerBootstrap` as a separate class is removed. The GCP bootstrap script
(`WORKER_BOOTSTRAP_SCRIPT` in `bootstrap.py`) moves into `GcpPlatform.create_slice()`.

```
GCP create_slice(config, bootstrap):         CoreWeave create_slice(config, bootstrap):
1. gcloud compute tpus create                1. kubectl apply NodePool CRD
2. Return SliceHandle (CREATING)             2. Return SliceHandle (CREATING)
3. [internal] SSH wait_for_connection()      3. [internal] Poll until node ready
4. [internal] SSH run bootstrap script:      4. [internal] kubectl apply worker Pod:
   - install docker                               image: bootstrap.docker_image
   - pull images                                  ports: bootstrap.worker_port
   - docker run worker                            env: bootstrap.env_vars
   - poll /health                                 mounts: config, cache, cri socket
5. [internal] Mark READY                          readinessProbe: /health
                                             5. [internal] Wait for Pod readiness
                                             6. [internal] Mark READY
```

On failure at any step, the platform cleans up its resources and marks the handle as `FAILED`.
The autoscaler observes the `FAILED` state and calls `handle.terminate()` as a safety net.

### Decision 4: Task containers via `ContainerdRuntime` using `crictl`

CoreWeave bare metal nodes run containerd, not Docker. The existing `ContainerRuntime`
protocol (`runtime/types.py`) is transport-agnostic -- it defines `create_container()`,
`status()`, `logs()`, `stats()`, `cleanup()`. A new `ContainerdRuntime` implements this
protocol using `crictl` (the CRI CLI) instead of `docker`.

The worker Pod mounts the host's containerd socket at `/run/containerd/containerd.sock`
and uses `crictl` commands that mirror the existing Docker commands:

| Docker command | crictl equivalent |
|---|---|
| `docker create` | `crictl create <pod-sandbox-id> container.json pod.json` |
| `docker start <id>` | `crictl start <id>` |
| `docker inspect <id>` | `crictl inspect <id>` |
| `docker logs <id>` | `crictl logs <id>` |
| `docker stats <id>` | `crictl stats <id>` |
| `docker rm -f <id>` | `crictl rm -f <id>` |

`crictl` requires a pod sandbox. The worker creates one sandbox per task, then creates
the task container inside it. The sandbox provides network namespace isolation.

Task image pulling is simple: `crictl pull ghcr.io/marin-community/iris-task:latest` works
without any auth credentials because all images on ghcr.io are public.

**Single worker image**: There is one worker Dockerfile, not separate GCP/CoreWeave
variants. It installs both `docker-ce-cli` and `crictl`. The `--runtime=docker|containerd`
flag on worker startup selects the backend. Which socket to mount is a deployment concern:
- GCP: the bootstrap script runs `docker run ... -v /var/run/docker.sock:/var/run/docker.sock`
- CoreWeave: the worker Pod spec mounts `/run/containerd/containerd.sock`

The worker image doesn't care which runtime is available -- it just needs the CLI.

### Decision 5: Rename `SshConnection` to `RemoteExec`

The `SshConnection` protocol in `ssh.py` is transport-agnostic: it defines `run(command)`
and `run_streaming(command)`. Renaming it to `RemoteExec` clarifies intent. The file is
renamed `remote_exec.py`.

Existing implementations (renamed, behavior unchanged):
- `GcloudSshConnection` -> `GcloudRemoteExec`
- `GceSshConnection` -> `GceRemoteExec`
- `DirectSshConnection` -> `DirectSshRemoteExec`

The shared base class `SshVmBase` is renamed to `RemoteExecWorkerBase`.

No new `RemoteExec` implementation is added for CoreWeave. The CoreWeave platform does not
use `RemoteExec` at all -- bootstrap is handled by the Pod spec, and ongoing communication
is via heartbeat RPC. If operators need to debug a worker Pod, they use `kubectl exec`
directly from their terminal.

### Decision 6: Rename `VmHandle` to `RemoteWorkerHandle`

On CoreWeave, the handle represents a **worker Pod**, not a VM. On GCP it represents a TPU
VM. The name `VmHandle` is misleading for non-VM platforms. Renaming to `RemoteWorkerHandle`
reflects what it actually represents: a handle to a remote worker process.

All platform-layer types are renamed:
- `VmHandle` -> `RemoteWorkerHandle`
- `CoreweaveVmHandle` -> `CoreweaveWorkerHandle`
- `GcpVmHandle` -> `GcpWorkerHandle`
- `StandaloneVmHandle` -> `StandaloneWorkerHandle`
- `SshVmBase` -> `RemoteExecWorkerBase`
- `CloudVmState` -> `CloudWorkerState`
- `VmStatus` -> `WorkerStatus`
- `vm_id` property -> `worker_id` on the protocol (GCP implementation keeps `vm_id` as
  backing field)
- `SliceStatus.vms` field -> `SliceStatus.workers`
- `TrackedVm` in autoscaler -> `TrackedWorker`

`bootstrap()` and `wait_for_connection()` are **removed** from the `RemoteWorkerHandle`
protocol. These are SSH-specific implementation details that belong on `RemoteExecWorkerBase`
only. `CoreweaveWorkerHandle` does not implement them.

The `RemoteWorkerHandle` protocol becomes:

```python
class RemoteWorkerHandle(Protocol):
    """Handle to a single worker within a slice.

    Represents a remote worker process: a TPU VM on GCP, a Pod on CoreWeave,
    a thread on LocalPlatform. Provides infrastructure-level operations.
    Lifecycle state machines, retries, and health checking are the
    orchestration layer's responsibility.

    No terminate -- slices are the atomic unit. Individual slice members
    cannot be terminated independently.

    Thread safety: implementations must be safe for concurrent run_command() calls.
    """

    @property
    def worker_id(self) -> str: ...

    @property
    def internal_address(self) -> str:
        """Internal/private IP address of this worker.

        This is the primary address used for all intra-cluster communication.
        """
        ...

    @property
    def external_address(self) -> str | None:
        """External/public IP address, if available. Returns None when no external IP."""
        ...

    def status(self) -> WorkerStatus:
        """Cloud-level worker status."""
        ...

    def run_command(
        self,
        command: str,
        timeout: Duration | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        """Run a command on the worker. Optionally stream output lines."""
        ...

    def reboot(self) -> None:
        """Reboot the worker."""
        ...
```

Methods that remain on `RemoteExecWorkerBase` only (not on the protocol):

```python
class RemoteExecWorkerBase:
    """Base class for SSH-based worker handles (GCP, Manual platforms)."""

    def wait_for_connection(
        self,
        timeout: Duration,
        poll_interval: Duration = Duration.from_seconds(5),
    ) -> bool:
        """Poll until SSH connection is available. Returns False on timeout."""
        ...

    def bootstrap(self, script: str) -> None:
        """Run a bootstrap script on the worker via SSH.

        Equivalent to run_command(f'bash -c {script}') with streaming.
        Raises on non-zero exit.
        """
        ...
```

### Decision 7: Images on ghcr.io (public, no auth needed)

All Iris images (`iris-controller`, `iris-worker`, `iris-task`) are pushed to
`ghcr.io/marin-community/` and are **public**. This means:

- `docker pull` and `crictl pull` work without auth credentials
- No `imagePullSecrets` needed in Pod specs for standard deployments
- `imagePullSecrets` can be added for private registry deployments (e.g., internal forks)
- The task image pulling story on CoreWeave is simple:
  `crictl pull ghcr.io/marin-community/iris-task:latest` just works

---

## 5. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  CoreWeave CKS Cluster                                           │
│                                                                  │
│  ┌─────────────────────────────────┐                             │
│  │  Controller Deployment          │  <-- operator applies       │
│  │  (always-on, controller pool)   │      (Deployment + Service) │
│  │                                 │                             │
│  │  Pod: iris-controller           │                             │
│  │    ghcr.io/.../iris-controller  │                             │
│  │    port 10000                   │                             │
│  │    in-cluster K8s auth          │ <-- ServiceAccount          │
│  │    /etc/iris/config.yaml        │ <-- ConfigMap               │
│  └────────┬────────────────────────┘                             │
│           │                                                      │
│  Service: iris-controller-svc ──────────────────────────┐        │
│  (ClusterIP, port 10000)                                │        │
│           │                                             │        │
│  ┌────────▼────────────────────────┐  ┌─────────────────▼──────┐ │
│  │  Worker NodePool: iris-slice-A  │  │ Worker NodePool: ...   │ │
│  │  (created by platform.py)       │  │ (created by platform)  │ │
│  │  instanceType: gd-8xh100ib-i128│  │                        │ │
│  │  targetNodes: 1                 │  │                        │ │
│  │                                 │  │                        │ │
│  │  Pod: iris-worker-slice-A       │  │  Pod: iris-worker-...  │ │
│  │  (created by platform.py)       │  │  (created by platform) │ │
│  │    ghcr.io/.../iris-worker      │  │                        │ │
│  │    port 10001                   │  │                        │ │
│  │    nvidia.com/gpu: 8            │  │                        │ │
│  │    containerd socket mounted    │  │                        │ │
│  │    task containers via crictl   │  │                        │ │
│  └─────────────────────────────────┘  └────────────────────────┘ │
│                                                                  │
│  Operator-managed resources:                                     │
│    Namespace, ConfigMap, Secrets, RBAC,                           │
│    Controller NodePool, Deployment, Service                      │
│                                                                  │
│  Platform.py-managed resources:                                  │
│    Worker NodePools, Worker Pods (dynamic, per-slice)            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. Control Flow: End-to-End Lifecycle

### Phase 0: Cluster Bootstrap (operator, one-time)

See [Operator Guide](#2-operator-guide-setting-up-iris-on-coreweave) above.

The operator applies these Kubernetes resources:

**`infra/coreweave/k8s/rbac.yaml`** -- ServiceAccount + ClusterRole for the controller:

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: iris-controller
  namespace: iris
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: iris-controller
rules:
  - apiGroups: ["compute.coreweave.com"]
    resources: ["nodepools"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: [""]
    resources: ["pods", "pods/exec", "pods/log"]
    verbs: ["get", "list", "watch", "create", "delete"]
  - apiGroups: [""]
    resources: ["nodes"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: iris-controller
subjects:
  - kind: ServiceAccount
    name: iris-controller
    namespace: iris
roleRef:
  kind: ClusterRole
  name: iris-controller
  apiGroup: rbac.authorization.k8s.io
```

**`infra/coreweave/k8s/configmap.yaml`** -- Iris cluster config:

```yaml
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
        docker_image: ghcr.io/marin-community/iris-worker:latest
        task_image: ghcr.io/marin-community/iris-task:latest
        worker_port: 10001
        runtime: containerd
      timeouts:
        boot_timeout: { milliseconds: 1800000 }
        init_timeout: { milliseconds: 1800000 }
      autoscaler:
        evaluation_interval: { milliseconds: 10000 }
        scale_up_delay: { milliseconds: 60000 }
        scale_down_delay: { milliseconds: 300000 }
        startup_grace_period: { milliseconds: 2400000 }
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
```

**`infra/coreweave/k8s/controller-nodepool.yaml`**:

```yaml
apiVersion: compute.coreweave.com/v1alpha1
kind: NodePool
metadata:
  name: iris-controller-pool
spec:
  computeClass: default
  instanceType: <cpu-instance>   # small CPU-only instance
  targetNodes: 1
  autoscaling: false
  nodeLabels:
    iris-role: controller
```

**`infra/coreweave/k8s/controller-deployment.yaml`**:

```yaml
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
      serviceAccountName: iris-controller   # uses in-cluster auth
      nodeSelector:
        iris-role: controller
      containers:
        - name: controller
          image: ghcr.io/marin-community/iris-controller:latest
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
          readinessProbe:
            httpGet:
              path: /health
              port: 10000
            initialDelaySeconds: 10
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 10000
            initialDelaySeconds: 30
            periodSeconds: 30
      volumes:
        - name: config
          configMap:
            name: iris-cluster-config
        - name: bundle-credentials
          secret:
            secretName: iris-bundle-credentials
---
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

The `/health` endpoint works with `httpGet` because the Iris server uses the Connect
protocol, which serves JSON over HTTP on the same port as gRPC. The health endpoint
responds to plain HTTP GET requests.

### Phase 1: Controller Startup (automatic after operator applies manifests)

The controller Pod starts and runs:
```
python -m iris.cluster.controller.main serve --config /etc/iris/config.yaml
```

Startup sequence (same as GCP, from `controller/main.py`):
1. `load_config()` reads `/etc/iris/config.yaml` (from ConfigMap)
2. `create_platform()` returns `CoreweavePlatform` (dispatched by `factory.py`)
   - Uses **in-cluster Kubernetes auth** (ServiceAccount token at
     `/var/run/secrets/kubernetes.io/serviceaccount/token`)
   - No kubeconfig file needed -- `kubectl` auto-detects in-cluster config
3. `create_autoscaler()` creates `ScalingGroup` per config entry (each group holds a
   reference to the platform for `create_slice()` calls)
4. `autoscaler.reconcile()` -- discovers existing worker NodePools/Pods by label
5. `Controller.start()` -- scheduling loop, heartbeat loop, autoscaler loop, HTTP server

### Phase 2: Scale-Up (`platform.py` creates worker resources asynchronously)

When demand arrives (job submitted), the autoscaler evaluates and calls into `platform.py`.
The key difference from the old model: `create_slice()` returns immediately with a handle
in `CREATING` state. The platform monitors progress internally.

1. **Autoscaler decides** to scale up group `h100_8x`
2. **`CoreweavePlatform.create_slice(config, bootstrap_config)`** returns immediately:
   ```
   a. Generate slice ID: iris-h100-8x-{timestamp_ms}
   b. kubectl apply NodePool CRD:
        apiVersion: compute.coreweave.com/v1alpha1
        kind: NodePool
        metadata:
          name: iris-h100-8x-1738000000000
          labels:
            iris-managed: "true"
            iris-scale-group: h100_8x
            iris-slice-id: iris-h100-8x-1738000000000
        spec:
          instanceType: gd-8xh100ib-i128
          targetNodes: 1
          autoscaling: false
          nodeLabels:
            iris-managed: "true"
            iris-scale-group: h100_8x
            iris-slice-id: iris-h100-8x-1738000000000
   c. Return CoreweaveSliceHandle (state = CREATING)
   ```

3. **Platform monitors internally** (background thread):
   ```
   d. Poll NodePool status until readyNodes >= 1
      (SliceHandle.describe() returns CREATING during this phase)
   e. kubectl apply worker Pod:
        name: iris-worker-iris-h100-8x-1738000000000
        namespace: iris
        labels:
          iris-managed: "true"
          iris-slice-id: iris-h100-8x-1738000000000
        spec:
          nodeSelector:
            iris-slice-id: iris-h100-8x-1738000000000
          containers:
            - name: iris-worker
              image: {bootstrap_config.docker_image}
              command:
                - .venv/bin/python
                - -m
                - iris.cluster.worker.main
                - serve
                - --host=0.0.0.0
                - --port={bootstrap_config.worker_port}
                - --cache-dir=/var/cache/iris
                - --config=/etc/iris/config.yaml
                - --runtime=containerd
              resources:
                limits:
                  nvidia.com/gpu: 8
                  rdma/ib: 1
              volumeMounts:
                - name: config
                  mountPath: /etc/iris
                  readOnly: true
                - name: cache
                  mountPath: /var/cache/iris
                - name: containerd-socket
                  mountPath: /run/containerd/containerd.sock
              env:
                - name: IRIS_VM_ADDRESS
                  valueFrom:
                    fieldRef:
                      fieldPath: status.podIP
                - name: CONTAINER_RUNTIME_ENDPOINT
                  value: unix:///run/containerd/containerd.sock
              readinessProbe:
                httpGet:
                  path: /health
                  port: {bootstrap_config.worker_port}
                initialDelaySeconds: 5
                periodSeconds: 5
          volumes:
            - name: config
              configMap:
                name: iris-cluster-config
            - name: cache
              hostPath:
                path: /var/cache/iris
                type: DirectoryOrCreate
            - name: containerd-socket
              hostPath:
                path: /run/containerd/containerd.sock
                type: Socket
      (SliceHandle.describe() returns BOOTSTRAPPING during this phase)
   f. Wait for Pod readiness probe to pass
   g. Mark handle state READY
      (SliceHandle.describe() returns READY)
   ```

4. **Autoscaler monitors state transitions** via its reconcile cycle:
   - Calls `handle.describe()` to observe state
   - When state reaches `READY`: calls `mark_slice_ready()`
   - When state reaches `FAILED`: calls `mark_slice_failed()`, then
     `handle.terminate()` for cleanup, then records failure

5. **On failure**: If any step fails internally (NodePool quota, Pod never ready, etc.):
   - Platform cleans up its own resources (deletes Pod, deletes NodePool)
   - Platform marks handle state as `FAILED`
   - Autoscaler observes `FAILED`, calls `handle.terminate()` as safety net, records failure

6. **No separate bootstrap step.** The worker Pod starts as its entrypoint, discovers the
   controller via K8s Service DNS, and registers. The `readinessProbe` on `/health` serves
   the same purpose as the GCP bootstrap health poll.

### Phase 3: Worker Registration

Worker Pod entrypoint runs:
```
python -m iris.cluster.worker.main serve --config /etc/iris/config.yaml --runtime=containerd
```

1. `load_config()` reads config from ConfigMap mount
2. `create_platform()` returns `CoreweavePlatform` (in-cluster auth)
3. `platform.discover_controller()` returns `iris-controller-svc.iris.svc.cluster.local:10000`
4. Worker creates `ContainerdRuntime` (instead of `DockerRuntime`)
5. `Worker._register()` sends RPC to controller via Service DNS
6. Controller accepts, assigns `worker_id`
7. Worker enters heartbeat serve loop

### Phase 4: Task Execution

Task execution follows the standard Iris flow with one difference: the `ContainerdRuntime`
replaces `DockerRuntime`.

1. Controller sends task via heartbeat RPC
2. Worker downloads code bundle from GCS
3. Worker calls `ContainerdRuntime.create_container(config)`:
   - Pulls task image: `crictl pull ghcr.io/marin-community/iris-task:latest` (no auth needed)
   - Creates a pod sandbox via `crictl runp pod-config.json`
   - Creates the task container via `crictl create <sandbox-id> container-config.json pod-config.json`
4. Worker calls `handle.build()` -- runs setup commands (uv sync) in a build container
5. Worker calls `handle.run()` -- starts main command container via `crictl start <id>`
6. Worker monitors via `crictl inspect <id>`, `crictl logs <id>`, `crictl stats <id>`
7. Worker cleans up via `crictl rm -f <id>` and `crictl stopp <sandbox-id>`

### Phase 5: Scale-Down (`platform.py` destroys worker resources)

1. Autoscaler picks the specific longest-idle slice
2. `handle.terminate()` calls `platform.py`:
   ```
   kubectl delete pod iris-worker-iris-h100-8x-1738000000000 -n iris
   kubectl delete nodepool iris-h100-8x-1738000000000
   ```
3. Node deprovisioned -- deterministic, exactly the chosen slice

---

## 7. Secrets and Credentials Summary

### What the Operator Creates

| Resource | Purpose | How to Obtain |
|----------|---------|---------------|
| CoreWeave API token | Terraform + kubeconfig | Console > Tokens > Create Token -> `CW-SECRET-...` |
| Kubeconfig | Operator's kubectl access | Console > Tokens > Download Kubeconfig |
| `iris-bundle-credentials` Secret | GCS or S3 access for job bundles | GCS SA key JSON or CoreWeave S3 credentials |
| `iris-controller` ServiceAccount | In-cluster K8s API auth for controller | `kubectl apply` RBAC manifests |

For private registry deployments only:
| `iris-image-pull` Secret | Pull from private ghcr.io | `kubectl create secret docker-registry` with GitHub PAT |

### What Platform.py Uses at Runtime

| Credential | How Obtained | Used For |
|------------|-------------|----------|
| In-cluster ServiceAccount token | Auto-mounted by Kubernetes at `/var/run/secrets/...` | `kubectl` calls to create/delete NodePools + Pods |
| `iris-cluster-config` ConfigMap | Mounted into Pods at `/etc/iris/config.yaml` | Config for controller and workers |
| Bundle credentials | Mounted from Secret into Pod at `/etc/iris/credentials/` | Downloading/uploading job bundles |

**No kubeconfig file in platform.py.** The controller runs inside the cluster and uses
Kubernetes in-cluster authentication (service account token). `kubectl` auto-detects this.
The `kubeconfig_path` field in `CoreweavePlatformConfig` is only needed if the controller
runs *outside* the cluster (e.g., during local development).

**No image pull credentials at runtime.** All images on ghcr.io are public.
`crictl pull` on worker nodes and `docker pull` on GCP VMs work without auth.

---

## 8. Code Changes: File-by-File

### `config.proto` -- Extend proto definitions

```protobuf
message CoreweavePlatformConfig {
  string region = 1;
  string namespace = 2;              // default: "iris"
  string kubeconfig_path = 3;        // optional, uses in-cluster auth if empty
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

message BootstrapConfig {
  string docker_image = 1;           // e.g. "ghcr.io/marin-community/iris-worker:latest"
  string task_image = 2;             // e.g. "ghcr.io/marin-community/iris-task:latest"
  int32 worker_port = 3;             // e.g. 10001
  map<string, string> env_vars = 4;  // additional env vars for the worker
  string cache_dir = 5;              // e.g. "/var/cache/iris"
  string runtime = 6;                // "docker" or "containerd"
}

// Add to ControllerVmConfig oneof:
message ControllerVmConfig {
  ...
  oneof controller {
    GcpControllerConfig gcp = 1;
    ManualControllerConfig manual = 2;
    LocalControllerConfig local = 3;
    CoreweaveControllerConfig coreweave = 4;
  }
}
```

### `base.py` -- Rename types and update `RemoteWorkerHandle` protocol

The `CloudSliceState` enum gains a `BOOTSTRAPPING` value. All "Vm" types are renamed:

```python
class CloudSliceState(StrEnum):
    CREATING = "CREATING"
    BOOTSTRAPPING = "BOOTSTRAPPING"   # new
    READY = "READY"
    REPAIRING = "REPAIRING"
    DELETING = "DELETING"
    UNKNOWN = "UNKNOWN"

class CloudWorkerState(StrEnum):      # was CloudVmState
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    TERMINATED = "TERMINATED"
    UNKNOWN = "UNKNOWN"

@dataclass
class WorkerStatus:                    # was VmStatus
    state: CloudWorkerState

@dataclass
class SliceStatus:
    state: CloudSliceState
    worker_count: int                  # was vm_count
    workers: list[RemoteWorkerHandle] = field(default_factory=list)  # was vms
```

### `Platform.create_slice()` -- Returns immediately, async state monitoring

The `Platform.create_slice()` signature changes to accept bootstrap configuration, and
the method returns a `SliceHandle` immediately:

```python
class Platform(Protocol):
    def create_slice(
        self,
        config: config_pb2.SliceConfig,
        bootstrap_config: config_pb2.BootstrapConfig,
    ) -> SliceHandle:
        """Initiate creation of a slice and return a handle for state monitoring.

        Returns immediately with a SliceHandle in CREATING state. The platform
        monitors bootstrap progress internally. The caller polls
        handle.describe().state to observe transitions:

            CREATING -> BOOTSTRAPPING -> READY | FAILED

        On failure, the platform cleans up its own resources (TPU, NodePool, Pod)
        and marks the handle as FAILED. The caller should call handle.terminate()
        as a safety net after observing FAILED.

        Raises QuotaExhaustedError synchronously if quota is exhausted at the
        point of initial resource request. Other failures are reported via the
        FAILED state on the handle.
        """
        ...
```

**GCP implementation**: `GcpPlatform.create_slice()` initiates TPU creation, returns
the handle immediately, then internally monitors: SSH into each VM, run the bootstrap
script (install Docker, pull images, start worker container, poll health). If any VM fails
bootstrap, the platform deletes the TPU and marks the handle as FAILED. The existing
`WORKER_BOOTSTRAP_SCRIPT` template moves from `bootstrap.py` into `gcp.py`.

**CoreWeave implementation**: `CoreweavePlatform.create_slice()` applies the NodePool CRD,
returns the handle immediately, then internally monitors: poll NodePool status, apply
worker Pod, wait for Pod readiness. If the Pod never becomes ready, the platform deletes
the Pod and NodePool, and marks the handle as FAILED.

**LocalPlatform**: `LocalPlatform.create_slice(config, bootstrap_config)` accepts the
`bootstrap_config` parameter for protocol compliance and ignores it. LocalPlatform spawns
workers directly (ProcessRuntime), it doesn't need bootstrap. The handle is returned in
READY state immediately (synchronous, no async monitoring needed).

**Autoscaler simplification**:
- `WorkerBootstrap` class is removed
- `bootstrap_slice_vms()` function is removed from `autoscaler.py`
- `_bootstrap_slice()` and `_bootstrap_slice_safe()` are removed
- The autoscaler's `_do_scale_up()` calls `group.scale_up()` which calls
  `platform.create_slice(config, bootstrap_config)` -- this returns immediately
- `complete_scale_up()` is called right after `create_slice()` returns (handle tracked
  immediately in CREATING state)
- The autoscaler's existing reconcile cycle polls `handle.describe()` to observe
  state transitions
- When state reaches `READY`: calls `mark_slice_ready()`
- When state reaches `FAILED`: calls `mark_slice_failed()`, then `handle.terminate()`
  for cleanup, then records failure
- No more background bootstrap threads in the autoscaler

### `ssh.py` -> `remote_exec.py` -- Rename protocol and implementations

Pure rename, no new implementations. The protocol and all existing classes are renamed:

```python
# remote_exec.py

class RemoteExec(Protocol):
    """Execute commands on a remote host. Carries location metadata.

    Implementations: GcloudRemoteExec (TPU VMs via gcloud), GceRemoteExec
    (GCE VMs via gcloud), DirectSshRemoteExec (raw SSH).

    Used by GCP and Manual platforms for bootstrap (now inside create_slice())
    and health checks. CoreWeave does not use RemoteExec -- its bootstrap is
    handled by the Pod spec and ongoing communication is via heartbeat RPC.
    """

    def run(self, command: str, timeout: Duration = Duration.from_seconds(30)) -> subprocess.CompletedProcess:
        ...

    def run_streaming(self, command: str) -> Any:
        ...

    @property
    def address(self) -> str:
        """Target address (IP, hostname)."""
        ...

    @property
    def zone(self) -> str:
        """Zone/location, or empty string if not applicable."""
        ...
```

All utility functions (`wait_for_connection`, `run_streaming_with_retry`, etc.) are
renamed accordingly but behavior is unchanged.

### `_vm_base.py` -> `_worker_base.py` -- Rename `SshVmBase` to `RemoteExecWorkerBase`

All references updated: `_ssh` field becomes `_remote_exec`, type `SshConnection` becomes
`RemoteExec`. `bootstrap()` and `wait_for_connection()` live here, not on the protocol.
No behavioral change.

### `runtime/containerd.py` -- New `ContainerdRuntime`

Implements the `ContainerRuntime` protocol using `crictl`:

```python
class ContainerdRuntime:
    """Container runtime using crictl (CRI) for containerd-based hosts.

    Used on CoreWeave bare metal nodes where Docker is not available.
    The worker Pod mounts the host containerd socket and uses crictl
    to create/manage task containers. All images on ghcr.io are public,
    so crictl pull does not need auth credentials.
    """

    def __init__(self, socket_path: str = "/run/containerd/containerd.sock"):
        self._endpoint = f"unix://{socket_path}"

    def create_container(self, config: ContainerConfig) -> ContainerdContainerHandle:
        # 1. crictl pull {config.image} (no auth needed -- public images)
        # 2. Write pod sandbox config JSON (network namespace, labels)
        # 3. crictl runp pod-config.json -> sandbox_id
        # 4. Write container config JSON (image, command, mounts, env, resources)
        # 5. crictl create <sandbox_id> container-config.json pod-config.json -> container_id
        # 6. Return ContainerdContainerHandle(sandbox_id, container_id)
        ...


class ContainerdContainerHandle:
    """ContainerHandle backed by crictl commands.

    Maps the ContainerHandle protocol to crictl operations. Each task gets
    its own pod sandbox for network isolation.
    """

    def build(self) -> list[LogLine]:
        # crictl create + start a temporary build container in the same sandbox
        # Run setup_commands, capture logs
        # crictl rm build container when done
        ...

    def run(self) -> None:
        # crictl start <container_id>
        ...

    def status(self) -> ContainerStatus:
        # crictl inspect <container_id> --output json
        # Parse .status.state (CONTAINER_RUNNING, CONTAINER_EXITED, etc.)
        ...

    def logs(self, since: Timestamp | None = None) -> list[LogLine]:
        # crictl logs [--since <rfc3339>] <container_id>
        ...

    def stats(self) -> ContainerStats:
        # crictl stats <container_id> --output json
        ...

    def cleanup(self) -> None:
        # crictl rm -f <container_id>
        # crictl stopp <sandbox_id>
        # crictl rmp <sandbox_id>
        ...
```

### `coreweave.py` -- Implement `CoreweavePlatform`

Replace stub with full implementation. All `kubectl` commands use in-cluster auth by default
(no `--kubeconfig` flag needed). If `kubeconfig_path` is set, pass `--kubeconfig`.

- `create_slice(config, bootstrap_config)`:
  1. `kubectl apply` NodePool CRD
  2. Return `CoreweaveSliceHandle` immediately (state = CREATING)
  3. Spawn internal monitoring thread:
     a. Poll NodePool status until `readyNodes >= 1` (CREATING phase)
     b. `kubectl apply` worker Pod (image, ports, env from `bootstrap_config`)
        (BOOTSTRAPPING phase)
     c. Wait for Pod readiness
     d. Mark handle READY
     e. On failure at any step: delete Pod + NodePool, mark handle FAILED
- `list_slices()`: `kubectl get nodepools -l iris-scale-group=X -o json`
- `list_all_slices()`: `kubectl get nodepools -l iris-managed=true -o json`
- `describe()` on handle: Query NodePool status + Pod status, return current `CloudSliceState`
  and `RemoteWorkerHandle` per Pod
- `terminate()` on handle: Delete Pod, then delete NodePool
- `discover_controller()`: `{service_name}.{namespace}.svc.cluster.local:{port}`
- `tunnel()`: `kubectl port-forward`

**Quota error detection**: When `kubectl apply` for a NodePool fails, or the NodePool
status shows a condition with `type: Failed` and a message containing "quota" or
"insufficient capacity", `create_slice()` raises `QuotaExhaustedError` synchronously
(before returning the handle). The NodePool CRD status conditions are also checked during
the internal readiness polling loop, causing a transition to FAILED:

```python
def _check_nodepool_status(self, name: str) -> CloudSliceState:
    result = subprocess.run(
        ["kubectl", "get", "nodepool", name, "-o", "json"],
        capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    conditions = data.get("status", {}).get("conditions", [])
    for cond in conditions:
        if cond.get("type") == "Failed" and cond.get("status") == "True":
            msg = cond.get("message", "").lower()
            if "quota" in msg or "insufficient" in msg or "capacity" in msg:
                raise QuotaExhaustedError(cond["message"])
            raise PlatformError(cond["message"])
    ready_nodes = data.get("status", {}).get("readyNodes", 0)
    if ready_nodes >= 1:
        return CloudSliceState.READY
    return CloudSliceState.CREATING
```

**Platform cleanup on failure**: Each platform is responsible for cleaning up its own
resources if bootstrap fails:
- GCP: if SSH bootstrap fails on any VM, `GcpPlatform` deletes the TPU (inside its
  internal monitoring thread)
- CoreWeave: if Pod never becomes ready, `CoreweavePlatform` deletes the Pod and NodePool
  (inside its internal monitoring thread)
- The `SliceHandle.describe()` returns `FAILED` state, and the autoscaler observes this
  and calls `terminate()` as a safety net

### `factory.py` -- Wire up CoreweavePlatform

```python
def create_platform(config: config_pb2.IrisClusterConfig) -> Platform:
    which = config.platform.WhichOneof("platform")
    if which == "coreweave":
        cw = config.platform.coreweave
        return CoreweavePlatform(
            namespace=cw.namespace or "iris",
            region=cw.region,
            label_prefix=config.platform.label_prefix or "iris",
            kubeconfig_path=cw.kubeconfig_path or None,
        )
    ...
```

### `worker/main.py` -- Accept `--runtime` flag

```python
@click.option("--runtime", type=click.Choice(["docker", "containerd"]), default="docker")
def serve(host, port, cache_dir, config, runtime):
    if runtime == "containerd":
        container_runtime = ContainerdRuntime()
    else:
        container_runtime = DockerRuntime()
    worker = Worker(..., runtime=container_runtime)
```

### Dockerfiles -- Single worker image with both runtimes

The existing worker Dockerfile is updated to install both `docker-ce-cli` and `crictl`.
No separate CoreWeave variant. The `--runtime` flag on worker startup selects the backend.

```dockerfile
# Added to existing Dockerfile.worker:
# Install crictl for containerd-based platforms (CoreWeave)
RUN curl -fsSL https://github.com/kubernetes-sigs/cri-tools/releases/download/v1.32.0/crictl-v1.32.0-linux-amd64.tar.gz \
    | tar -xz -C /usr/local/bin
```

The controller image needs a CoreWeave variant (`Dockerfile.controller.coreweave`) because
the GCP controller uses `gcloud` for SSH tunneling and VM management, while the CoreWeave
controller uses `kubectl`. The controller is operator-managed infrastructure, so having
separate images is acceptable:
- `Dockerfile.controller.coreweave`: `python:3.11-slim` + `kubectl` (no `gcloud`)
- `Dockerfile.controller` (existing): `python:3.11-slim` + `gcloud`

Images are built via `iris build` CLI (`lib/iris/src/iris/cli/build.py`). The CLI currently
supports GCP Artifact Registry push via `--region` / `--project`. For CoreWeave, the `push`
command is extended with a `--registry ghcr` option that pushes to `ghcr.io/marin-community/`
(or a custom org for private forks):

```bash
# Build
uv run iris build worker-image --tag iris-worker:latest
uv run iris build controller-image --tag iris-controller-coreweave:latest \
    --dockerfile lib/iris/Dockerfile.controller.coreweave
uv run iris build task-image --tag iris-task:latest

# Push to ghcr.io (new)
uv run iris build push iris-worker:latest --registry ghcr --image-name iris-worker
uv run iris build push iris-controller-coreweave:latest --registry ghcr --image-name iris-controller-coreweave
uv run iris build push iris-task:latest --registry ghcr --image-name iris-task

# Push to GCP Artifact Registry (existing)
uv run iris build push iris-worker:latest --region us-central1 --project hai-gcp-models
```

### New manifests: `infra/coreweave/k8s/`

- `namespace.yaml`
- `rbac.yaml` (ServiceAccount + ClusterRole + ClusterRoleBinding)
- `configmap.yaml` (template -- operator fills in scale groups)
- `controller-nodepool.yaml`
- `controller-deployment.yaml` (Deployment + Service)

---

## 9. Testing Plan

### Unit Tests (run without CoreWeave access)

Tests use mocked `subprocess.run` to validate command construction and response parsing.

1. **`CoreweavePlatform.create_slice()` flow**: Mock kubectl calls to verify:
   - NodePool CRD YAML contains correct labels, instanceType, autoscaling: false
   - Worker Pod YAML contains correct image, ports, env vars, volume mounts from bootstrap config
   - Handle returns `CREATING` immediately, transitions to `BOOTSTRAPPING` then `READY`
   - `QuotaExhaustedError` raised when NodePool status has Failed condition with quota message
   - On failure: platform cleans up resources (deletes Pod + NodePool) and marks handle FAILED

2. **`CoreweavePlatform.list_slices()`**: Mock `kubectl get nodepools -o json` response,
   verify correct `CoreweaveSliceHandle` objects returned with correct labels and slice IDs.

3. **`CoreweavePlatform.discover_controller()`**: Verify DNS name construction from
   service_name, namespace, and port.

4. **`ContainerdRuntime` command construction**: Mock `crictl` subprocess calls to verify:
   - Pod sandbox config JSON is correct
   - Container config JSON maps `ContainerConfig` fields correctly
   - `crictl pull` is called without auth credentials (public images)
   - `status()` parses crictl inspect output correctly
   - `logs()` handles `--since` flag correctly
   - `cleanup()` removes container and sandbox

5. **Autoscaler state monitoring**: Verify the autoscaler correctly:
   - Calls `complete_scale_up()` immediately after `create_slice()` returns
   - Polls `handle.describe()` for state transitions
   - Calls `mark_slice_ready()` when state reaches `READY`
   - Calls `mark_slice_failed()` + `handle.terminate()` when state reaches `FAILED`

### E2E Test (requires CoreWeave cluster)

Submit a simple job to an Iris cluster running entirely on CoreWeave, verify it succeeds.

```bash
# 1. Operator deploys (one-time)
kubectl apply -f tests/e2e/coreweave-bootstrap/

# 2. Wait for controller
kubectl wait --for=condition=available deployment/iris-controller -n iris-test --timeout=600s

# 3. Port-forward
kubectl port-forward svc/iris-controller-svc 10000:10000 -n iris-test &

# 4. Run test
IRIS_COREWEAVE_URL=http://localhost:10000 \
  uv run pytest tests/e2e/test_coreweave_smoke.py -v -m coreweave
```

Validates: controller startup, create_slice with NodePool + Pod, worker registration,
task execution via ContainerdRuntime, scale-down.

### CI Integration for ContainerdRuntime

The `ContainerdRuntime` can be tested in CI without CoreWeave by running a containerd
daemon in the CI environment. This validates the crictl command construction and response
parsing against a real containerd instance.

---

## 10. Implementation Order

The implementation is split into two spirals. **Spiral 1** refactors the platform layer on
existing platforms (GCP + Manual + Local) and is independently shippable and testable before
any CoreWeave code is written. **Spiral 2** builds CoreWeave support on the clean interface
from Spiral 1.

### Spiral 1: Platform refactor (GCP + Manual + Local)

This spiral is independently shippable and testable on existing platforms BEFORE writing
any CoreWeave code.

1. **Rename `SshConnection` -> `RemoteExec`** (mechanical): rename file `ssh.py` ->
   `remote_exec.py`, rename protocol and all implementations, update all imports.

2. **Rename `VmHandle` -> `RemoteWorkerHandle`**: rename protocol, all implementations
   (`GcpVmHandle` -> `GcpWorkerHandle`, `StandaloneVmHandle` -> `StandaloneWorkerHandle`,
   `_LocalVmHandle` -> `_LocalWorkerHandle`), rename `CloudVmState` -> `CloudWorkerState`,
   `VmStatus` -> `WorkerStatus`, `vm_id` -> `worker_id` on protocol, `SliceStatus.vms` ->
   `SliceStatus.workers`, `TrackedVm` -> `TrackedWorker`. Update all call sites.

3. **Move `bootstrap()` and `wait_for_connection()` off the `RemoteWorkerHandle` protocol**:
   these become methods on `RemoteExecWorkerBase` (the SSH implementation base class) only.
   Remove from protocol definition. Update `_LocalWorkerHandle` to drop these methods.
   Update any code that calls these through the protocol type.

4. **Add `BootstrapConfig` proto** and `bootstrap_config` parameter to
   `Platform.create_slice()`.

5. **Move GCP bootstrap into `GcpPlatform.create_slice()`** with async model:
   `create_slice()` returns immediately with a handle in `CREATING` state. Internal
   monitoring thread runs SSH bootstrap, transitions through `BOOTSTRAPPING` -> `READY`.
   On failure, GcpPlatform deletes the TPU and marks handle `FAILED`.

6. **Same for `ManualPlatform`**: move bootstrap into `create_slice()` with async model.

7. **`LocalPlatform` accepts `bootstrap_config` for compliance, ignores it**:
   `LocalPlatform.create_slice(config, bootstrap_config)` accepts the parameter for
   protocol compliance and ignores it. LocalPlatform spawns workers directly
   (ProcessRuntime), it doesn't need bootstrap. Returns handle in READY state immediately.

8. **Simplify autoscaler**: remove `WorkerBootstrap` dependency, remove
   `bootstrap_slice_vms()`, `_bootstrap_slice()`, `_bootstrap_slice_safe()`.
   `_do_scale_up()` calls `group.scale_up()` -> `platform.create_slice()` which returns
   immediately. `complete_scale_up()` is called right after. The reconcile cycle polls
   `handle.describe()` to monitor state. No more background bootstrap threads.

9. **Add `BOOTSTRAPPING` to `CloudSliceState`** enum.

10. **Run existing platform tests + GCP E2E to validate** that the refactor is
    behavior-preserving.

### Spiral 2: CoreWeave platform + ContainerdRuntime

Built on the clean interface from Spiral 1.

1. **`CoreweavePlatform` implementation**: `CoreweaveWorkerHandle`,
   `CoreweaveSliceHandle`, `CoreweavePlatform` with `create_slice` (async model),
   `list_slices`, `terminate`, `discover_controller`. Quota error detection from
   NodePool status conditions. Unit tests with mocked `kubectl` subprocess.

2. **`ContainerdRuntime` + `ContainerdContainerHandle`** in `runtime/containerd.py`.
   Unit tests with mocked `crictl` subprocess. CI test with real containerd daemon.

3. **`--runtime` flag on worker** (`worker/main.py`).

4. **Dockerfiles**: Add `crictl` to existing worker Dockerfile (single image, both
   runtimes). CoreWeave controller Dockerfile (`kubectl` instead of `gcloud`).

5. **Extend `iris build push` for ghcr.io**: Add `--registry ghcr` option to
   `build.py` push command. Supports `ghcr.io/marin-community/` (default) and
   custom orgs for private forks. Update CI pipeline to push all images
   (`iris-worker`, `iris-controller-coreweave`, `iris-task`) to ghcr.io as
   public packages in addition to GCP Artifact Registry.

6. **K8s manifests**: `infra/coreweave/k8s/` (Namespace, RBAC, ConfigMap,
   controller NodePool, controller Deployment + Service).

7. **E2E test on CKS**: deploy to a CoreWeave CKS cluster, run smoke test validating
   full lifecycle: create_slice -> worker register -> task execute -> scale down.

---

## 11. Resolved Decisions

1. **Autoscaler**: Iris owns all scaling. `autoscaling: false`. One NodePool per slice.
2. **Bootstrap**: Moved into `Platform.create_slice()` for all platforms with async state
   model. `create_slice()` returns immediately; platform monitors internally. State
   transitions: `CREATING -> BOOTSTRAPPING -> READY | FAILED`. GCP bootstrap (SSH + Docker
   script) moves from `WorkerBootstrap` into `GcpPlatform`. CoreWeave bootstrap is Pod spec
   + readiness probe. `WorkerBootstrap` class removed. Autoscaler no longer drives bootstrap
   -- it polls `handle.describe()` to observe state and acts on `READY`/`FAILED`.
3. **Platform cleanup on failure**: Each platform cleans up its own resources when bootstrap
   fails. GCP deletes the TPU. CoreWeave deletes the Pod and NodePool. The autoscaler calls
   `handle.terminate()` as a safety net after observing `FAILED`.
4. **Task containers**: `ContainerdRuntime` via `crictl`. No Docker socket on CoreWeave.
5. **Single worker image**: One Dockerfile with both `docker-ce-cli` and `crictl`.
   `--runtime` flag selects backend. Socket mount is a deployment concern (bootstrap
   script on GCP, Pod spec on CoreWeave).
6. **Remote execution**: `SshConnection` renamed to `RemoteExec`. No new implementations.
   CoreWeave does not use `RemoteExec` -- it has no SSH/exec in the production path.
7. **Handle rename**: `VmHandle` renamed to `RemoteWorkerHandle`. On CoreWeave it represents
   a Pod, on GCP a TPU VM. `internal_address` = Pod IP on CoreWeave, TPU VM IP on GCP.
   `vm_id` renamed to `worker_id` on protocol. `CloudVmState` -> `CloudWorkerState`.
   `SliceStatus.vms` -> `SliceStatus.workers`.
8. **`bootstrap()` and `wait_for_connection()` removed from protocol**: These are SSH
   implementation details. They live on `RemoteExecWorkerBase` only.
   `CoreweaveWorkerHandle` does not implement them.
9. **Controller**: Kubernetes Deployment + Service, operator-managed. In-cluster
   ServiceAccount auth.
10. **Images**: `ghcr.io/marin-community/` for all Iris images. **Public** -- no auth
    needed for pull. Built via `iris build` CLI (`build.py`), which is extended with
    `--registry ghcr` for ghcr.io push alongside existing GCP Artifact Registry support.
    `imagePullSecrets` only for private registry deployments.
11. **Operator vs. platform.py**: Operator manages infrastructure (cluster, RBAC, Deployment,
    Secrets). `platform.py` manages dynamic worker resources (NodePools, Pods).
12. **Health probes**: HTTP GET on `/health`. Works because Connect protocol serves JSON
    over HTTP.
13. **`BootstrapConfig` proto**: Defines image, task_image, worker_port, env_vars, cache_dir,
    runtime. Passed to `Platform.create_slice()`. LocalPlatform accepts and ignores it.
14. **Spiral implementation**: Spiral 1 refactors platform layer on existing platforms (GCP +
    Manual + Local) before writing any CoreWeave code. Spiral 2 builds CoreWeave on the clean
    interface.
15. **Async `create_slice()` model**: `create_slice()` returns immediately with a handle in
    `CREATING` state. Platform monitors internally. Autoscaler polls `describe()`.

## 12. Open Questions

1. **Multi-node slices**: `num_vms > 1` may need NodePool with `targetNodes: N` or grouped
   NodePools. InfiniBand co-scheduling needs investigation.

2. **NodePool creation rate limits**: Many 1-node NodePools at scale (50+) -- need to validate
   with CoreWeave.

3. **Bundle storage**: GCS with cross-cloud pull, or CoreWeave's S3-compatible storage.

4. **TODO: Research task container networking for crictl sandboxes.** The worker Pod runs in
   the Kubernetes pod network. Task containers created via `crictl runp` get their own pod
   sandbox with a separate network namespace by default. The only requirement is that workers
   and controller can reach each other -- what is the right equivalent of `--net=host` (which
   GCP uses for Docker) in the crictl/containerd world? Options: (a) set
   `"linux.security_context.namespace_options.network": 2` (NODE mode) on the sandbox to
   share the host network, (b) use the worker Pod's network namespace for the sandbox. Needs
   testing on a real containerd setup.

---

## References

- [CoreWeave CKS Introduction](https://docs.coreweave.com/docs/products/cks)
- [CKS Cluster Creation](https://docs.coreweave.com/docs/products/cks/clusters/create)
- [API Access Tokens and Kubeconfig](https://docs.coreweave.com/docs/products/cks/auth-access/manage-api-access-tokens)
- [Managed Auth](https://docs.coreweave.com/docs/products/cks/auth-access/managed-auth/introduction)
- [Unmanaged Auth (Service Account tokens)](https://docs.coreweave.com/docs/changelog/release-notes/unmanaged-auth)
- [CoreWeave Node Pools](https://docs.coreweave.com/docs/products/cks/nodes/nodes-and-node-pools)
- [CoreWeave Node Pool Reference](https://docs.coreweave.com/docs/products/cks/reference/node-pool)
- [CoreWeave Autoscaling](https://docs.coreweave.com/docs/products/cks/nodes/autoscaling)
- [CoreWeave GPU Instances](https://docs.coreweave.com/docs/products/instances/gpu-instances)
- [CoreWeave Terraform Provider](https://docs.coreweave.com/docs/products/cks/terraform/about)
- [IAM Access Policies](https://docs.coreweave.com/docs/security/iam/access-policies)
- [CRI tools / crictl](https://github.com/kubernetes-sigs/cri-tools)
