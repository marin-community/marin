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

---

## 1. Responsibility Boundary: Operator vs. `platform.py`

A clean separation between what a human operator does (one-time / infrequent) and what
`CoreweavePlatform` does at runtime (every scale-up/down cycle):

```
OPERATOR (one-time / config change)           PLATFORM.PY (runtime, every scale cycle)
─────────────────────────────────────         ─────────────────────────────────────────
1. Create CKS cluster (Terraform/UI)         1. create_slice():
2. Obtain kubeconfig + API token                 kubectl apply NodePool CRD
3. Push images to ghcr.io                        kubectl apply worker Pod
4. kubectl apply:                             2. list_slices():
   - Namespace                                   kubectl get nodepools -l ...
   - ConfigMap (iris config)                  3. describe():
   - Secrets (image pull, bundle creds)          kubectl get nodepool status
   - RBAC (ServiceAccount, ClusterRole)          kubectl get pod status
   - Controller NodePool                      4. terminate():
   - Controller Deployment + Service             kubectl delete pod
5. Verify controller is healthy                  kubectl delete nodepool
6. Port-forward for external access           5. discover_controller():
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
5. **Copy the token immediately** — it is shown only once. Format: `CW-SECRET-XXXXXXXXXXXXX`

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

The kubeconfig uses CoreWeave **Managed Auth** — your API token (`CW-SECRET-...`) is embedded
in the kubeconfig as a bearer token. This authenticates against CoreWeave's managed auth
endpoint.

**Important**: The controller Pod running *inside* the cluster uses **in-cluster service
account authentication**, not this kubeconfig. In-cluster auth uses the standard Kubernetes
service account token mounted at `/var/run/secrets/kubernetes.io/serviceaccount/token`.
This works against the **unmanaged auth endpoint** which CKS provides automatically.

### Step 4: Push Images to GHCR

```bash
# Authenticate with GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u <github-username> --password-stdin

# Build and push from lib/iris/
cd lib/iris

# Controller image (CoreWeave variant — no gcloud, has kubectl)
docker build -f Dockerfile.controller.coreweave -t ghcr.io/marin-community/iris-controller:latest .
docker push ghcr.io/marin-community/iris-controller:latest

# Worker image (CoreWeave variant)
docker build -f Dockerfile.worker.coreweave -t ghcr.io/marin-community/iris-worker:latest .
docker push ghcr.io/marin-community/iris-worker:latest

# Task image (unchanged from GCP)
docker build -f Dockerfile.task -t ghcr.io/marin-community/iris-task:latest .
docker push ghcr.io/marin-community/iris-task:latest
```

### Step 5: Create Kubernetes Secrets

```bash
# Image pull secret for GHCR
kubectl create namespace iris

kubectl create secret docker-registry iris-image-pull \
  --namespace iris \
  --docker-server=ghcr.io \
  --docker-username=<github-username> \
  --docker-password=$GITHUB_TOKEN

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

### Decision 3: `kubectl exec` as drop-in for SSH

`KubectlExecConnection` implements `SshConnection` protocol. All existing infrastructure
(`SshVmBase`, `WorkerBootstrap`, `run_streaming_with_retry`) works unchanged.

### Decision 4: Images on ghcr.io

All Iris images (`iris-controller`, `iris-worker`, `iris-task`) are pushed to
`ghcr.io/marin-community/`. Pods use `imagePullSecrets` referencing a GHCR token.

---

## 5. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  CoreWeave CKS Cluster                                           │
│                                                                  │
│  ┌─────────────────────────────────┐                             │
│  │  Controller Deployment          │  ◄── operator applies       │
│  │  (always-on, controller pool)   │      (Deployment + Service) │
│  │                                 │                             │
│  │  Pod: iris-controller           │                             │
│  │    ghcr.io/.../iris-controller  │                             │
│  │    port 10000                   │                             │
│  │    in-cluster K8s auth          │ ◄── ServiceAccount          │
│  │    /etc/iris/config.yaml        │ ◄── ConfigMap               │
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

**`infra/coreweave/k8s/rbac.yaml`** — ServiceAccount + ClusterRole for the controller:

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

**`infra/coreweave/k8s/configmap.yaml`** — Iris cluster config:

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
        worker_port: 10001
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
      imagePullSecrets:
        - name: iris-image-pull
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
          livenessProbe:
            httpGet:
              path: /health
              port: 10000
            initialDelaySeconds: 10
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
   - No kubeconfig file needed — the `kubernetes` client auto-detects in-cluster config
3. `WorkerBootstrap(cluster_config)` created
4. `create_autoscaler()` creates `ScalingGroup` per config entry
5. `autoscaler.reconcile()` — discovers existing worker NodePools/Pods by label
6. `Controller.start()` — scheduling loop, heartbeat loop, autoscaler loop, HTTP server

### Phase 2: Scale-Up (`platform.py` creates worker resources)

When demand arrives (job submitted), the autoscaler evaluates and calls into `platform.py`:

1. **Autoscaler decides** to scale up group `h100_8x`
2. **`CoreweavePlatform.create_slice(config)`** (this is `platform.py`):
   ```
   a. Generate slice ID: iris-h100-8x-{timestamp_ms}
   b. kubectl apply NodePool CRD:
        apiVersion: compute.coreweave.com/v1alpha1
        kind: NodePool
        metadata:
          name: iris-h100-8x-1738000000000
        spec:
          instanceType: gd-8xh100ib-i128
          targetNodes: 1
          autoscaling: false
          nodeLabels:
            iris-managed: "true"
            iris-scale-group: h100_8x
            iris-slice-id: iris-h100-8x-1738000000000
   c. Return CoreweaveSliceHandle
   ```

3. **Bootstrap thread** polls until node ready (up to 30 min), then `platform.py` creates
   worker Pod:
   ```
   kubectl apply Pod:
     name: iris-worker-iris-h100-8x-1738000000000
     namespace: iris
     labels:
       iris-managed: "true"
       iris-slice-id: iris-h100-8x-1738000000000
     spec:
       nodeSelector:
         iris-slice-id: iris-h100-8x-1738000000000
       imagePullSecrets:
         - name: iris-image-pull
       containers:
         - name: iris-worker
           image: ghcr.io/marin-community/iris-worker:latest
           command: [worker serve ...]
           resources:
             limits:
               nvidia.com/gpu: 8
               rdma/ib: 1
           volumeMounts:
             /etc/iris from ConfigMap
             /var/cache/iris from hostPath
             /var/run/docker.sock from hostPath
           env:
             - name: IRIS_VM_ADDRESS
               valueFrom:
                 fieldRef:
                   fieldPath: status.podIP
   ```

4. **Bootstrap via `kubectl exec`**: Once Pod is running, `WorkerBootstrap.bootstrap_vm()`
   connects via `KubectlExecConnection` and runs a health-check script.

### Phase 3: Worker Registration

Worker Pod entrypoint runs:
```
python -m iris.cluster.worker.main serve --config /etc/iris/config.yaml
```

1. `load_config()` reads config from ConfigMap mount
2. `create_platform()` returns `CoreweavePlatform` (in-cluster auth)
3. `platform.discover_controller()` returns `iris-controller-svc.iris.svc.cluster.local:10000`
4. `Worker._register()` sends RPC to controller via Service DNS
5. Controller accepts, assigns `worker_id`
6. Worker enters heartbeat serve loop

### Phase 4: Task Execution

Standard Iris flow — no CoreWeave-specific changes.

### Phase 5: Scale-Down (`platform.py` destroys worker resources)

1. Autoscaler picks the specific longest-idle slice
2. `handle.terminate()` calls `platform.py`:
   ```
   kubectl delete pod iris-worker-iris-h100-8x-1738000000000 -n iris
   kubectl delete nodepool iris-h100-8x-1738000000000
   ```
3. Node deprovisioned — deterministic, exactly the chosen slice

---

## 7. Secrets and Credentials Summary

### What the Operator Creates

| Resource | Purpose | How to Obtain |
|----------|---------|---------------|
| CoreWeave API token | Terraform + kubeconfig | Console > Tokens > Create Token → `CW-SECRET-...` |
| Kubeconfig | Operator's kubectl access | Console > Tokens > Download Kubeconfig |
| `iris-image-pull` Secret | Pull from ghcr.io | `kubectl create secret docker-registry` with GitHub PAT |
| `iris-bundle-credentials` Secret | GCS or S3 access for job bundles | GCS SA key JSON or CoreWeave S3 credentials |
| `iris-controller` ServiceAccount | In-cluster K8s API auth for controller | `kubectl apply` RBAC manifests |

### What Platform.py Uses at Runtime

| Credential | How Obtained | Used For |
|------------|-------------|----------|
| In-cluster ServiceAccount token | Auto-mounted by Kubernetes at `/var/run/secrets/...` | `kubectl` calls to create/delete NodePools + Pods |
| `iris-image-pull` Secret | Referenced in Pod spec `imagePullSecrets` | Pulling worker images from ghcr.io |
| `iris-cluster-config` ConfigMap | Mounted into Pods at `/etc/iris/config.yaml` | Config for controller and workers |
| Bundle credentials | Mounted from Secret into Pod at `/etc/iris/credentials/` | Downloading/uploading job bundles |

**No kubeconfig file in platform.py.** The controller runs inside the cluster and uses
Kubernetes in-cluster authentication (service account token). `kubectl` auto-detects this.
The `kubeconfig_path` field in `CoreweavePlatformConfig` is only needed if the controller
runs *outside* the cluster (e.g., during local development).

---

## 8. Code Changes: File-by-File

### `config.proto` — Extend proto definitions

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
Same interface as `DirectSshConnection`. The controller Pod's ServiceAccount gives `kubectl`
permission to exec into worker Pods.

### `coreweave.py` — Implement `CoreweavePlatform`

Replace stub with full implementation. All `kubectl` commands use in-cluster auth by default
(no `--kubeconfig` flag needed). If `kubeconfig_path` is set, pass `--kubeconfig`.

- `create_slice()`: `kubectl apply` NodePool CRD + worker Pod
- `list_slices()`: `kubectl get nodepools -l iris-scale-group=X -o json`
- `list_all_slices()`: `kubectl get nodepools -l iris-managed=true -o json`
- `describe()` on handle: Query NodePool status + Pod status
- `terminate()` on handle: Delete Pod, then delete NodePool
- `discover_controller()`: `{service_name}.{namespace}.svc.cluster.local:{port}`
- `tunnel()`: `kubectl port-forward`

### `bootstrap.py` — Add CoreWeave bootstrap script

Simplified script: config already mounted via ConfigMap, worker running as Pod entrypoint.
Script only verifies health.

### `factory.py` — Wire up CoreweavePlatform

Pass config fields. No `ssh_config` needed (uses `kubectl exec`, not SSH).

### New Dockerfiles

`Dockerfile.controller.coreweave` and `Dockerfile.worker.coreweave`:
- Base: `python:3.11-slim`
- Install: `curl`, `git`, `kubectl` (no `google-cloud-cli`)
- Install iris via `uv sync`
- Worker additionally needs `docker-ce-cli` (for task containers via Docker socket)

### New manifests: `infra/coreweave/k8s/`

- `namespace.yaml`
- `rbac.yaml` (ServiceAccount + ClusterRole + ClusterRoleBinding)
- `configmap.yaml` (template — operator fills in scale groups)
- `controller-nodepool.yaml`
- `controller-deployment.yaml` (Deployment + Service)

---

## 9. Minimal End-to-End Test Plan

### Goal

Submit a simple CPU job to an Iris cluster running entirely on CoreWeave, verify it succeeds.

### Test Config

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
    docker_image: ghcr.io/marin-community/iris-worker:latest
    worker_port: 10001
  timeouts:
    boot_timeout: { milliseconds: 1800000 }
    init_timeout: { milliseconds: 1800000 }
  autoscaler:
    evaluation_interval: { milliseconds: 5000 }
    scale_up_delay: { milliseconds: 5000 }
    scale_down_delay: { milliseconds: 30000 }
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
        instance_type: <cpu-instance>
        gpus_per_node: 0
```

### Running

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

### What the Test Validates

1. Controller starts from ConfigMap, authenticates to K8s via ServiceAccount
2. `CoreweavePlatform.create_slice()` creates NodePool + worker Pod
3. Worker discovers controller via K8s Service DNS, registers
4. Controller schedules task, worker executes, reports completion
5. Autoscaler scales down idle slice

---

## 10. Implementation Order

### Phase 1: Core Platform

1. Extend `config.proto` with CoreWeave messages
2. Regenerate proto bindings
3. `KubectlExecConnection` in `ssh.py`
4. `CoreweaveVmHandle`, `CoreweaveSliceHandle`
5. `CoreweavePlatform` core: `create_slice`, `list_slices`, `terminate`, `discover_controller`
6. Unit tests with mocked `kubectl` subprocess

### Phase 2: Deployment Manifests + Dockerfiles

7. CoreWeave Dockerfiles (controller + worker — no gcloud, has kubectl)
8. K8s manifests: `infra/coreweave/k8s/` (Namespace, RBAC, ConfigMap, Deployment, Service)
9. CoreWeave bootstrap script (simplified health-check only)
10. Test controller starts in a CKS cluster

### Phase 3: End-to-End

11. E2E test config + bootstrap manifests
12. Deploy to CKS, run smoke test
13. Validate full lifecycle

### Phase 4: Production Hardening

14. NodePool status condition parsing (quota, capacity errors)
15. Multi-node slice support
16. Monitoring + CI integration

---

## 11. Resolved Decisions

1. **Autoscaler**: Iris owns all scaling. `autoscaling: false`. One NodePool per slice.
2. **SSH -> kubectl exec**: `KubectlExecConnection` as `SshConnection` drop-in.
3. **Controller**: Kubernetes Deployment + Service, operator-managed. In-cluster ServiceAccount auth.
4. **Images**: `ghcr.io/marin-community/` for all Iris images.
5. **Operator vs. platform.py**: Operator manages infrastructure (cluster, RBAC, Deployment, Secrets). `platform.py` manages dynamic worker resources (NodePools, Pods).

## 12. Open Questions

1. **Multi-node slices**: `num_vms > 1` may need NodePool with `targetNodes: N` or grouped
   NodePools. InfiniBand co-scheduling needs investigation.

2. **NodePool creation rate limits**: Many 1-node NodePools at scale (50+) — need to validate
   with CoreWeave.

3. **Task container model**: Workers use Docker socket for task containers. Need to confirm
   Docker is available on CoreWeave bare metal nodes and `hostPath` mount works.

4. **Bundle storage**: GCS with cross-cloud pull, or CoreWeave's S3-compatible storage.

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
