# CoreWeave Platform Integration

**Issue**: [#2822 -- Iris: Implement CoreWeave platform](https://github.com/marin-community/marin/issues/2822)

## 0. Quickstart

Zero to a running job on a CoreWeave H100 cluster. The rest of this document is
the full operator runbook (RBAC, NodePools, Kueue, troubleshooting).

Active clusters:

All clusters share one kubeconfig at `~/.kube/coreweave-iris`; each cluster
config pins its own `kube_context` inside it, so iris/kubectl operations are
context-bound per `--cluster` and never depend on the file's current-context
or an exported `KUBECONFIG`.

| Iris cluster | CW cluster / region | Fleet | Kube context |
|--------------|---------------------|-------|--------------|
| `cw-us-east-02a` | `marin-gpu`, US-EAST-02A | 32× 8xH100 + 4× CPU Genoa, pinned warm | `marin-gpu_US-EAST-02A` |
| `cw-rno2a` | `marin-rn02a`, RNO2A | 64× 8xH100 + 1× CPU Turin, pinned warm | `marin-rn02a_RNO2A` |

Console links:
- Tokens (kubeconfig): https://console.coreweave.com/tokens
- Cluster details: https://console.coreweave.com/zones/US-EAST-02A/clusters/marin-gpu#details
- Health dashboard: https://cks-grafana.coreweave.com/d/cluster-health/cluster-health?var-cluster-org=208261&var-cluster=marin-gpu&var-region=US-EAST-02

**1. Make a token / kubeconfig.** In the [Tokens console](https://console.coreweave.com/tokens),
create a token and download the kubeconfig — it carries a context per cluster
(named `<cw-cluster>_<REGION>`).

**2. Install the kubeconfig** at `~/.kube/coreweave-iris`, plus controller
extras:

```bash
mkdir -p ~/.kube
mv ~/Downloads/kubeconfig.yaml ~/.kube/coreweave-iris
kubectl --kubeconfig ~/.kube/coreweave-iris config get-contexts   # sanity check

uv pip install 'marin-iris[controller]'
```

No `KUBECONFIG` export is needed: the cluster configs pin `kubeconfig_path` and
`kube_context`, and every operation binds to that context explicitly.

That's all a job submitter needs. `CW_KEY_ID` / `CW_KEY_SECRET` (CoreWeave
Object Storage access keys) are only required when running
`iris cluster start` — they seed the in-cluster `iris-task-env` Secret (see
"Storage defaults" below).

**3. Check cluster status.** `--cluster=cw-us-east-02a` resolves the in-tree
config and opens a `kubectl port-forward` to the controller for you:

```bash
uv run iris --cluster=cw-us-east-02a cluster status
```

If the controller isn't up yet, start it (idempotent):
`uv run iris --cluster=cw-us-east-02a cluster start`.

**4. Hello world.**

```bash
# CPU
uv run iris --cluster=cw-us-east-02a job run \
  --cpu 1 --memory 2GB --extra cpu \
  -- python -c "print('Hello from CoreWeave!')"

# One H100, proving JAX sees the GPU
uv run iris --cluster=cw-us-east-02a job run \
  --cpu 8 --memory 64GB --gpu H100x1 --enable-extra-resources --extra gpu \
  -- python -c "import jax; print(jax.devices())"
```

Follow logs of a detached job with
`uv run iris --cluster=cw-us-east-02a job logs <job-id> -f`.

**Storage defaults.** CoreWeave clusters default to CoreWeave AI Object
Storage — no per-job storage setup is needed:

- `MARIN_PREFIX` is preset to `s3://marin-us-east-02a/marin` (via
  `defaults.task_env` in the cluster config) on both clusters.
- Task pods carry the CoreWeave Object Storage S3 configuration and
  credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
  `AWS_ENDPOINT_URL`, `FSSPEC_S3`, ...) via the platform-managed
  `iris-task-env` Secret, so `s3://` reads/writes work out of the box.
- Task pods carry ONE endpoint/credential set. Data on other S3-compatible
  stores is not reachable unless the job overrides `AWS_*`/`FSSPEC_S3` itself.

## 1. Overview

Iris runs on CoreWeave CKS (bare-metal Kubernetes) using a shared NodePool model.
Each Iris scale group maps to one CoreWeave NodePool with autoscaling enabled.
CoreWeave manages node provisioning and deprovisioning; Iris manages only Pods.
Tasks execute as independent Kubernetes Pods via `KubernetesRuntime`
(Pod-per-task).

Example config: `lib/iris/config/examples/coreweave.yaml`

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

- **`CLUSTER_VIEW` `TaskBackend`**: When the cluster config sets
  `kubernetes_provider`, the controller runs `K8sTaskProvider`
  (`src/iris/cluster/backends/k8s/tasks.py`) — a `TaskBackend` whose
  `capabilities` is `{CLUSTER_VIEW}`. Kueue performs scheduling and the cluster
  autoscaler provisions nodes, so its `schedule`/`autoscale` are effectively
  no-ops and `reconcile` only reconciles desired vs. observed Pods each tick. The
  controller calls the same three uniform phase methods regardless. The dashboard
  reflects this via the backend descriptor served by
  `/auth/config`: capability `cluster` shows the **Cluster** panel, and the
  Workers/Autoscaler panels are hidden (no worker daemons, no Iris autoscaler).
  See `docs/architecture.md` "The TaskBackend contract".
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

### CoreWeave controller networking (IP-locked to marin)

The controller Service is `ClusterIP:10000` — reachable only in-cluster or via a
`kubectl port-forward`. CoreWeave has no user surface of its own: end users reach
Iris through `iris.oa.dev` (IAP) and the GCP `marin` controller federates jobs
outward, so the only external caller of a CoreWeave controller is marin. Its one
off-cluster ingress is therefore locked to marin's egress IP — there is no
world-open route.

`marin` federates whole jobs by dialing the CoreWeave RPC surface directly (the
pull model — CoreWeave must be reachable *inbound*; a peer behind NAT with no
ingress is out of scope). The auth surface is two factors, both required and
neither alone sufficient:

1. **IP allowlist** — a Traefik `ipAllowList` Middleware admits only the egress IPs
   of the marin-side controllers that federate here (`FEDERATION_ALLOW_SOURCES` in
   `install_cw_network.py`). The *network* factor.
2. **The controller's own auth** — the `aud="federation"` verifier for the handoff
   RPCs, the general auth chain for the rest. The *identity* factor.

Because the controller is the identity gate, it **must be enforcing** — a
permissive (null-auth) controller behind only an IP lock hands anonymous admin over
its whole control plane to anything from the allowlisted IP. Both CoreWeave
controllers enforce via `auth.trusted_cidrs`: in-cluster peers (RFC1918 pods/nodes,
loopback for `kubectl port-forward`) authenticate by network location, while an
off-cluster request arrives through Traefik with an appended `X-Forwarded-For`,
never matches a CIDR, and must present a bearer.

CKS ships no ingress controller and no TLS issuer, and the controller's own
ServiceAccount can't install CRDs, so an operator runs `scripts/install_cw_network.py`
once per cluster to set up everything the controller can't do itself — Traefik,
cert-manager, the HTTP-01 Let's Encrypt issuers, and the single IP-locked ingress
that publishes the whole controller host to marin. It is idempotent and warns if
pointed at a still-permissive controller; dry-run without `--apply`:

```bash
# Traefik + cert-manager + HTTP-01 issuers + the IP-locked ingress, in one pass.
uv run lib/iris/scripts/install_cw_network.py --cluster <name> \
    install --acme-email you@oa.dev --allow-source <marin-egress-ip> --apply
# Reconcile just the ingress on a cluster whose stack is already installed
# (e.g. to flip staging->prod once the cert validates):
uv run lib/iris/scripts/install_cw_network.py --cluster <name> install \
    --allow-source <marin-egress-ip> --skip-traefik --skip-cert-manager --skip-issuers \
    --cluster-issuer letsencrypt-http01-prod --apply
# Tear it all back down (ingress, releases, namespaces, CRDs/webhooks/RBAC), verified:
uv run lib/iris/scripts/install_cw_network.py --cluster <name> uninstall --apply
```

The ingress is a standalone object, so applying it does **not** restart the
controller. If CoreWeave's LoadBalancer SNATs (Traefik sees the LB, not the real
client), pass `--xff-depth 1` so the allowlist reads the client IP from
`X-Forwarded-For`; confirm a request from a non-allowlisted host is refused.

#### External address and DNS (`oa.dev` -> `coreweave.app`)

The external address is served by Traefik's LoadBalancer, not the controller Pod.
CoreWeave publishes a wildcard record, `*.<tenant>.coreweave.app`, that resolves to
that LoadBalancer. `install_cw_network.py install --apply` prints the exact CNAME
record to create, substituting the ingress host's first label into the wildcard
(`iris-cw-<cluster>.oa.dev  CNAME  iris-cw-<cluster>.<tenant>.coreweave.app`); `oa.dev`
DNS is at Namecheap, Advanced DNS panel. Routing, Host matching, and TLS all key on
the `oa.dev` name; `coreweave.app` is only the CNAME target.

TLS terminates in-cluster (no IAP/edge layer; Namecheap doesn't proxy TLS).
`install_cw_network.py` creates HTTP-01 Let's Encrypt ClusterIssuers
(`letsencrypt-http01-staging`, `letsencrypt-http01-prod`) validated through
Traefik — CoreWeave's bundled issuers only cover `*.coreweave.app` (DNS-01 via
`acme.coreweave.com`), so a custom `oa.dev` host needs these. HTTP-01 needs the
CNAME live first (Let's Encrypt fetches `http://<host>/.well-known/...`); issue with
the staging issuer first to avoid LE rate limits, then re-run with
`--cluster-issuer letsencrypt-http01-prod`. cert-manager's HTTP-01 solver runs on
its own unrestricted Ingress, so the IP allowlist does not block ACME validation.

**Verify from the marin controller VM:**

- `ListBackends` **with** the federation JWT from the allowlisted egress IP -> succeeds.
- the same call **without** the JWT (controller enforcing) -> `UNAUTHENTICATED`.
- the same call from a **non-allowlisted IP** -> refused (`403`).

#### Stable egress IPs for the marin controllers

The allowlist names each federating controller by address, so every one of them
needs a **stable** egress IP: `iris-controller-marin` and `iris-controller-marin-dev`,
reserved as `iris-marin-fed-egress` and `iris-marin-dev-fed-egress`. A controller's
GCE VM (zone `us-central1-a`, project `hai-gcp-models`) is created with an
*ephemeral* external IP, which changes on stop/start. Make it stable **without
touching the VM** by promoting that in-use address to a reserved static IP — same
address, no access-config change, no restart:

```bash
PROJECT=hai-gcp-models ZONE=us-central1-a REGION=us-central1 VM=iris-controller-marin
# 1. Read the VM's current external IP.
EGRESS_IP=$(gcloud compute instances describe "$VM" --project "$PROJECT" --zone "$ZONE" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
echo "marin controller egress IP: $EGRESS_IP"
# 2. Promote that exact IP to a reserved static address (idempotent; no VM change,
#    no connectivity drop — the VM keeps the same address).
gcloud compute addresses create iris-marin-fed-egress --project "$PROJECT" \
  --region "$REGION" --addresses "$EGRESS_IP"
```

Add `$EGRESS_IP` to `FEDERATION_ALLOW_SOURCES`, which is what `--allow-source`
defaults to. The Middleware's `sourceRange` is replaced wholesale on every install
rather than merged, so an install naming a subset strands the omitted controller at
a `403` the peer never logs. Reserving an in-use address is safe and reversible
(`gcloud compute addresses delete iris-marin-fed-egress` releases it back to
ephemeral). **Alternative** (only if the controller is later moved to
IAP-only inbound with *no* external IP, so egress goes through Cloud NAT): reserve
a regional static IP and pin Cloud NAT to it (`gcloud compute routers nats update
… --nat-external-ip-pool=iris-marin-fed-egress`). Removing the VM's external IP is
a larger change — do it only with explicit approval, never as a side effect here.

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

§0 is the quickstart for the `marin-gpu` cluster. This section is the generic
operator reference (any `--cluster=NAME`) and the lifecycle details behind it.

### Prerequisites

- A CoreWeave CKS cluster (created via Console or Terraform)
- A kubeconfig downloaded from CoreWeave Console > Tokens (see §0)
- Images pushed to `ghcr.io/marin-community/`
- Controller extras: `uv pip install 'marin-iris[controller]'`

CoreWeave clusters default to CoreWeave AI Object Storage, and the cluster configs
give two endpoints for it. `object_storage_endpoint: http://cwlota.com` is LOTA, the
in-cluster cache, which pods use. `external_object_storage_endpoint:
https://cwobject.com` reaches the same buckets from an operator's own machine, which
`iris` uses for its own S3 calls — among them the rollout record that
`iris cluster controller restart` reads. LOTA's name resolves to a private address,
so both fields are needed on a cluster an operator drives from outside.

Export `CW_KEY_ID` / `CW_KEY_SECRET` (CoreWeave Object Storage access keys) before
`iris cluster start`; it folds them — plus the derived
endpoint/region/`FSSPEC_S3` config — into the `iris-task-env` Secret, projected
into the controller and task pods via `envFrom`. From then on every task has
working S3 credentials embedded; job submitters never handle storage keys.

> **Note**: CoreWeave AI Object Storage (`cwobject.com`, `cwlota.com`) uses
> virtual-hosted-style S3 addressing, which is auto-detected and configured
> (including JAX/tensorstore checkpointing).

### CoreWeave AI Object Storage access

Use `s3://marin-us-east-02a` for CoreWeave-local object storage — it is the
shared bucket for both clusters, and `MARIN_PREFIX` is preset to
`s3://marin-us-east-02a/marin` in every task. **Inside the cluster none of the
setup below is needed**: task pods already carry the endpoint, addressing
style, and credentials (see "Storage defaults" in §0). The rest of this
section is for access from *outside* CoreWeave (laptop, GCP). The bucket is
browsable in the
[CoreWeave console](https://console.coreweave.com/object-storage/buckets/marin-us-east-02a).
Follow CoreWeave's
[endpoint](https://docs.coreweave.com/products/storage/object-storage/using-object-storage/configure-endpoints)
and
[object-management](https://docs.coreweave.com/products/storage/object-storage/using-object-storage/manage-objects)
docs; Marin-specific settings are:

- Credentials: create an Object Storage access key in the
  [CoreWeave console](https://console.coreweave.com/object-storage/access-keys);
  use the Key ID as `CW_ACCESS_KEY_ID` and the Key secret as
  `CW_SECRET_ACCESS_KEY`.
- Endpoint: `https://cwobject.com` outside CoreWeave, `http://cwlota.com`
  inside CoreWeave.
- Region: `US-EAST-02A`.
- Addressing: `s3.addressing_style = virtual`; path-style requests are not
  supported.

One-off AWS CLI check, without persistent AWS config:

```bash
export CW_ACCESS_KEY_ID=<your-coreweave-object-storage-key-id>
export CW_SECRET_ACCESS_KEY=<your-coreweave-object-storage-key-secret>

tmp_config="$(mktemp)"
trap 'rm -f "$tmp_config"' EXIT

cat >"$tmp_config" <<'EOF'
[default]
s3 =
    addressing_style = virtual
EOF

AWS_CONFIG_FILE="$tmp_config" \
AWS_ACCESS_KEY_ID="$CW_ACCESS_KEY_ID" \
AWS_SECRET_ACCESS_KEY="$CW_SECRET_ACCESS_KEY" \
AWS_REGION=US-EAST-02A \
AWS_ENDPOINT_URL_S3=https://cwobject.com \
AWS_PAGER="" \
aws s3 ls s3://marin-us-east-02a/
```

### Lifecycle

```bash
iris --cluster=<name> cluster start      # idempotent; reconciles everything below
iris --cluster=<name> cluster status
iris --cluster=<name> cluster dashboard
iris --cluster=<name> cluster stop       # deletes Pods + controller; NodePools survive
```

`cluster start` creates/reconciles, in order:

1. Namespace (`iris`) and RBAC (ServiceAccount, ClusterRole, ClusterRoleBinding)
2. S3 credentials Secret (if S3 storage URIs are configured)
3. ConfigMap (`iris-cluster-config`) with the cluster config as JSON
4. Shared NodePools (one per scale group, in parallel)
5. Controller Deployment (`iris-controller`) — images are built and pushed automatically
6. Controller Service (`iris-controller-svc`, ClusterIP)

`cluster stop` leaves NodePools in place; they scale to zero when idle (but
still bill — see the NodePool cleanup under §4 Gotchas).

### Kueue (gang admission + TAS)

Multi-host GPU jobs are gang-admitted (all-or-nothing) by Kueue's plain-Pod
integration, with Topology-Aware Scheduling placing gangs on the InfiniBand
fabric. Install per cluster with:

```bash
uv run python lib/iris/scripts/install_kueue.py --variant coreweave \
  --kubeconfig <kubeconfig> --with-queues --apply
```

This installs the CoreWeave `cks-kueue` chart, the Topology CRs, the
cluster-scoped `cw-ib` + `cw-cpu` ResourceFlavors, and the `iris-cq` ClusterQueue.
Iris reconciles its namespaced LocalQueue (`{label_prefix}-lq`) at controller
start, bound via `kubernetes_provider.kueue.cluster_queue`.

**Flavor routing.** Every pod on the k8s backend routes through Kueue (not just
gangs), so `iris-cq` carries two flavors in one resourceGroup:

- **`cw-ib`** selects InfiniBand nodes (`backend.coreweave.cloud/flavor=infiniband`)
  and binds the `infiniband` Topology for TAS. GPU pods request `nvidia.com/gpu` +
  `rdma/ib` and match here.
- **`cw-cpu`** (listed first) zeroes the `nvidia.com/gpu` and `rdma/ib` quota, so a
  GPU pod can never be admitted under it and falls through to `cw-ib`, while a
  CPU-only pod matches `cw-cpu`. It is **selector-less** by default — its spec is
  empty, so Kueue injects no `nodeSelector` and the admitted CPU pod stays
  schedulable on any node. It prefers CPU nodes (GPU nodes carry a soft
  `PreferNoSchedule` taint) but reuses idle GPU nodes when CPU capacity is full,
  instead of being fenced onto CPU-only capacity. Pass `--cpu-flavor-node-label
  KEY=VALUE` to pin `cw-cpu` to specific CPU nodes instead.

**GB200 NVLink-domain placement.** NVL72 nodes (GB200/GB300, `gb200-4x` — 4
Blackwell GPUs each) carry `ds.coreweave.com/nvlink.domain`; one NVL72 rack is one
NVLink domain of 18 physical nodes, of which CoreWeave keeps only **16 schedulable
at once** (the rest absorb host failures and maintenance — a floor, not a cap).
Iris picks a placement level per multi-node GB200 gang from its replica count:

- **≤ 16 nodes** bind **hard** to a single `nvlink.domain`
  (`podset-required-topology`) — the whole gang shares one rack's NVLink fabric.
  16 is the largest hard single-domain gang; binding a 17–18-node gang hard would
  demand a fully-healthy rack and could sit unschedulable whenever one is short a
  node.
- **> 16 nodes** use Kueue **PodSet slices** (`podset-slice-required-topology:
  nvlink.domain` + a computed `podset-slice-size`). The gang is split evenly over
  the fewest racks that hold ≤ 16 nodes each, and each slice is hard-bound to its
  own NVLink domain, so it lands as an exact **balanced N-rack layout**: 24 → 12+12,
  32 → 16+16, 48 → 16+16+16. Because two slices always exceed a rack (each is more
  than half of 18), no two share a rack. A soft `leafgroup` preference is paired so
  the racks also cluster on one IB leaf group.

The sliced level requires **node-saturating pods** (one pod = one whole `gb200-4x`
node = 4 GPUs) so a slice fills whole nodes, and a gang size that splits into equal,
more-than-half-a-rack slices — sizes that cannot (e.g. 17, 18, 40) are rejected at
submit. PodSet slices need **Kueue ≥ 0.13**; use **0.18+** for the `IsTAS()`
recognition fix for slice-only pod groups (upstream #10282). On an older Kueue the
slice annotations are silently ignored and a multi-rack gang degrades to a soft
leafgroup gang with no per-rack NVLink guarantee.

**Never install Kueue with unscoped admission webhooks on a CoreWeave cluster.**
The script scopes them to the `iris` namespace (`--pod-namespace`); the chart
default intercepts pod CREATEs in every namespace fail-closed, including the
cilium CNI pods on freshly delivered nodes — if the Kueue manager is down (e.g.
zero-node cluster), the node can never start its CNI, never goes Ready, and the
manager can never schedule: node delivery deadlocks cluster-wide. Details in
`install_kueue.py`'s module docstring.

On a zero-node cluster the install's controller-rollout wait times out (the
manager has nowhere to schedule) — provision the controller node first, or
re-run the install after it is Ready.

### Bringing up a new cluster (end-to-end rollout runbook)

The ordered sequence to take a fresh CKS cluster to a reachable, federated,
log-forwarding Iris cluster. `cw-us-east-08a` (a GB200 NVL72 fleet) is the worked
example; substitute your `<cluster>`. Steps marked **(console)** or **(manual)**
are the only ones not driven from a repo checkout on the cluster's branch — an
operator does them in the CoreWeave console or the DNS registrar.

**One-time prerequisites**

- **(console)** CKS cluster created (Console or Terraform). Kubeconfig from
  Console > Tokens installed at `~/.kube/coreweave-iris`; the context is
  `<cks-name>_<REGION>` (e.g. `marin-us-east-08a_US-EAST-08A`).
- **(console)** Object-storage bucket created (Console or `cwic`) and an Object
  Storage access key: `export CW_KEY_ID=… CW_KEY_SECRET=…`.
- Local tooling: `uv pip install 'marin-iris[controller]'`; `docker login
  ghcr.io` with a `write:packages` PAT (`cluster start` builds + pushes images);
  `gcloud auth application-default login` (ADC — `init-keys` and the deploy-time
  `gcp-secret://` resolution read Secret Manager through it).

**1. Write the cluster config.** Copy an existing `lib/iris/config/cw-*.yaml` and
adjust `name`, region, `platform.coreweave.kube_context`, the CKS
`provisioning.coreweave.cluster.name`, scale groups, and fleet sizes. The console
capacity view's display label is NOT the k8s `spec.instanceType` (e.g.
"turin-gp-l4" vs `turin-gp-l`); to probe a SKU, create a NodePool with `minNodes:
0, maxNodes: 0, targetNodes: 0` and read its `Validated` condition (server
dry-run accepts any string). For a reserved/prepaid fleet set `buffer_slices ==
max_slices` — there is nothing to save by autoscaling it down. GB200 NVL72
deploys in whole racks of 18, so a rack pool's node count must be a multiple of
18.

**2. Provision the static prerequisites (IaC).** The `infra/iac` Pulumi program
declares the namespace, controller RBAC, the reserved NodePools, and the whole
cluster-scoped Kueue substrate (`cks-kueue` release, Topology CRs, `cw-ib`
ResourceFlavor, `iris-cq` ClusterQueue, `iris-system` PriorityClass) from the
`provisioning:` section of the config. See `infra/iac/README.md`. (Pre-IaC path:
`install_kueue.py --with-queues` for Kueue and let `cluster start` create the
RBAC + NodePools.)

**3. Mint the controller signing key.** The cluster's identity — signs every
worker/proxy token it mints.

```bash
uv run iris cluster init-keys \
  --gcp-secret projects/<project-number>/secrets/iris-<cluster>-signing-key
```

Creates the secret and stores the private key as **version 1**; pin that version
in `auth.signing_key` alongside the `env:IRIS_SIGNING_KEY` marker. The
credential-less controller pod never reads Secret Manager: at deploy time the
operator's shell resolves the `gcp-secret://` ref and projects it into the
`iris-controller-env` Secret. `--rotate` replaces it later.

**4. Start the controller.**

```bash
export CW_KEY_ID=… CW_KEY_SECRET=…
uv run iris --cluster=<cluster> cluster start
uv run iris --cluster=<cluster> cluster status
```

Builds + pushes the controller/worker/task images, then applies (idempotently)
the `iris-cluster-config` ConfigMap, the `iris-task-env` + `iris-controller-env`
Secrets, the namespaced LocalQueue (`{label_prefix}-lq`), the controller
Deployment + Service + PDB, and waits for rollout. Needs a Ready controller-pool
node (the `cpu-*` scale group); on a still-delivering cluster the Deployment
stays Pending until one lands.

**5. Install the network stack, then the DNS record.** Publishes the controller
off-cluster behind an IP-locked ingress.

```bash
uv run lib/iris/scripts/install_cw_network.py --cluster <cluster> \
  install --acme-email <ops-email> --apply           # staging cert first
```

Installs Traefik + cert-manager + the HTTP-01 Let's Encrypt issuers + the
federation `Ingress` (Traefik `ipAllowList` admitting only marin's egress IPs).
Read the CNAME target off the Traefik LoadBalancer's `*.coreweave.app` wildcard:

```bash
kubectl get svc traefik -n traefik \
  -o=jsonpath='{.status.conditions[?(@.type=="ExternalRecords")].message}'
```

**(manual)** Create the record in the `oa.dev` registrar (**Namecheap**, Advanced
DNS panel — not Cloudflare):

```
iris-cw-<cluster>.oa.dev   CNAME   iris-cw-<cluster>.<tenant>.coreweave.app
```

The CNAME must be live before TLS issuance (HTTP-01 validates through Traefik).
Once it resolves and the staging cert validates, flip to the prod issuer:

```bash
uv run lib/iris/scripts/install_cw_network.py --cluster <cluster> install \
  --cluster-issuer letsencrypt-http01-prod \
  --skip-traefik --skip-cert-manager --skip-issuers --apply
```

**6. Register federation (hub → cluster).** So marin/marin-dev can route jobs in.
Add an address-only peer entry (no key — the cluster already trusts the hubs via
its own `auth.federation_peers`) under `peers:` in both `lib/iris/config/marin.yaml`
and `lib/iris/config/marin-dev.yaml`:

```yaml
  <cluster>:
    controller_address: https://iris-cw-<cluster>.oa.dev
```

**7. Wire finelog (roll the hub first).** Until this lands the controller logs to
an in-process store (lost on restart).

```bash
# Mint the forwarding key by hand (Ed25519); keep the printed PUBLIC half.
openssl genpkey -algorithm ed25519 -out /tmp/<cluster>.pem
openssl pkey -in /tmp/<cluster>.pem -pubout
gcloud secrets create finelog-<cluster>-signing-key --project=<project> \
  --replication-policy=automatic --labels=component=finelog,purpose=forwarding
gcloud secrets versions add finelog-<cluster>-signing-key --project=<project> \
  --data-file=/tmp/<cluster>.pem
shred -u /tmp/<cluster>.pem
```

- Add a `- cluster: <cluster>` entry with that **public** key under the `jwt`
  layer of `lib/finelog/config/marin.yaml`, then **roll the hub before the
  sender** — a sender whose key the hub doesn't yet trust gets 401:
  `uv run finelog deploy restart marin`.
- Create `lib/finelog/config/<cluster>.yaml` (copy an existing one; set `name`,
  `remote_log_dir`, `kube_context`, `object_storage_endpoint`,
  `forwarding.cluster`, and `forwarding.signing_key`), then deploy the sender:
  `uv run finelog deploy up <cluster> --no-build` (the default `--build`
  recompiles the Rust image first). finelog archives to **Cloudflare R2**
  (`s3://marin-na/finelog/<cluster>`, the R2 `object_storage_endpoint`),
  authenticated by `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` in the deploy
  shell — *not* the CoreWeave `CW_KEY_*` keys iris uses for its own
  `marin-us-east-*` state and task storage. Pointing `remote_log_dir` at a
  CoreWeave bucket with R2 keys (or vice-versa) fails the archive with
  `403 InvalidAccessKeyId`.
- Add `finelog: { config: <cluster> }` to the iris config and `cluster start`
  again to pick it up. CI enforces the pairing: a sender config cannot merge
  until some hub's `jwt` layer names its cluster.

**8. Verify node assumptions before trusting multi-host NCCL:**
`NCCL_SOCKET_IFNAME` (the host ethernet PF carrying the node IP — same SKU has
different PCI names per region, e.g. `enp157s0np0` on US-EAST-02A vs `enp90s0np0`
on RNO2A; check with a job running `ls /sys/class/net`), and scale-group
`cpu`/`ram`/`disk` against `kubectl get node -o
jsonpath={.status.allocatable}`.

**9. Smoke:** a CPU hello-world, an 8-GPU `jax.devices()` job, then the multinode
grug smoke (below).

### Connecting

Preferred: use `--cluster=NAME` so Iris opens and closes the controller tunnel:

```bash
iris --cluster=cw-rno2a job logs /runner/my-job
iris cluster list
```

`--cluster=NAME` resolves to a config under `lib/iris/config/` and opens a
`kubectl port-forward` to the controller service. This path requires the
`iris[controller]` extras (`kubernetes`). Without them,
auto-tunneled CoreWeave commands fail before connecting:
`ImportError: Install iris[controller] to use CloudK8sService`.

Fallback: manual port-forward if you need a long-lived tunnel:

```bash
kubectl --kubeconfig ~/.kube/coreweave-iris --context <kube_context> \
  port-forward -n <namespace> svc/<service_name> 10000:10000 &
iris --controller-url=http://localhost:10000 ...
```

| Cluster name | Namespace | Service | Config file |
|--------------|-----------|---------|-------------|
| `cw-us-east-02a` | `iris` | `iris-controller-svc` | `cw-us-east-02a.yaml` |
| `cw-rno2a` | `iris` | `iris-controller-svc` | `cw-rno2a.yaml` |
| `ci-coreweave` | `iris-ci` | `iris-ci-controller-svc` | `ci-coreweave.yaml` (CI only) |

### GPU Configs

Marin's `gpu` extra installs the JAX CUDA 13 wheel stack from PyPI. CoreWeave
GPU nodes must expose NVIDIA driver 580 or newer; `nvidia-smi` should report
CUDA 13.x.

The `gpu` extra also pulls the CUDA toolchain wheels (`ptxas`/`nvlink` from
`nvidia-cuda-nvcc`, `libdevice.10.bc` from `nvidia-nvvm`) into the task venv. A
GPU job's setup scripts then expose them (see
`iris.cluster.setup_scripts.cuda_toolchain_setup_script`): the toolchain binaries are
symlinked into the venv's `bin` (already on `PATH` once the venv is activated),
and `libdevice.10.bc` is staged into XLA's default CUDA data dir
(`./cuda_sdk_lib`) and the working directory, where XLA and Mosaic probe.
JAX/Pallas Mosaic GPU kernels therefore compile without per-job
`ptxas`/`nvlink`/`libdevice` setup. The staging is a no-op unless the venv
carries the toolchain, so CPU/TPU jobs and bring-your-own images are untouched.

This staging is appended only to the default setup for a job that requests the
`gpu` extra. A job that supplies its own `setup_scripts` (run verbatim) or
installs JAX another way must stage the toolchain itself — call
`cuda_toolchain_setup_script()` in its setup.

### Grug MoE Multinode Smoke

Multi-host GPU jobs are gang-admitted by Kueue (see the Kueue section above) — no warm-node
preflight or `targetNodes` patching is needed. Submit a small
`experiments.grug.moe.launch_cw_scale` run as the smoke; the driver is a tiny
CPU job that dispatches the GPU gang itself:

```bash
uv run iris --cluster=<cluster> job run \
  --cpu 2 --memory 3GB --extra cpu \
  --job-name grug-moe-2node-smoke \
  -e SCALE_GPU_REPLICAS 2 -e SCALE_HIDDEN_DIM 1024 -e SCALE_NUM_LAYERS 8 \
  -e SCALE_NUM_EXPERTS 16 -e SCALE_TOP_K 2 -e SCALE_BATCH 32 \
  -e SCALE_SEQ_LEN 1024 -e SCALE_STEPS 10 -e RUN_ID <run-id> \
  -- python -m experiments.grug.moe.launch_cw_scale --version dev --run
```

`--version` sets the checkpoint version (required; `dev` to iterate) and `--run` builds it —
without `--run` the launcher prints the lowered plan and exits.

Success signals: every replica enters `initialize_jax` with
`IRIS_NUM_TASKS=<replicas>`, steps complete, a checkpoint commits, and the
parent job exits `JOB_STATE_SUCCEEDED`.

Small-shape caveat: pick `SCALE_HIDDEN_DIM` so `num_heads = hidden_dim/128` is
divisible by `SCALE_EXPERT_AXIS` (default 8); otherwise grug attention fails
with a `ShardingTypeError` (conflicting `@model` shardings). The ferry-style
alternative is `experiments.ferries.canary_ferry` with `CANARY_*` env vars
(`CANARY_ACCELERATOR=gpu`, `CANARY_GPU_TYPE=H100`, ...), which replicates the
model per node instead of sharding across nodes.

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
- **NHC verification pods occupy idle GPU nodes.** CoreWeave's node health checker
  (`cw-hpc-verification` namespace) runs preemptible GPU pods on idle nodes; Kueue
  TAS counts them as fixed usage and cannot preempt non-Kueue pods. Iris evicts
  them itself when it has gang work — list the namespace in
  `kubernetes_provider.preempt_namespaces`.
- **`NCCL_SOCKET_IFNAME` is per-region.** The same GPU SKU exposes different PCI
  interface names in different regions; verify on a live node (see "Bringing up
  a new cluster").
- **A controller restart can CrashLoop on a stale node-local state DB.** The
  controller keeps its SQLite state on node-local NVMe (`storage.local_state_dir`,
  a hostPath), and `cluster controller restart` may reschedule the new pod onto a
  different CPU node. If that node holds a corrupt leftover state DB from an earlier
  stint whose mtime is at least as fresh as the latest remote checkpoint, startup
  trusts it (skips the checkpoint restore) and crashes with `database disk image is
  malformed`. The deploy's post-restart health check catches the CrashLoopBackOff
  and **auto-rolls-back** to the previous image + pre-deploy checkpoint, so the
  cluster stays healthy. Recovery: just re-run the restart — a fresher pre-deploy
  checkpoint makes a stale node restore the clean checkpoint, and landing back on
  the current controller node reuses its good DB. A persistently bad node needs its
  `local_state_dir/db` wiped so startup falls back to the checkpoint.

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
| `region` | string | — | CoreWeave region (e.g. `RNO2A`) |
| `namespace` | string | `iris` | Kubernetes namespace for all resources |
| `kubeconfig_path` | string | — | Only needed when running CLI outside the cluster |
| `kube_context` | string | — | Kubeconfig context to bind every operation to; empty uses the file's current-context |
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

> **Warning — Disk layout**: CoreWeave bare-metal nodes boot a **15 GB RAM disk**
> as the root filesystem, with the node's NVMe RAID (`/dev/md127`, 7.7 TB on CPU
> nodes, 15–31 TB on GPU nodes) mounted at `/mnt/local`. The `cache_dir` must
> point to NVMe (e.g. `/mnt/local/iris-cache`). Using the default root path will
> fill the RAM disk immediately and cause Pod eviction.
>
> Only `cache_dir` needs this: the kubelet root is on the NVMe, so task `emptyDir`
> volumes (`/app`, `/tmp`) and container writable layers already land there.

### Task storage layout

Task pods use no PVCs — nothing on the task path touches the `shared-vast`
(distributed) storage class, which backs only the controller state and finelog
caches. Every task volume is node-local NVMe:

| Container path | Volume | Lifetime |
|----------------|--------|----------|
| `/app`, `/tmp` | `emptyDir` (kubelet root) | Pod |
| `/uv/cache`, `/hf/cache`, `/cargo` | `hostPath` under `cache_dir` | Node |
| `/dev/shm` | `emptyDir` (memory) | Pod |

Iris points `UV_CACHE_DIR`, `HF_HUB_CACHE`, and `CARGO_HOME` at those
`hostPath` mounts for every task, including tasks that bring their own image, so
wheels and model weights are fetched once per node rather than once per task.
`HF_HOME` is deliberately not among them. It holds the submitter's `HF_TOKEN`, so
it stays under the pod's own `$HOME` (`iris job run` defaults it to
`~/.cache/huggingface`) rather than a directory every task on the node can read.
`HF_HUB_CACHE` covers the part worth sharing: the content-addressed blobs.

The `cache_dir` tree is never pruned; it grows until the node is rebuilt or an
operator clears it. Watch it on long-lived reserved fleets:

```bash
kubectl exec -n iris <pod> -c task -- du -sh /uv/cache /hf/cache
```

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

| Instance Type | GPUs | vCPUs | RAM | Disk | Use Case |
|---------------|------|-------|-----|------|----------|
| `gd-8xh100ib-i128` | 8x H100 | 128 | 2 TB | — | GPU training (primary) |
| `cd-gp-a192-genoa` | none | 192 | 1.5 TB | 7.68 TB | Controller / CPU tasks (US-EAST-02A) |
| `turin-gp-l` | none | 192 | 1.5 TB | 29 TB | Controller / CPU tasks (RNO2A) |

The console capacity view shows display labels, not instance types (e.g.
"turin-gp-l4" for `turin-gp-l`); the NodePool webhook accepts any string on
dry-run, and only the async NodePool controller sets `Validated=False` on a bad
SKU. Probe a SKU with a `minNodes: 0, maxNodes: 0` pool before trusting it.

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
Pod uses in-cluster ServiceAccount auth for all kubectl operations. It requests
16 CPU and 64Gi memory with a memory limit but no CPU limit, so it runs Burstable
(not BestEffort, not Guaranteed): it is never CFS-throttled and can burst onto
spare node cores during reconcile-loop spikes, while memory stays capped to
protect the node. The liveness/readiness probes use a 10s timeout and tolerate 6
failures so a busy-but-alive controller is not liveness-killed mid-reconcile
under a large tokenize fan-out (issue #6944).

Cost note: the smallest CoreWeave CPU instance (`cd-gp-i64-erapids`, 64 vCPU,
512 GB RAM) is overprovisioned for the controller. CoreWeave does not offer
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
| `KUBECONFIG` | Overrides the config's `kubeconfig_path` (file only — the config's `kube_context` still binds the context) |
| `CW_KEY_ID` | S3/CoreWeave Object Storage access key (required if storage uses `s3://`) |
| `CW_KEY_SECRET` | S3/CoreWeave Object Storage secret key |
| `CW_ACCESS_KEY_ID` | CoreWeave Object Storage key ID |
| `CW_SECRET_ACCESS_KEY` | CoreWeave Object Storage secret key |

### Auto-injected into worker and task Pods

| Variable | Source | Description |
|----------|--------|-------------|
| `IRIS_WORKER_NODE_NAME` | Downward API (`spec.nodeName`) | Kubernetes node name |
| `IRIS_POD_NAMESPACE` | Downward API (`metadata.namespace`) | Pod's namespace |
| `IRIS_POD_NAME` | Downward API (`metadata.name`) | Pod's name |
| `IRIS_POD_UID` | Downward API (`metadata.uid`) | Pod's UID |
| `IRIS_SERVICE_ACCOUNT_NAME` | Platform | ServiceAccount for task Pods (set when `runtime: kubernetes`) |
| `AWS_ACCESS_KEY_ID` | `envFrom` | From the `iris-task-env` Secret |
| `AWS_SECRET_ACCESS_KEY` | `envFrom` | From the `iris-task-env` Secret |
| `AWS_ENDPOINT_URL` | `envFrom` | From `iris-task-env`; derived from `object_storage_endpoint` |
| `AWS_REGION` / `AWS_DEFAULT_REGION` | `envFrom` | From `iris-task-env`; `auto` for CoreWeave Object Storage endpoints |
| `FSSPEC_S3` | `envFrom` | From `iris-task-env`; JSON-encoded fsspec S3 config (endpoint + addressing style) |
| `MARIN_PREFIX` | `defaults.task_env` (cluster config) | Preset to `s3://marin-us-east-02a/marin` on both CoreWeave clusters |

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
3. Creates `KubernetesRuntime` (reads `IRIS_SERVICE_ACCOUNT_NAME` from
   environment; S3 credentials arrive via `envFrom` on the `iris-task-env` Secret)
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
    coscheduling=CoschedulingConfig(group_by="leafgroup"),
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
| `iris-task-env` Secret | S3 object storage auth + operator-injected env (`defaults.inject_env`) | `start_controller()` via `ensure_task_env_secret()`, from `CW_KEY_ID` / `CW_KEY_SECRET` + the configured `object_storage_endpoint` |
| `iris-cluster-config` ConfigMap | Cluster config for controller and workers | `start_controller()` |
| In-cluster ServiceAccount token | kubectl calls from controller Pod | Auto-mounted by Kubernetes |

### Operator-managed

| Resource | Purpose | How to Obtain |
|----------|---------|---------------|
| CoreWeave API token | kubeconfig auth | Console > Tokens > Create Token |
| Kubeconfig file | Operator's kubectl access | Console > Tokens > Download Kubeconfig |
| CoreWeave Object Storage access key | S3-compatible access to CoreWeave buckets | Console > Object Storage > Access Keys |

The `kubeconfig_path` / `kube_context` config fields are only needed when
running the CLI **outside** the cluster (e.g., `iris cluster start` from a
laptop). Inside the cluster, Pods use in-cluster auth automatically (both
fields are stripped from the `iris-cluster-config` ConfigMap).

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

A pod evicted for `ephemeral-storage` while `cache_dir` is correct is a different
fault: that limit comes from the task's own `disk` resource request and covers
only `emptyDir` plus the container layer, not the `hostPath` caches. Either the
task is writing large files under `/app` or `/tmp`, or it is writing to a cache
path Iris does not mount (check `env | grep -iE 'CACHE|HF_'` against the task
storage layout above) and needs a larger `disk` request.

## 17. References

- [CoreWeave CKS Introduction](https://docs.coreweave.com/docs/products/cks)
- [CKS Cluster Creation](https://docs.coreweave.com/docs/products/cks/clusters/create)
- [API Access Tokens and Kubeconfig](https://docs.coreweave.com/docs/products/cks/auth-access/manage-api-access-tokens)
- [CoreWeave Node Pools](https://docs.coreweave.com/docs/products/cks/nodes/nodes-and-node-pools)
- [CoreWeave Autoscaling](https://docs.coreweave.com/docs/products/cks/nodes/autoscaling)
- [CoreWeave GPU Instances](https://docs.coreweave.com/docs/platform/instances/gpu-instances)
- [CoreWeave Observe (Managed Grafana)](https://docs.coreweave.com/docs/observability/managed-grafana)
- [CoreWeave AI Object Storage: Set endpoints](https://docs.coreweave.com/products/storage/object-storage/using-object-storage/configure-endpoints)
- [CoreWeave AI Object Storage: Manage objects](https://docs.coreweave.com/products/storage/object-storage/using-object-storage/manage-objects)
- [CoreWeave Terraform Provider](https://docs.coreweave.com/docs/products/cks/terraform/about)

### Source files

| File | Description |
|------|-------------|
| `lib/iris/src/iris/providers/k8s/coreweave.py` | CoreWeave platform implementation (includes `ensure_rbac()`) |
| `lib/iris/src/iris/cluster/runtime/kubernetes.py` | KubernetesRuntime (Pod-per-task) |
| `lib/iris/src/iris/providers/k8s/service.py` | Kubectl CLI wrapper |
| `lib/iris/config/examples/coreweave.yaml` | Example cluster config |
| `lib/iris/AGENTS.md` | CoreWeave integration notes for agents |
