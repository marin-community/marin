# IaC for Marin — Spec

Contract layer for [design.md](design.md). Pins: the `provisioning:` config schema,
the `infra/iac/` layout, the Pulumi component surface, the Iris cede/verify change,
new error types, migration/failure semantics, and an explicit out-of-scope list.
CoreWeave is fully specified; the GCP components are named (committed, sequenced after)
and their ownership boundary sketched, but not signed here.

**Config namespaces (read first).** Marin has two config files per CoreWeave cluster:

- `config/coreweave.yaml` — per-*cloud* rigging/data config (`region_buckets`, keyed
  `iris: coreweave`); shared by all CW clusters.
- `lib/iris/config/<cluster>.yaml` — per-*cluster* Iris config (`scale_groups`,
  `kubernetes_provider`, `platform.coreweave`, `controller.coreweave`).

The new `provisioning:` section goes in the **per-cluster** file
(`lib/iris/config/<cluster>.yaml`), because it is per-cluster (one CKS cluster, one
NodePool set) and must sit alongside the `scale_groups` it is derived against. Both
`load_provisioning(cluster)` and the Iris config loader resolve the same `<cluster>`
key to that one file via the existing iris config search path.

---

## 1. `provisioning:` config schema

A new **provider-discriminated** section in `lib/iris/config/<cluster>.yaml`, extending the schema the [cluster-admin-unification design](../2026-06-23_cluster_admin_unification.md) introduces (`provisioning.gcp`); this design adds the `coreweave` variant. Typed as pydantic.
**Canonical home: `infra/iac/src/iac/config.py`** — this design operationalizes the
section; `marin-cluster admin` imports the models from here. If `lib/cluster` lands its
`ClusterConfig`, it embeds these by import, not re-definition.

**Single-source rules (load-bearing).** `provisioning.coreweave` carries only cluster-level
facts *absent* from the Iris config. Everything already in the Iris config is read from there,
never re-declared:

- **NodePools** derive from `scale_groups` (§3 `derive_nodepools`).
- **Namespace** derives from `kubernetes_provider.namespace` — so RBAC namespace and the Kueue
  webhook scope are structurally one value (the webhook-deadlock invariant is then guaranteed,
  not merely validated).
- **ClusterQueue name** derives from `kubernetes_provider.kueue.cluster_queue` (the queue Iris
  binds its LocalQueue to). IaC creates the queue Iris expects; the name is not duplicated.

```python
# infra/iac/src/iac/config.py
from enum import StrEnum
from pydantic import BaseModel, Field

class Provider(StrEnum):
    COREWEAVE = "coreweave"
    GCP = "gcp"

class CksClusterSpec(BaseModel):
    """The CoreWeave CKS cluster object (coreweave_cks_cluster)."""
    name: str                       # e.g. "marin-gpu"
    zone: str                       # e.g. "US-EAST-02A"
    vpc: str | None = None          # existing VPC name; None => provider default
    import_existing: bool = False   # True => adopt a console-created cluster (§4 migration)

class KueueProvSpec(BaseModel):
    """Cluster-scoped Kueue objects owned by IaC (KueueAddon). cluster_queue and pod_namespace
    are NOT here — they derive from the Iris config (single-source rules above)."""
    resource_flavor: str            # e.g. "cw-ib"
    topologies: list[str]           # Topology CRs, e.g. ["infiniband", "multinode-nvlink-ib"]

class IngressSpec(BaseModel):
    """Traefik + cert-manager + ACME issuers (TraefikAddon)."""
    ingress_class: str = "traefik"
    acme_email: str
    cluster_issuers: list[str]      # e.g. ["letsencrypt-http01-staging", "letsencrypt-http01-prod"]

class RbacSpec(BaseModel):
    """Controller RBAC ceded from ensure_rbac(). namespace derives from the Iris config."""
    service_account: str = "iris-controller"

class BucketSpec(BaseModel):
    name: str                       # e.g. "marin-us-east-02a"
    region: str                     # e.g. "US-EAST-02A"

class ObjectStorageSpec(BaseModel):
    """Buckets + access key(s). Bucket *lifecycle rules* are OUT OF SCOPE (§6)."""
    buckets: list[BucketSpec] = Field(default_factory=list)
    access_key_secret_ref: str | None = None   # gcp-secret://… ; None => keys managed out of band

class CoreweaveProvisioning(BaseModel):
    region: str
    cluster: CksClusterSpec
    kueue: KueueProvSpec
    ingress: IngressSpec
    rbac: RbacSpec = RbacSpec()
    object_storage: ObjectStorageSpec = ObjectStorageSpec()

class ProvisioningConfig(BaseModel):
    """Top-level `provisioning:` section. Exactly one provider block is populated."""
    provider: Provider
    coreweave: CoreweaveProvisioning | None = None
    # gcp: GcpProvisioning | None = None   # defined by admin-unification; ported after CoreWeave

def load_provisioning(cluster: str) -> ProvisioningConfig:
    """Load and validate `provisioning:` from lib/iris/config/<cluster>.yaml.

    Resolves the file via the existing iris config search path (same resolution the Iris
    cluster-config loader uses for the same `cluster` key). Raises pydantic.ValidationError on
    a malformed section, and ValueError if `provider` is set without its matching block. No
    namespace/queue cross-checks are needed — those values are single-sourced from the Iris
    config, not duplicated here.
    """
```

Example addition to `lib/iris/config/cw-us-east-02a.yaml` (values from that same file's existing
`scale_groups` / `kubernetes_provider`):

```yaml
provisioning:
  provider: coreweave
  coreweave:
    region: US-EAST-02A
    cluster: { name: marin-gpu, zone: US-EAST-02A }
    kueue:
      resource_flavor: cw-ib
      topologies: [infiniband, multinode-nvlink-ib]
    ingress:
      ingress_class: traefik
      acme_email: ops@oa.dev
      cluster_issuers: [letsencrypt-http01-staging, letsencrypt-http01-prod]
    rbac: { service_account: iris-controller }
    object_storage:
      buckets:
        - { name: marin-us-east-02a, region: US-EAST-02A }
# namespace (iris) comes from kubernetes_provider.namespace;
# cluster_queue (iris-cq) comes from kubernetes_provider.kueue.cluster_queue — not repeated here.
```

---

## 2. `infra/iac/` layout

```text
infra/iac/
├── Pulumi.yaml                     # project: name=marin-iac, runtime=python
├── Pulumi.cw-us-east-02a.yaml      # stack config: marin-iac:cluster = cw-us-east-02a, secretsprovider = gcpkms://…
├── Pulumi.cw-rno2a.yaml            # stack config: marin-iac:cluster = cw-rno2a, secretsprovider = gcpkms://…
├── Pulumi.gcp.yaml                 # RESERVED (GCP) — not populated this pass
├── __main__.py                     # entry: dispatch by provider → build components
├── pyproject.toml                  # marin-iac package; deps: pulumi, pulumi-kubernetes, marin-iris, marin-rigging
└── src/iac/                        # src-layout package (import name `iac`), mirrors lib/*/src/<pkg>
    ├── config.py                   # §1 schema + load_provisioning
    ├── nodepools.py                # derive_nodepools(IrisClusterConfig) -> list[NodePoolSpec]
    └── coreweave/
        ├── cluster.py              # CoreweaveCluster
        ├── rbac.py                 # IrisRbac
        ├── kueue.py                # KueueAddon
        ├── traefik.py              # TraefikAddon
        └── storage.py              # ObjectStorage
    # gcp/ added when the GCP surface ports (GcpIam, GcpGclbIap, GcpRegistry)
```

**Backend bootstrap (manual, out of IaC).** The GCS state bucket (`gs://marin-iac-state`) and
the KMS key that encrypts secrets are provisioned once, by hand, *before* any `pulumi up` — they
cannot be IaC-managed without a chicken-and-egg. Documented as a two-command bootstrap in the
`infra/iac/` README. `pulumi login gs://marin-iac-state`; `secretsprovider: gcpkms://…` in
`Pulumi.<stack>.yaml`. Both are operator/CI environment, overridable by third-party deployers.

**Who runs apply.** `pulumi up` is **operator-run**; CI runs `pulumi preview` and posts the plan
to the PR (no CI apply in the first cut, because these stacks touch reserved GPU fleets). Full
contract — CI job, WIF auth, IAM split, secrets model, and the later CD phase — is §9.

### Entry contract (`__main__.py`)

```python
# Contract, not implementation.
cluster  = pulumi.Config("marin-iac").require("cluster")       # e.g. "cw-us-east-02a"
prov     = load_provisioning(cluster)                          # §1
iris_cfg = load_iris_cluster_config(cluster)                   # iris.cluster.config loader
if prov.provider is Provider.COREWEAVE:
    ns    = iris_cfg.kubernetes_provider.namespace             # single-source namespace
    cq    = iris_cfg.kubernetes_provider.kueue.cluster_queue   # single-source queue name
    cw    = CoreweaveCluster("cluster", CoreweaveClusterArgs(
                cluster=prov.coreweave.cluster, region=prov.coreweave.region,
                nodepools=derive_nodepools(iris_cfg)))
    k8s   = pulumi_kubernetes.Provider("cw-k8s", kubeconfig=cw.kubeconfig)
    IrisRbac("rbac", IrisRbacArgs(namespace=ns, spec=prov.coreweave.rbac), k8s_provider=k8s)
    KueueAddon("kueue", KueueAddonArgs(namespace=ns, cluster_queue=cq, spec=prov.coreweave.kueue),
               k8s_provider=k8s)
    TraefikAddon("traefik", TraefikAddonArgs(spec=prov.coreweave.ingress), k8s_provider=k8s)
    ObjectStorage("storage", ObjectStorageArgs(spec=prov.coreweave.object_storage))
```

---

## 3. Pulumi component surface (CoreWeave)

`pulumi.ComponentResource` subclasses; args are frozen dataclasses. Each addon takes a
`k8s_provider` built from `CoreweaveCluster.kubeconfig`, so CRD→CR ordering is a real dependency
edge (the Pulumi win over TF's `kubernetes_manifest` two-apply problem).

```python
@dataclass(frozen=True)
class NodePoolSpec:
    """One CoreWeave NodePool, projected from an Iris scale group. Mirrors the manifest Iris
    builds today (controller.py:_ensure_one_nodepool) so a port preserves behavior."""
    name: str                    # normalized {label_prefix}-{scale_group} (RFC1123: lower, _→-)
    instance_type: str           # slice_template.coreweave.instance_type
    min_nodes: int               # buffer_slices * slice_template.num_vms
    max_nodes: int               # max_slices   * slice_template.num_vms
    node_labels: dict[str, str]  # iris scale-group/managed/region labels + worker.attributes (§below)
    system_critical: bool        # min_nodes > 0 → cks.coreweave.cloud/system-critical
    autoscaling: bool = True     # kept True as today; CoreWeave autoscaler owns runtime targetNodes
    # NOTE: targetNodes is NOT part of the spec — the manifest omits it and CoreWeave owns the
    # runtime count. Pulumi declares only the [min,max] envelope + labels.

class CoreweaveCluster(pulumi.ComponentResource):
    """The CKS cluster (coreweave_cks_cluster via the bridged CoreWeave TF provider — or adopted
    via import when cluster.import_existing) plus the reserved NodePools. Reconciles the set of
    IaC-managed NodePools to exactly `args.nodepools`. Each NodePool is declared with
    `ignore_changes=["spec.targetNodes"]` so CoreWeave's autoscaler may move the runtime count
    within [min,max] without Pulumi fighting it — IaC owns the envelope, not the live count.
    Exports `kubeconfig`: from the provider output in create mode, or read from the Iris config's
    `platform.coreweave.kubeconfig_path` in import mode.
    """
    kubeconfig: pulumi.Output[str]
    def __init__(self, name: str, args: CoreweaveClusterArgs,
                 opts: pulumi.ResourceOptions | None = None) -> None: ...

@dataclass(frozen=True)
class IrisRbacArgs:
    namespace: str               # from kubernetes_provider.namespace
    spec: RbacSpec

class IrisRbac(pulumi.ComponentResource):
    """The Namespace, `iris-controller` ServiceAccount, and namespace-qualified ClusterRole +
    ClusterRoleBinding formerly created by ensure_rbac(). ClusterRole verbs reproduce
    ensure_rbac() in controller.py exactly (nodepools; pods, pods/exec, pods/log; nodes;
    configmaps; metrics/pods; poddisruptionbudgets; kueue workloads; priorityclasses). This is
    the contract Iris's verify_prerequisites() checks."""
    def __init__(self, name: str, args: IrisRbacArgs, *, k8s_provider: pulumi.ProviderResource,
                 opts: pulumi.ResourceOptions | None = None) -> None: ...

@dataclass(frozen=True)
class KueueAddonArgs:
    namespace: str               # webhook scope (== rbac namespace, structurally)
    cluster_queue: str           # from kubernetes_provider.kueue.cluster_queue
    spec: KueueProvSpec

class KueueAddon(pulumi.ComponentResource):
    """The `cks-kueue` Helm release, Topology CRs, the cluster-scoped ResourceFlavor and the
    ClusterQueue named `cluster_queue`, plus the controller-manager Configuration. Admission
    webhooks are scoped to `namespace` (managedJobsNamespaceSelector) — an unscoped webhook
    deadlocks node delivery cluster-wide, so this is asserted in tests. Does NOT create the
    namespaced LocalQueue — Iris keeps reconciling that at controller start."""
    def __init__(self, name: str, args: KueueAddonArgs, *, k8s_provider: pulumi.ProviderResource,
                 opts: pulumi.ResourceOptions | None = None) -> None: ...

@dataclass(frozen=True)
class TraefikAddonArgs:
    spec: IngressSpec

class TraefikAddon(pulumi.ComponentResource):
    """Traefik + cert-manager Helm releases and the HTTP-01 Let's Encrypt ClusterIssuers named in
    `spec.cluster_issuers`. cert-manager CRDs are established before the ClusterIssuers apply (a
    dependsOn edge, replacing install_traefik_proxy.py's wait_for_crd loop)."""
    def __init__(self, name: str, args: TraefikAddonArgs, *, k8s_provider: pulumi.ProviderResource,
                 opts: pulumi.ResourceOptions | None = None) -> None: ...

@dataclass(frozen=True)
class ObjectStorageArgs:
    spec: ObjectStorageSpec

class ObjectStorage(pulumi.ComponentResource):
    """CoreWeave AI Object Storage buckets in `spec.buckets` and their access key(s). Bucket
    *lifecycle rules* are NOT managed here (§6). Access-key material is a Pulumi secret
    (encrypted in state) and/or read from `spec.access_key_secret_ref`."""
    def __init__(self, name: str, args: ObjectStorageArgs,
                 opts: pulumi.ResourceOptions | None = None) -> None: ...
```

```python
# infra/iac/src/iac/nodepools.py
def derive_nodepools(config: IrisClusterConfig) -> list[NodePoolSpec]:
    """Project the Iris config's scale_groups onto CoreWeave NodePool specs, byte-compatible with
    the manifests ensure_nodepools() builds today.

    One NodePool per scale group that HAS a coreweave slice_template; groups without one are
    SKIPPED (matching current ensure_nodepools, controller.py:841), not errored. For each:
      name          = _nodepool_name(label_prefix, scale_group)   # lower(), '_'→'-'
      instance_type = slice_template.coreweave.instance_type
      min_nodes     = buffer_slices * slice_template.num_vms
      max_nodes     = max_slices    * slice_template.num_vms
      node_labels   = {iris scale-group label, iris managed label, region} + worker.attributes
      system_critical = min_nodes > 0
    verify_prerequisites (§4) MUST use this same function so expected names match created names.
    """
```

---

## 4. Iris-side change: cede, verify, migrate

File: `lib/iris/src/iris/cluster/platforms/k8s/controller.py` (the K8s controller platform; the
`providers/k8s/coreweave.py` path in the docs is stale).

- **Removed from `start_controller()`** (`controller.py:411`, `:435`): `self.ensure_rbac()` and
  `self.ensure_nodepools(config)`. The methods `ensure_rbac()` (`:669`) and `ensure_nodepools()`
  (`:830`) are **deleted** (no back-compat). The manifest construction in `_ensure_one_nodepool`
  and the `_nodepool_name` logic move into `iac.nodepools` as the single owner of NodePool shape.
- **Added**: `verify_prerequisites(config)`, called where `ensure_rbac()` was.

```python
def verify_prerequisites(self, config: IrisClusterConfig) -> None:
    """Assert IaC-provisioned prerequisites exist before starting the controller.

    Provider-scoped: on CoreWeave it checks presence (not exact spec — see design Open Questions
    on strictness) of the Namespace + `iris-controller` SA + ClusterRole/Binding; one NodePool
    per non-skipped scale group (names from derive_nodepools); the Kueue ClusterQueue (from
    kubernetes_provider.kueue.cluster_queue) and the ResourceFlavor (from
    provisioning.coreweave.kueue.resource_flavor); and the IngressClass (from
    provisioning.coreweave.ingress.ingress_class). On GCP it checks the GCP-arm prerequisites
    (§6) — there are no per-scale-group NodePools there, so the check is a different set, not a
    reused CoreWeave shape. Presence only: it does not assert health (a running Kueue manager) —
    that is `pulumi up`'s success gate, not cluster-start's. Raises PrerequisitesNotProvisionedError
    enumerating every missing object with remediation `cd infra/iac && pulumi stack select
    <cluster> && pulumi up`. Creates nothing.
    """
```

Unchanged in `start_controller()` (still Iris-owned): ConfigMap, `iris-task-env` Secret
(`ensure_task_env_secret`, `:968`), the namespaced LocalQueue (`ensure_kueue_queues`, `:436`),
priority classes (`ensure_priority_classes`, `:437`), the state PVC, the controller Deployment,
the Service.

### Migration of the live clusters (adoption)

`cw-us-east-02a` and `cw-rno2a` already run RBAC + NodePools that Iris created and labelled
`iris-…-managed`. First managed apply must **adopt**, not recreate — a naive apply would
conflict, and a reconcile-delete could deprovision a reserved 256-GPU fleet. Procedure, pinned:

1. Bootstrap state bucket + KMS (§2).
2. `pulumi import` the existing Namespace, RBAC objects, and each NodePool into the stack (an
   `infra/iac/import/<cluster>.sh` mapping k8s object → Pulumi resource address), so state
   reflects reality with **no diff**.
3. `pulumi preview` must then show an **empty plan** (or only additive add-ons: Kueue
   ResourceFlavor/ClusterQueue, Traefik, buckets). A non-empty destructive plan on NodePools is
   a stop-the-line bug.
4. Only after a clean preview, `pulumi up`, then deploy new Iris with `verify_prerequisites`.

`cluster.import_existing` covers the CKS cluster object itself the same way.

---

## 5. New error types

```python
class PrerequisitesNotProvisionedError(InfraError):
    """Raised by verify_prerequisites() when one or more IaC-provisioned prerequisites are absent.
    Message enumerates every missing object and prints the `pulumi up` remediation. Subclass of
    the existing iris InfraError so existing cluster-start handling still catches it."""
```

`load_provisioning` (§1) raises `pydantic.ValidationError` (malformed) and `ValueError`
(provider/block mismatch). No new config-parse exception type.

---

## 6. Failure modes & partial apply

- **Partial apply.** `pulumi up` is transactional per resource, not globally atomic; a failed run
  can leave NodePools up but Kueue half-installed. `verify_prerequisites` is presence-only, so it
  can pass against an unhealthy partial state — therefore the operator contract is: a failed
  `pulumi up` must be re-run to green *before* `iris cluster start`, and CI preview never leaves a
  half-applied stack. The specific hazard (a Kueue webhook installed but manager not Ready)
  deadlocks node delivery; the webhook-namespace scoping (asserted in tests) bounds the blast
  radius to the `iris` namespace regardless.
- **Drift.** CoreWeave autoscaler moving `targetNodes` is *expected* drift, absorbed by
  `ignore_changes` on NodePools; it never shows in `pulumi preview`. Any other drift surfaces in
  preview and is reconciled on the next operator apply.

---

## 7. Out of scope

Reviewers: do **not** push back on these being absent — they are deliberate.

- **R2 / CoreWeave bucket *lifecycle* rules** — `infra/configure_buckets.py` keeps them; no
  first-class provider exists. `ObjectStorage` does buckets + keys only.
- **GCP components** (`GcpIam`, `GcpGclbIap`, `GcpRegistry`) — committed, sequenced after
  CoreWeave; named + ownership-sketched (§4), not signed here. **Approving this design approves
  the GCP port in principle** (all-or-nothing); the GCP component signatures are a follow-on spec.
- **IAP OAuth brand / clients** — externally created, read as inputs (`read_oauth_client`).
- **TPU / worker slice lifecycle** — Iris's autoscaler (class-2, imperative) on all clouds.
- **Iris-owned runtime objects** — ConfigMap, `iris-task-env` Secret, LocalQueue, priority
  classes, controller Deployment + Service stay in `start_controller()`.
- **`marin-cluster admin` CLI wiring.** This spec pins the *invocation surface*: admin drives the
  same `infra/iac` program **in-process via the Pulumi Automation API** (not by shelling to the
  CLI); the acceptance test and operator flow shell `pulumi` directly against the same program.
  The CLI wiring itself lands with admin-unification.
- **CKS cluster-*create* maturity** — if `coreweave_cks_cluster` create proves weak (design Open
  Questions), `import_existing: true` adopts a console-created cluster; the signature is unchanged.

---

## 8. File-path summary

| Piece                                                | Path                                                                              |
| ---------------------------------------------------- | --------------------------------------------------------------------------------- |
| Provisioning schema + loader                         | `infra/iac/src/iac/config.py`                                               |
| NodePool derivation (single owner of NodePool shape) | `infra/iac/src/iac/nodepools.py`                                            |
| Pulumi entry / dispatch                              | `infra/iac/__main__.py`                                                           |
| Project / stacks / backend bootstrap README          | `infra/iac/Pulumi.yaml`, `infra/iac/Pulumi.<cluster>.yaml`, `infra/iac/README.md` |
| Import/adoption scripts                              | `infra/iac/import/<cluster>.sh`                                                   |
| CoreWeave components                                 | `infra/iac/src/iac/coreweave/{cluster,rbac,kueue,traefik,storage}.py`       |
| Iris cede/verify                                     | `lib/iris/src/iris/cluster/platforms/k8s/controller.py`                           |
| New config section                                   | `lib/iris/config/<cluster>.yaml` → `provisioning:`                                |
| GCP components (later)                               | `infra/iac/src/iac/gcp/`                                                    |

---

## 9. Deploy workflow

Two phases; phase 1 is what this design builds.

### Phase 1 — CI preview + manual apply

- **CI (preview only).** A GitHub Actions workflow triggers on PRs touching `infra/iac/**` and
  the cluster configs it reads. It authenticates to GCP via **Workload Identity Federation**
  (keyless OIDC — no service-account JSON in GitHub secrets), runs `pulumi preview` per affected
  stack, and posts the plan as a PR comment (`pulumi/actions`). **CI never runs `pulumi up`.**
- **CI identity** — service account `pulumi-ci@<proj>`, least-privilege for preview:
  - `roles/cloudkms.cryptoKeyDecrypter` on the state key (decrypt state secrets to compute the diff),
  - object write/delete permissions on the `gs://marin-iac-state` lock prefix (e.g. `roles/storage.objectUser` on the `.pulumi/locks` path to acquire/release stack locks),
  - `roles/secretmanager.secretAccessor` on referenced secrets (once `access_key_secret_ref` is wired),
  - a kubeconfig for the target cluster with write permissions (create/patch/update/delete) for the specific resources managed by the IaC stack, which is required for Kubernetes Server-Side Apply (SSA) dry-run validation.
- **Gate.** Reviewers read the posted plan; a NodePool `replace`/`delete` blocks the PR — the
  same gate as §4/§6.
- **Apply.** After review/merge an **operator** runs `pulumi up <stack>` from a trusted
  environment, re-reading the preview. Operators hold `roles/cloudkms.cryptoKeyEncrypterDecrypter`
  + `roles/storage.objectAdmin` (write encrypted state).

### Secrets model

For configuration values specific to a stack/deployment, we use Pulumi's native secrets system:
- Secrets are set locally using the CLI: `pulumi config set --secret <key> <value>`.
- Pulumi encrypts these values using the GCP **KMS** key specified as the `secretsprovider` in `Pulumi.<stack>.yaml`.
- The encrypted ciphertext is stored directly in the `Pulumi.<stack>.yaml` file, which is safe to check into git.
- At runtime, `pulumi up` and `pulumi preview` decrypt these values using the KMS key (requiring `roles/cloudkms.cryptoKeyDecrypter` on the key).

The self-managed GCS backend does per-stack locking, so a concurrent CI preview and operator apply cannot corrupt state.

### Phase 2 — full CD (later, not built now)

Promote apply into CI: `pulumi up` on merge to `main`, gated behind a **GitHub Environment** with
required reviewers — the human checkpoint moves from "operator runs it" to "operator approves the
deploy." Same WIF identity, now granted encrypt; the preview stays the artifact of record.
