# marin-iac

Infrastructure-as-code for the static substrate of Marin clusters, per the design in
[`.agents/projects/iac/`](../../.agents/projects/iac/). Pulumi (Python). **CoreWeave first.**

This is the **minimal cut**: it provisions the ceded RBAC, reserved NodePools, Kueue objects,
and the Traefik/cert-manager/federation-ingress stack for a CoreWeave cluster. The CKS cluster
object itself is out of scope permanently (see `coreweave/cluster.py`); object storage and the
GCP arm's remaining pieces are tracked in [`gaps.md`](../../.agents/projects/iac/gaps.md), which
inventories every remaining prerequisite of a live deployment with the exact landing site for
each.

Stacks: one per cluster, each a `Pulumi.<cluster>.yaml` pointer to the cluster name. CoreWeave
— `cw-us-east-02a`, `cw-us-east-08a` (GB200). GCP — `marin`, which so far declares only the
reserved federation-egress static IPs (`GcpStaticAddresses`, the GCP arm's first slice; its
live adoption is operator-run — see `gaps.md`).

Beyond cluster prerequisites, the `iac` package also carries the reusable *service*
components other `infra/<service>/` Pulumi projects build on: `iac.gcp.cloud_run`
(IAP-gated Cloud Run, used by `infra/grafana`) and `iac.iris` (always-on Iris service
jobs via a `local.Command` around the `iac.iris.deploy` CLI, used by `infra/ducky`).

## What it reads

Everything comes from the per-cluster Iris config (`lib/iris/config/<cluster>.yaml`):

- NodePools derive from `scale_groups` (`iac.nodepools.derive_nodepools`).
- Namespace from `kubernetes_provider.namespace`; ClusterQueue name from
`kubernetes_provider.kueue.cluster_queue`.
- The residual cluster facts (CKS cluster name, ResourceFlavor, ACME issuers, buckets) from the
new `provisioning:` section in that same file. Iris carries `provisioning:` as an opaque dict;
`iac.config` owns the typed schema. (The package is `infra/iac/src/iac/`, imported as
`iac` — a `src/<pkg>` layout mirroring `lib/*/src/<pkg>`.)



## Prerequisites

- The [Pulumi CLI](https://www.pulumi.com/docs/install/) (`brew install pulumi/tap/pulumi`).
- The repo's uv venv, synced so `iris`, `rigging`, `pulumi`, `pulumi_kubernetes` all import.
`marin-iac` is a member of the root uv workspace, so its deps live in the shared repo
`.venv`. The pulumi SDK sits behind the package's `deploy` extra (it is deploy-only
tooling, kept out of the plain workspace sync so Iris job venvs don't pull the pulumi
providers), so pass `--extra deploy` when preparing a preview/up:
  ```bash
  uv sync --all-packages --extra deploy    # run from the repo root or anywhere in the tree
  ```
- For the k8s dry-run: the CoreWeave kubeconfig at the path in the cluster's
`platform.coreweave.kubeconfig_path` (read access to the live cluster).
- The [Helm CLI](https://helm.sh/docs/intro/install/) (`brew install helm`), with the CoreWeave
chart repo registered locally:
  ```bash
  helm repo add coreweave https://charts.core-services.ingress.coreweave.com
  helm repo update coreweave
  ```
  Required before any `pulumi preview`/`up` on a CoreWeave stack. `TraefikAddon`'s Traefik and
  cert-manager `Release`s omit Pulumi's `repository_opts` (a workaround for an upstream Pulumi
  bug — see `.agents/projects/iac/gaps.md`'s "Pulumi Helm chart resolution"), so Pulumi resolves
  those two charts through this local Helm config. Without it, `pulumi preview` fails with
  `"coreweave" is not a valid chart repository`. Re-run `helm repo update coreweave` after a
  `TRAEFIK_VERSION`/`CERT_MANAGER_VERSION` bump in `src/iac/coreweave/traefik.py` — a stale local
  index won't list the new version. This is per-machine and per-CI-runner (runners are ephemeral).
  gaps.md records the options for folding this step into `pulumi up` itself (a `command.local.run`
  invoke, or vendored charts); this PR keeps it as the documented prerequisite here.



## Running a preview

The committed `Pulumi.<cluster>.yaml` is only the stack's **config**, not the stack itself —
you create the stack once with `pulumi stack init` before it can be selected.

```bash
cd infra/iac

# 1. one-time: pick a state backend. --local keeps state in a local file (good for
#    experimentation); production uses gs://marin-iac-state (see spec.md §2 backend bootstrap).
pulumi login --local

# 2. the secrets provider is a passphrase for local, throwaway previews (production uses a GCP
#    KMS key — see "Backend bootstrap" below). An empty passphrase is fine here:
export PULUMI_CONFIG_PASSPHRASE=""

# 3. one-time: create the stack. This reads the committed Pulumi.cw-us-east-02a.yaml.
pulumi stack init cw-us-east-02a
#    (on later runs, just: pulumi stack select cw-us-east-02a)

# 4. preview
pulumi preview
```

Against an empty state this plans a Namespace, the `iris-controller` ServiceAccount, the
`iris-controller-iris` ClusterRole + ClusterRoleBinding, and one `NodePool` per CoreWeave
scale group (`cw-use02a-cpu-genoa`, `cw-use02a-h100-8x`) — all as **creates**, because
Pulumi's state doesn't yet know these already exist on the live cluster. To see the *real*
diff, adopt them first (next section).

## Adoption check — safe, read-only recon

The running CW clusters already hold Iris-created RBAC + NodePools, so before a first apply we
need to see whether Pulumi would *adopt* them (no-op/update) or *recreate* them (a NodePool
replace = deprovisioning the reserved fleet). This is read-only — the only command that mutates
the cluster is `pulumi up`, which you must **not** run until the preview is clean (`spec.md §4`).

Setting `marin-iac:import=true` makes the program stamp `import_=<live id>` on every resource, so
`pulumi preview` shows the true adoption diff — correctly parented and on the real `cw-k8s`
provider. Preview writes nothing; it reports per resource whether adoption is a no-op, an update,
or a **replace**. Do this against a **local** backend so state stays in a throwaway local file.

```bash
pulumi login --local
export PULUMI_CONFIG_PASSPHRASE=""
export KUBECONFIG=~/.kube/coreweave-iris       # read access to the live cluster
pulumi stack select cw-us-east-02a

pulumi config set marin-iac:import true
pulumi preview                                 # read the adoption diff
pulumi config set marin-iac:import false       # (or `pulumi config rm marin-iac:import`)
```

Reading it: **no-change / update-in-place** on a NodePool → safe to adopt; **replace / delete** on
a NodePool → **stop.** A NodePool replace deprovisions the bare-metal fleet (a reserved 256-GPU
cluster); reconcile the program to match reality before going further, and never `pulumi up`
through a destructive NodePool diff.

Two caveats when reading diffs: (1) the live RBAC/NodePools were applied by Iris under the
server-side-apply field manager `iris`, so Pulumi's SSA dry-run reports a **field conflict**
(e.g. `conflict with "iris": .rules`) unless ownership is forced. The k8s provider declares
`enable_patch_force=True` directly (`__main__.py`) rather than relying on the
`PULUMI_K8S_ENABLE_PATCH_FORCE` env var, so this is automatic — still just a dry-run (nothing is
written); it lets Pulumi compute the merged result. Taking ownership for real happens only at
`pulumi up`, which has a hard ordering rule — see **Production adoption** below. (2) Fields the
live object carries but the program doesn't declare won't show as diffs (Pulumi only manages
declared fields).

To discard the recon entirely, `pulumi stack rm cw-us-east-02a` (local backend) — it drops the
state file without touching the cluster.

## Production adoption (GCS state)

Recon (above) is all dry-run and safe to run anytime. The **real** adoption — `pulumi up` writing
the durable state in `gs://marin-iac-state` — is gated on an ordering constraint you cannot skip.

**The ordering constraint: cede *then* adopt.** Iris applies RBAC/NodePools with
`server_side_apply(field_manager="iris", force_conflicts=True)`
(`[service.py:257](../../lib/iris/src/iris/cluster/platforms/k8s/service.py#L257)`). Because it
**forces conflicts**, adopting while old Iris is still deployed does *not* fail — it flaps: every
`pulumi up` takes field ownership, and the next `iris cluster start` force-reclaims it, forever.
No outage (the specs match, so values don't change), but drift detection becomes meaningless and
the two managers ping-pong `managedFields` on every controller restart. So:

1. **Ship the cede first.** Land + roll out the Iris change that deletes `ensure_rbac()` and
  `ensure_nodepools()` from `start_controller()` (spec §4), so no running controller applies
   these resources anymore.
2. **Then adopt.** With no `iris` manager re-applying, Pulumi force-takes ownership once and stays
  the sole owner — clean.

Doing it in the other order (or adopting while any old controller can still restart) reintroduces
the flap until the ceded Iris is deployed everywhere.

### Backend bootstrap (one-time, provisioned)

The shared backend is a GCS bucket + a GCP KMS secrets provider, both in `hai-gcp-models`:

- **State bucket** `gs://marin-iac-state` (us-central1, uniform bucket-level access, versioned).
- **Secrets provider: a GCP KMS key**,
`gcpkms://projects/hai-gcp-models/locations/us-central1/keyRings/marin-iac-keyring/cryptoKeys/marin-iac-key`.
Unlike a passphrase, KMS access is asymmetric: CI holds only
`roles/cloudkms.cryptoKeyDecrypter` (can decrypt to compute a preview diff, cannot write new
secrets), operators hold `roles/cloudkms.cryptoKeyEncrypterDecrypter` (can `pulumi up`) — see
`spec.md §9`. No `PULUMI_CONFIG_PASSPHRASE` is set against this backend; decryption is
authorized by each caller's own GCP credentials (`gcloud auth application-default login`
locally, WIF in CI), so IAM on the key is the only access control. Each stack's
`secretsprovider` URI is recorded in its committed `Pulumi.<stack>.yaml` at `stack init`.

Re-provisioning from scratch (already done once):

```bash
gcloud kms keyrings create marin-iac-keyring --project=hai-gcp-models --location=us-central1
gcloud kms keys create marin-iac-key --project=hai-gcp-models --location=us-central1 \
  --keyring=marin-iac-keyring --purpose=encryption
gcloud storage buckets create gs://marin-iac-state --project=hai-gcp-models \
  --location=us-central1 --uniform-bucket-level-access --public-access-prevention
gsutil versioning set on gs://marin-iac-state
```

Creating the key only grants **admin** over it (rotate/destroy/set IAM policy); it does not
grant encrypt/decrypt. `roles/cloudkms.admin` alone cannot run `pulumi stack init`/`up` — you'll
hit `PermissionDenied: cloudkms.cryptoKeyVersions.useToEncrypt`. Each operator needs the
encrypt/decrypt role granted explicitly, on the key:

```bash
gcloud kms keys add-iam-policy-binding marin-iac-key \
  --keyring=marin-iac-keyring --location=us-central1 --project=hai-gcp-models \
  --member="user:<operator email>" \
  --role="roles/cloudkms.cryptoKeyEncrypterDecrypter"
```

CI's service account gets the narrower `roles/cloudkms.cryptoKeyDecrypter` instead (§9 above).

**The adoption run itself** (per cluster, one-time):

```bash
pulumi login gs://marin-iac-state
pulumi stack init cw-us-west-04a \
  --secrets-provider="gcpkms://projects/hai-gcp-models/locations/us-central1/keyRings/marin-iac-keyring/cryptoKeys/marin-iac-key"
#    (on later runs, just: pulumi stack select cw-us-east-02a — the secretsprovider is already
#    recorded in the committed Pulumi.<stack>.yaml)
export KUBECONFIG=~/.kube/coreweave-iris

pulumi config set marin-iac:import true
pulumi preview          # gate: every resource `import` + no-op/update; ANY NodePool replace/delete → STOP
pulumi up               # adopts live resources into GCS state; does not recreate them
pulumi config rm marin-iac:import               # import_ is ONE-SHOT — see below
```

`marin-iac:import` **is one-shot.** It tells Pulumi "adopt the existing object on the next `up`."
Once adopted, the resource is in state; leaving the flag on makes the *next* `up` try to import an
already-managed resource and error. Flow is strictly: set true → `up` once → set false.

> **`cw-us-west-04a` has no live Traefik/cert-manager to adopt** (see the cluster's
> `provisioning.coreweave.ingress` comment in `lib/iris/config/cw-us-west-04a.yaml`) — the
> `TraefikAddon` resources are net-new there, not a pre-existing object with an ID to import.
> `marin-iac:import` is a single program-wide flag (`__main__.py`'s `adopt`), so the blanket
> `pulumi up` above would try to `import_` a Helm release/ClusterIssuer/Ingress that was never
> created and fail. For this cluster, scope the import pass to the three components that do
> pre-exist and leave `traefik` out of it:
>
> ```bash
> pulumi config set marin-iac:import true
> pulumi preview --target 'urn:pulumi:cw-us-west-04a::marin-iac::marin:coreweave:CoreweaveCluster::cluster' \
>                 --target 'urn:pulumi:cw-us-west-04a::marin-iac::marin:coreweave:IrisRbac::rbac' \
>                 --target 'urn:pulumi:cw-us-west-04a::marin-iac::marin:coreweave:KueueAddon::kueue'
> pulumi up --target 'urn:pulumi:cw-us-west-04a::marin-iac::marin:coreweave:CoreweaveCluster::cluster' \
>           --target 'urn:pulumi:cw-us-west-04a::marin-iac::marin:coreweave:IrisRbac::rbac' \
>           --target 'urn:pulumi:cw-us-west-04a::marin-iac::marin:coreweave:KueueAddon::kueue'
> pulumi config rm marin-iac:import
> pulumi up               # normal run, adopt=false now — creates TraefikAddon's stack fresh
> ```
>
> A cluster whose Traefik/cert-manager genuinely does pre-exist (the common case for the other
> CoreWeave clusters) uses the untargeted flow above unmodified.

**After adoption:** protect the fleet — add `protect=true` (or `retainOnDelete=true`) to the
NodePools so an accidental `pulumi destroy`/rename can't deprovision the reserved bare-metal
nodes. Then normal ops are plain `pulumi preview`/`up` with the flag off; state lives in GCS.

> **Keep local recon off the production stack.** Do recon under a throwaway stack name (e.g.
> `cw-us-east-02a-recon` on `--local`) so the production `cw-us-east-02a` stack config keeps its
> `secretsprovider: gcpkms://…` pointer — not a local passphrase.

> **Do not** `pulumi up` **against a live cluster until the cede is deployed.** Recon (dry-run) is
> always fine; the real adoption is ordered. See `spec.md §4`.
