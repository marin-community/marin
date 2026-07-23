# Pulumi infrastructure

Infrastructure-as-code for the static substrate of Marin clusters, per the design in
[`.agents/projects/iac/`](../../.agents/projects/iac/). Pulumi (Python). **CoreWeave first.**

This is the **minimal cut**: it provisions RBAC, reserved NodePools, Kueue objects, the
Traefik/cert-manager/federation-ingress stack, and configured Cloudflare CNAMEs for a CoreWeave
cluster. It is the sole owner of these resources — Iris no longer provisions any of them
(`verify_prerequisites()` in
[`k8s/controller.py`](../../lib/iris/src/iris/cluster/platforms/k8s/controller.py) only checks
presence and fails with a `pulumi up` remediation if something is missing). The CKS cluster
object itself is not yet managed by Pulumi (see `coreweave/cluster.py` and "Future work" below).
The cluster project retains the `marin-iac` Pulumi name so this directory's move from
`infra/iac` did not change resource URNs or require a state migration.

Stacks: one per cluster, each a `Pulumi.<cluster>.yaml` pointer to the cluster name. CoreWeave —
`cw-us-west-04a`, `cw-us-east-02a`, `cw-rno2a`, `cw-us-east-08a` (GB200), all adopted into
`gs://marin-iac-state`. GCP — `marin`, which so far declares only the reserved
federation-egress static IPs (`GcpStaticAddresses`, the GCP arm's first slice).

Beyond cluster prerequisites, the `iac` package also carries the reusable *service* components
other `infra/<service>/` Pulumi projects build on: `iac.gcp.cloud_run` (IAP-gated Cloud Run,
used by `infra/grafana`) and `iac.iris` (always-on Iris service jobs via a `local.Command`
around the `iac.iris.deploy` CLI, used by `infra/ducky` and `infra/xprof`).

GitHub organization and repository resources live in the independent
[`github`](github/README.md) Pulumi project. Its stack YAML declares existing Actions secrets
while their values remain outside Pulumi.

## What it reads

Everything comes from the per-cluster Iris config (`lib/iris/config/<cluster>.yaml`):

- NodePools derive from `scale_groups` (`iac.nodepools.derive_nodepools`).
- Namespace from `kubernetes_provider.namespace`; ClusterQueue name from
  `kubernetes_provider.kueue.cluster_queue`.
- The residual cluster facts (CKS cluster name, ResourceFlavor, ACME issuers, and optional
  federation CNAME) from the `provisioning:` section in that same file. Iris carries
  `provisioning:` as an opaque dict;
  `iac.config` owns the typed schema. (The package is `infra/pulumi/src/iac/`, imported as
  `iac` — a `src/<pkg>` layout mirroring `lib/*/src/<pkg>`.)

## Operations

### First-time setup

- **Pulumi CLI**: [install](https://www.pulumi.com/docs/install/) (`brew install pulumi/tap/pulumi`).
- **Python deps**: `marin-iac` is a member of the root uv workspace, but the Pulumi SDK sits
  behind its `deploy` extra (kept out of the plain workspace sync so Iris job venvs don't pull
  the Pulumi providers):
  ```bash
  uv sync --all-packages --extra deploy    # run from the repo root or anywhere in the tree
  ```
- **Helm CLI**: [install](https://helm.sh/docs/intro/install/) (`brew install helm`), then
  register the CoreWeave chart repo:
  ```bash
  helm repo add coreweave https://charts.core-services.ingress.coreweave.com
  helm repo update coreweave
  ```
  Required before any `pulumi preview`/`up` on a CoreWeave stack: `TraefikAddon`'s Traefik and
  cert-manager `Release`s resolve their charts through this local Helm config rather than
  Pulumi's own `repository_opts` (a workaround for an upstream Pulumi bug that intermittently
  fails Helm chart resolution — see "Future work"). Without it, `pulumi preview` fails with
  `"coreweave" is not a valid chart repository"`. Re-run `helm repo update coreweave` after a
  `TRAEFIK_VERSION`/`CERT_MANAGER_VERSION` bump in `src/iac/coreweave/traefik.py`. This is
  per-machine and per-CI-runner (runners are ephemeral).
- **GCP credentials**: `gcloud auth application-default login`. Decrypting stack secrets is
  authorized by your own GCP credentials against the shared KMS key — see "Backend" below for
  getting the IAM role granted.
- **Cloudflare credential**: stacks with `provisioning.coreweave.federation_dns` read the
  DNS-only Cloudflare token straight from Secret Manager (`cloudflare-oa-dns-token` in
  `hai-gcp-models`, the same one `infra/grafana` uses) under your GCP credentials above — no
  separate export needed, just `roles/secretmanager.secretAccessor` on that secret.
- **Backend login**: `pulumi login gs://marin-iac-state`.
- **Cluster access** (for the k8s dry-run): the CoreWeave kubeconfig at the path in the
  cluster's `platform.coreweave.kubeconfig_path` (typically `~/.kube/coreweave-iris`).

### Making a change

```bash
cd infra/pulumi
pulumi stack select <cluster>
pulumi preview
```

Read the diff before doing anything else. **No-change / update-in-place is safe. Any `replace`
or `delete` on a NodePool is not** — it deprovisions a reserved bare-metal fleet. Stop and
reconcile the program to match reality; never `pulumi up` through a destructive NodePool diff.
Once the preview is clean, `pulumi up`.

No `KUBECONFIG` export needed anywhere in this flow: `__main__.py` builds the k8s provider with
an explicit `kubeconfig=`/`context=` read from the cluster's own
`platform.coreweave.kubeconfig_path`/`kube_context`, never from the env var or your shell's
current context.

### Adopting a new cluster

A cluster whose RBAC/NodePools/Kueue/Traefik already exist live (the normal case — the CKS
cluster and its kubeconfig were provisioned by hand first) needs one *import* pass so Pulumi
takes ownership of the existing objects instead of planning creates for them. Setting
`marin-iac:import=true` stamps `import_=<live id>` on every resource the program declares:

```bash
pulumi login gs://marin-iac-state
pulumi stack init <cluster> \
  --secrets-provider="gcpkms://projects/hai-gcp-models/locations/us-central1/keyRings/marin-iac-keyring/cryptoKeys/marin-iac-key"
#    (on later runs, just: pulumi stack select <cluster>)

pulumi config set marin-iac:import true
pulumi preview          # gate: every resource `import` + no-op/update; ANY NodePool replace/delete → STOP
pulumi up               # adopts live resources into GCS state; does not recreate them
pulumi config rm marin-iac:import   # import_ is ONE-SHOT: set true → up once → remove
```

Leaving the flag set makes the *next* `up` try to import an already-managed resource and error.

If only some components pre-exist (e.g. RBAC/NodePools/Kueue are live but Traefik was never
installed on this cluster), scope the import pass to just those with `--target`, then run a
normal untargeted `up` afterward to create the rest fresh:

```bash
pulumi config set marin-iac:import true
targets="--target urn:pulumi:<cluster>::marin-iac::marin:coreweave:CoreweaveCluster::cluster \
         --target urn:pulumi:<cluster>::marin-iac::marin:coreweave:IrisRbac::rbac \
         --target urn:pulumi:<cluster>::marin-iac::marin:coreweave:KueueAddon::kueue"
pulumi preview $targets
pulumi up $targets
pulumi config rm marin-iac:import
pulumi up       # normal run, adopt=false now — creates the remaining components fresh
```

### Backend

The shared backend is a GCS bucket + a GCP KMS secrets provider, both in `hai-gcp-models`,
already provisioned:

- **State bucket**: `gs://marin-iac-state` (us-central1, uniform bucket-level access, versioned).
- **Secrets provider**: the KMS key
  `gcpkms://projects/hai-gcp-models/locations/us-central1/keyRings/marin-iac-keyring/cryptoKeys/marin-iac-key`.
  Access is asymmetric, not a shared passphrase: preview-only CI holds
  `roles/cloudkms.cryptoKeyDecrypter`; operators who run `pulumi up` need
  `roles/cloudkms.cryptoKeyEncrypterDecrypter` on the key (see
  [`infra/permissions`](../permissions/README.md)):
  ```bash
  gcloud kms keys add-iam-policy-binding marin-iac-key \
    --keyring=marin-iac-keyring --location=us-central1 --project=hai-gcp-models \
    --member="user:<operator email>" \
    --role="roles/cloudkms.cryptoKeyEncrypterDecrypter"
  ```

## Unsupported

- **Signing keys** (`iris-<cluster>-signing-key`, `finelog-<cluster>-signing-key`) stay manual,
  minted with `iris cluster init-keys` — the key material must never pass through Pulumi state.

## Future work

- **CKS cluster object + VPC**: not yet managed by Pulumi; `CoreweaveCluster` only records the
  resulting cluster identity as config (`CksClusterSpec`). CoreWeave publishes an official
  Terraform provider Pulumi could bridge
  (`pulumi package add terraform-provider coreweave/coreweave`).
- **Object storage** (`s3://marin-<region>` buckets + access keys): no schema or component
  exists yet; buckets are created by hand plus `configure_buckets.py` for lifecycle rules.
  Clusters currently mix per-cluster buckets (`cw-us-west-04a`) and shared cross-region reuse
  (`cw-rno2a`/`cw-us-east-08a` both read/write `marin-us-east-02a`'s bucket) — undecided whether
  Pulumi should provision a bucket per cluster or this reuse is the standing choice.
- **finelog server Deployment**: a planned `FinelogServer` component, not yet built.
- **Federation peers**: `lib/iris/config/marin.yaml`/`marin-dev.yaml`'s `peers:` entries are
  hand-edited per cluster; generate or CI-validate the peer set from the cluster configs so a
  cluster can't be reachable-but-unregistered or registered-but-missing.
- **Folding `helm repo add coreweave ...` into the program itself**, e.g. via `pulumi_command`'s
  `local.run` invoke.
- **`FEDERATION_ALLOW_SOURCES`** is still duplicated between `iac/config.py`'s default and
  `install_cw_network.py`'s own constant; delete the script's copy once nothing calls
  `install_cw_network.py` directly for this.

Everything else in the original design (RBAC, NodePools, Kueue, Traefik/cert-manager, the
federation ingress and DNS, the GCP static IPs, and Artifact Registry mirrors) is landed.
