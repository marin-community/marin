# Infrastructure as Code for Marin clusters

_Why are we doing this? What's the benefit?_

Bringing up and maintaining a cluster today means running a sequence of imperative
Python scripts (`install_kueue.py`, `install_traefik_proxy.py`, `setup_iam.py`,
`iap_gclb.py`, `configure_buckets.py`) in the right order, interleaved with manual
console steps (create the CKS cluster, mint tokens, create NodePools, create object
buckets) — and remembering which ones. As we add clusters this is increasingly
error-prone and undocumented-in-code. We want the *static, low-churn* foundation of
a cluster — the cluster/NodePools, the cluster-wide k8s add-ons, RBAC, and object
storage — described declaratively so a cluster's prerequisites are one reviewable
`preview`/`apply` with real drift detection, instead of tribal script-ordering.

This is deliberately scoped: **IaC owns the static machine and access substrate;
Iris keeps owning everything dynamic** (per-job Pod dispatch, and on GCP the
autoscaler's TPU-slice lifecycle). We migrate **CoreWeave first** because that
surface is largely greenfield (k8s add-ons + NodePools + buckets, no IAP/IAM edge),
then bring GCP onto the same engine.

## Background

Marin has no IaC today. The #6879 research split cluster startup into two classes:
**class 1** (foundational, per-cluster, low-churn — scripts/console) and **class 2**
(per-job slice lifecycle — the in-process autoscaler, a scheduler hot path). Only class 1
is an IaC candidate; class 2 stays imperative in Iris. The approved
[cluster-admin-unification design](../2026-06-23_cluster_admin_unification.md) already adds
a typed top-level **`provisioning:`** config section; the question #6879 raised is *what
execution engine backs it* — this doc decides **Pulumi**, all-or-nothing (both clouds or
neither, no half-Pulumi end state), CoreWeave migrated first as the proof. Full findings,
resource inventory, and ownership boundary are in [research.md](research.md).

## Challenges

The hard part is drawing the **ownership boundary** cleanly. Resources created *today* by
`iris cluster start` (`ensure_rbac()`, `ensure_nodepools()`) must be ceded to IaC — Iris
verifies rather than double-creates. This cede is clean on CoreWeave: the reserved fleets
are pinned-warm NodePools (`minNodes == maxNodes`) whose node counts are driven by
*CoreWeave's* autoscaler inside those bounds, so Iris never managed them — but it does
**not** generalize to GCP TPUs, where Iris's own autoscaler owns slice lifecycle. Second, a
sharp hazard must survive the port: Kueue's admission webhooks must stay **scoped to the
`iris` namespace**; the chart default fail-closed-intercepts CNI/system pods and deadlocks
node delivery cluster-wide.

## Costs / Risks

- New operational surface: a Pulumi CLI, a **GCS state backend**, and a **GCP KMS secrets
  provider** — the state bucket + KMS key a one-time *manual* bootstrap outside IaC
  (chicken-and-egg).
- **Adoption risk on live fleets:** the running CW clusters already hold Iris-created RBAC +
  NodePools; the first apply must `pulumi import` them, not recreate — a botched adoption could
  deprovision a reserved 256-GPU fleet. Gated behind an empty-preview check (Testing).
- **Churn now, payoff deferred:** no user-visible behavior change, and CoreWeave alone doesn't
  justify the tooling — the win lands only when the GCP surface also ports. **Approving this
  is approving the GCP port in principle**, before its signatures exist. It supersedes the
  approved GCP reconciler plan (keeps its `provisioning:` schema, swaps the executor for Pulumi).
- **Bridge + coverage risk:** CKS cluster/VPC come via CoreWeave's official TF provider bridged
  into Pulumi (add-ons + NodePools are k8s manifests either way); the bridge can lag upstream and
  CKS *cluster-create* maturity is unproven by us (Open Questions).
- **IaC can't cover everything:** R2/CoreWeave bucket *lifecycle* has no first-class provider, so
  `configure_buckets.py` keeps it; IAP OAuth brand/clients stay external inputs
  (`read_oauth_client`) — no regression only if that boundary holds.

## Design

**Tool: Pulumi (Python).** We weighed it against Terraform/OpenTofu on the axes that map to
*our* stack, not in the abstract (full table + sources in [research.md](research.md)). Four
favor Pulumi concretely: **(1) Python reuse** — import `rigging.filesystem` and read the
cluster's `provisioning:` section directly, no config re-expression; **(2) CRD-then-CR in
one apply** — our stack is CRD-heavy (cert-manager,
Kueue, CoreWeave NodePool), and Terraform's `kubernetes_manifest` validates CRD schemas at
plan time so it can't install a CRD and its CR in one run (our scripts already loop on
`wait_for_crd_established`); **(3) encrypted secrets in state** — we're secret-heavy (SA
keys, R2/CW/IAP secrets, finelog delegation key), Terraform stores them plaintext; **(4) the
Automation API**, which lets `marin-cluster admin` drive `up`/`preview`/`destroy` in-process
— the concrete mechanism for "IaC is the engine behind `provisioning:`". Pulumi **bridges the
official CoreWeave TF provider** (`coreweave_cks_cluster` + VPC), reused from Python. The
honest counterweights — Terraform's larger ecosystem and native (un-bridged) CoreWeave/
Cloudflare providers — are real but general, and didn't outweigh the stack-specific wins.

**Layout: a top-level `infra/pulumi/` directory in the monorepo**, next to `config/` and
the CI that runs it. One Pulumi **project**, one **stack per cluster** (`cw-us-east-02a`,
`cw-rno2a`, later `marin` for GCP). Component resources model the surface:

- `CoreweaveCluster` — CKS cluster + reserved NodePools (via the CW TF provider).
- `KueueAddon` — `cks-kueue` chart, Topology CRs, `cw-ib` ResourceFlavor, `iris-cq`
  ClusterQueue, with **namespace-scoped webhooks** (the hazard above, asserted in tests).
- `TraefikAddon` — Traefik + cert-manager + Let's Encrypt HTTP-01 ClusterIssuers.
- `IrisRbac` — the `iris` Namespace, `iris-controller` ServiceAccount, ClusterRole,
  ClusterRoleBinding (ceded from `ensure_rbac()`).
- `ObjectStorage` — `s3://marin-*` buckets + access keys.

**Config stays single-source.** The NodePool envelope (instance types, node counts) is
*derived* from the Iris config's `scale_groups`, the namespace from
`kubernetes_provider.namespace`, and the Kueue queue name from
`kubernetes_provider.kueue.cluster_queue` — never re-declared. The new `provisioning:`
section (in the per-cluster `lib/iris/config/<cluster>.yaml`) carries only the residual
cluster facts not already in the Iris config: CKS cluster name, ResourceFlavor/topologies,
ACME issuers, buckets. `Pulumi.<stack>.yaml` is just a cluster-name pointer; the program
holds no Marin-specific constants. Deriving namespace from one source also makes the
webhook-scope invariant (Kueue `pod_namespace` == RBAC namespace) structural, not just
validated.

**Iris cedes the migrated resources.** `start_controller()` drops `ensure_rbac()` and
`ensure_nodepools()` and instead **verifies** they exist, failing with a clear "run
`infra/pulumi` apply first" message. Everything else it creates is unchanged (controller
Deployment + Service, ConfigMap, `iris-task-env` Secret, namespaced LocalQueue, per-job
Pods). The CoreWeave boundary: **IaC owns the node-capacity envelope + access substrate;
Iris owns the Pods inside it.** This boundary is *provider-shaped*, not universal — on GCP
there are no reserved per-scale-group NodePools (Iris's autoscaler owns TPU slices), so
`verify_prerequisites` is provider-scoped and checks a different GCP-arm set. NodePools are
declared with `ignore_changes` on their runtime node count, so CoreWeave's own autoscaler
moves capacity within the IaC-declared `[min, max]` without fighting Pulumi.

GCP is **committed but sequenced after CoreWeave**: the layout reserves a `marin`/GCP
stack, and this pass leaves the GCP `iris cluster start` path untouched. Once CoreWeave
proves the pattern, `setup_iam.py`/`iap_gclb.py`/`configure_gcp_registry.py` port onto
Pulumi as `GcpIam` / `GcpGclbIap` / `GcpRegistry` components reusing the same
`provisioning:` contract — that port is part of the commitment, not an optional extra.

## Deploying on your own infrastructure

The IaC lives in the monorepo, but the real portability contract is **not** the Pulumi
program — it's the prerequisite set `iris cluster start` now *verifies* (a usable
namespace + RBAC, NodePools matching the config's scale groups, Kueue with
namespace-scoped webhooks, an ingress for `/proxy`, an object bucket + creds). Because
Iris verifies rather than creates these, the contract is explicit and a deployer can
satisfy it however they like; our Pulumi program is a *reference implementation* of it, not
a requirement. Two paths: if your topology matches ours (CKS, or GCP+IAP), **fork
`infra/pulumi/` and bring your own `provisioning:` config + state/KMS backend** — the program
holds no Marin-specific constants, so only config differs. If your topology differs (EKS,
on-prem, a different object store), **satisfy the prerequisite contract with your own IaC**
and run `iris cluster start` against it. Detail and the tradeoff analysis are in
[research.md](research.md).

## Testing

Three levels, no paid CoreWeave in the default path:

- **Unit / plan**: `pulumi preview` on the diff in CI (plan is the reviewable artifact).
  Assert the rendered Kueue webhook `namespaceSelector` is scoped to `iris` — the single
  most important regression, since an unscoped webhook deadlocks node delivery cluster-wide.
- **kind**: the k8s-portable components (`KueueAddon` upstream variant, `IrisRbac`) apply
  against a kind cluster, reusing the existing `tests/e2e/gpu_gang_smoke.py` harness path.
- **Greenfield rollout check** (the acceptance test): on `cw-us-west-04a`, `infra/pulumi` apply
  provisions RBAC + Kueue + Traefik + NodePools, then `iris cluster start` succeeds against
  those prerequisites **without** its own `ensure_rbac()`/`ensure_nodepools()`, and a CPU
  hello-world + 8-GPU `jax.devices()` job run. Proves the cede boundary end-to-end.
- **Adoption check** (must run before touching the live fleets): `pulumi import` the existing
  RBAC + NodePools on a live cluster, then `pulumi preview` must show an **empty/additive-only**
  plan — a destructive NodePool diff is a stop-the-line bug (it would deprovision a reserved
  256-GPU fleet). The migration procedure is pinned in `spec.md §4`.

## Workflow

IaC changes ship like code, with a human in the loop for the apply:

1. **CI preview (every PR).** A GitHub Actions job authenticates to GCP via Workload Identity
   Federation (keyless — no static SA keys) and runs `pulumi preview` for each affected stack,
   posting the plan as a PR comment. The plan is the reviewable artifact; a destructive NodePool
   diff blocks the PR (the same gate as the adoption check).
2. **Manual apply.** After review/merge, an **operator** runs `pulumi up` from a trusted
   environment, re-reading the preview. Apply stays human-driven because these stacks touch
   reserved GPU fleets — CI holds *decrypt-only* KMS access (preview), operators hold *encrypt*
   (apply).
3. **Later: full CD.** Once trusted, promote to apply-on-merge behind a GitHub Environment with
   required approval — same WIF identity, preview still the gate, the approval the checkpoint.

State lives in `gs://marin-iac-state`, encrypted with a GCP KMS secrets provider; the CI job,
IAM split, secrets model, and CD gate are pinned in `spec.md §9`.

## Open Questions

- **CoreWeave declarative coverage — resolved.** IaC starts at the NodePool/add-on layer;
  `coreweave_cks_cluster` create/adopt is not built. Bridging the CoreWeave Terraform provider
  into Pulumi needs CoreWeave API credentials, which we do not have; without the bridge Pulumi
  has no resource type for the CKS cluster object, so it cannot even be *adopted* into state
  (import needs a provider, same as create). `CoreweaveCluster` records the cluster as external
  config (`CksClusterSpec`, exported as plain outputs); cluster-create stays a console step.
  Revisit if CoreWeave API credentials get provisioned and the bridge is worth its maintenance
  cost — we manage only the NodePools, so provider coverage of cluster-create is the open unknown.
- **Verification strictness.** How strict should `iris cluster start`'s new prerequisite
  check be — exact-match on NodePool specs/RBAC verbs (catches drift, brittle across Iris
  versions) or presence-only (tolerant, misses drift)? This defines the portability contract's
  edges.
