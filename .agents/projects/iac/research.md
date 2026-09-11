# IaC for Marin — Research

Prior work / seed material:
- [Issue #6879 — investigate pulumi for organizing resource startup](https://github.com/marin-community/marin/issues/6879) (with a detailed bot research comment)
- [`2026-06-23_cluster_admin_unification.md`](../2026-06-23_cluster_admin_unification.md) — approved A+B; adds a typed `provisioning:` section + `marin-cluster admin` config-driven reconcilers
- [Issue #6961 — top-level marin admin scripts for cross-service key provisioning](https://github.com/marin-community/marin/issues/6961)

## The two classes of "resource startup" (from #6879)

Only **class 1** is a Pulumi candidate:

1. **Foundational, per-cluster, low-churn** — created once, changed rarely; today done by hand or one-off Python scripts.
2. **Runtime, per-job, high-churn** — TPU/GPU slices created/destroyed continuously by the in-process autoscaler; a scheduler hot path reconciled every few seconds. Declarative plan/apply is the wrong tool here — **stays imperative in Iris**. This matches the user's constraint: "Iris will still manage the dynamic configuration of workers and TPUs."

## Marin has no IaC today

No Terraform/Pulumi/Ansible/Helm-of-record. Foundational setup is a set of imperative Python scripts + manual console steps:

| Concern | Today | Cloud |
|---|---|---|
| GCS/R2/CW bucket lifecycle | `infra/configure_buckets.py` | all |
| Artifact Registry cleanup | `infra/configure_gcp_registry.py` | GCP |
| Service accounts + IAM | `lib/iris/scripts/setup_iam.py` | GCP |
| GCLB + IAP proxy | `lib/iris/scripts/iap_gclb.py` | GCP |
| Kueue install (Helm) | `lib/iris/scripts/install_kueue.py` | CW + kind |
| Traefik + cert-manager + ACME issuers | `lib/iris/scripts/install_traefik_proxy.py` | CW |
| k8s Namespace + RBAC | `iris cluster start` → `CoreweavePlatform.ensure_rbac()` | CW |
| CKS cluster + reserved NodePools | **manual** (Console, or CW Terraform provider) | CW |
| CW object-storage buckets + access keys | **manual** (Console) | CW |

## The CoreWeave surface (first migration target)

Source of truth for the runbook: [`lib/iris/docs/coreweave.md`](../../../lib/iris/docs/coreweave.md). Active clusters `cw-us-east-02a`, `cw-rno2a` (reserved, pinned-warm H100 fleets).

What is **static / install-once / manual** today and thus an IaC candidate:
- **CKS cluster** itself + the **reserved NodePools** (pinned warm: `buffer_slices == max_slices` → `minNodes == maxNodes`, never scale to zero — effectively static machines). Today `ensure_nodepools()` in `iris cluster start` creates them, but for a reserved fleet this is really static provisioning.
- **Kueue** — `install_kueue.py --variant coreweave --with-queues`: the `cks-kueue` chart, Topology CRs, cluster-scoped `cw-ib` ResourceFlavor + `iris-cq` ClusterQueue. Namespace-scoped webhooks (unscoped webhooks deadlock node delivery — a sharp operational hazard).
- **Traefik + cert-manager + Let's Encrypt HTTP-01 issuers** — `install_traefik_proxy.py`. CKS ships no ingress controller / TLS issuer.
- **Namespace + RBAC** (`iris` ns, `iris-controller` SA, ClusterRole/Binding) — created by `ensure_rbac()` on every `cluster start`.
- **Object storage** buckets + access keys (`s3://marin-us-east-02a`, R2 endpoints).

What **stays in Iris** (dynamic, per the user's constraint): controller Deployment + Service, ConfigMap, `iris-task-env` Secret, worker/task Pods, LocalQueue reconcile, per-job Pod dispatch (Pod-per-task), NHC preemption.

Note: on CoreWeave there is **no IAP/JWT edge** — auth is by network location (`auth.trusted_cidrs`) + per-endpoint bearer for `/proxy`. So the GCP-specific `setup_iam.py`/`iap_gclb.py` provisioning does **not** apply to CoreWeave; the CoreWeave IaC surface is largely greenfield (k8s add-ons + cluster/nodepools + buckets), not a rewrite of the GCP admin scripts.

## GCP surface (must keep working; not migrated first)

GCP is the more complex arm: project/SAs/IAM (`setup_iam.py`), GCLB + IAP (`iap_gclb.py`), Artifact Registry (`configure_gcp_registry.py`), buckets (`configure_buckets.py`), controller VM. TPU slices are class-2 (autoscaler, stays imperative). GCP is **in scope** for IaC — the maintainer's framing is all-or-nothing (Pulumi for both clouds, or neither) — but CoreWeave is migrated first as the low-risk proof. The tool + repo layout must therefore leave a clean seat for the GCP stack from day one (see the reserved GCP stack in the layout), even though GCP resources are ported after CoreWeave lands.

## Relationship to the approved cluster-admin-unification work

`2026-06-23_cluster_admin_unification.md` (approved A+B) adds a typed top-level **`provisioning:`** section to `config/<cluster>.yaml` and refactors `setup_iam.py`/`iap_gclb.py` into config-driven idempotent Python reconcilers under `marin-cluster admin`. The #6879 bot comment's key point: the real question is not "Pulumi yes/no" but **what execution engine backs that `provisioning:` layer**. This design's position is decided: **Pulumi is that engine.** The typed `provisioning:` schema stays the config contract; Pulumi replaces the hand-rolled reconcilers as the executor, and `marin-cluster admin` drives it (via the Automation API) rather than shelling to `gcloud`. We keep the `provisioning:` schema work from that doc and fold its rollouts into Pulumi — a parallel Python-reconciler track is explicitly rejected (the bot warned against a competing design). This is all-or-nothing per the maintainer: either Pulumi is the IaC solution for **both** GCP and CoreWeave, or we keep the existing scripts unchanged — there is no half-Pulumi end state. CoreWeave (greenfield k8s add-ons, no IAP/IAM edge) is where we prove the pattern before porting the GCP `provisioning:` rollouts onto the same engine.

## Tool choice: Pulumi vs Terraform (evidence)

### What we provision today (concrete resource inventory)

**GCP** (all via `gcloud`, imperative reconcilers):
- `setup_iam.py` — `google_service_account` (controller/worker SAs), project IAM bindings (`roles/tpu.admin`, `storage.objectAdmin`, `artifactregistry.reader`, …), SA-on-SA bindings (`serviceAccountTokenCreator`/`User`). **Well-covered by both tools.**
- `iap_gclb.py` (1387 lines) — the GCLB+IAP edge: zonal **NEG** → health check → global **backend-service** → **url-map** (with `path-matcher` surgery via `export`/`import` JSON edit) → target proxy → global **forwarding-rule**, managed **ssl-certificate**, **IAP enable** + `oauth_settings` (`clientId` + `programmaticClients`), and `iap.httpsResourceAccessor` IAM bindings. **Well-covered declaratively** — and the ugly `url-map export/import` surgery becomes a plain declarative `url_map` with `path_matcher` blocks (an IaC *win*). One caveat: **`google_iap_brand`/`google_iap_client` are create-only** in the TF provider (brand can't be updated/deleted). We already sidestep this — `read_oauth_client()` treats OAuth clients as **pre-created inputs read from secret files**; we never provision the brand. Keep that boundary.
- `configure_gcp_registry.py` — Artifact Registry `set-cleanup-policies` → `google_artifact_registry_repository.cleanup_policies`. **Covered.**

**Buckets** (`configure_buckets.py`): GCS lifecycle via `gcloud storage` (→ `google_storage_bucket.lifecycle_rule`, covered), but **R2 and CoreWeave object-storage lifecycle via the raw botocore S3 API**. Neither Pulumi nor Terraform has a first-class CoreWeave-object-storage provider, and Cloudflare's R2 lifecycle support is thin. This is a **weak spot for IaC regardless of tool** (see "What's harder" below).

**CoreWeave** (`install_kueue.py`, `install_traefik_proxy.py`, `ensure_rbac`, `ensure_nodepools`): Helm releases (`cks-kueue`, `traefik`, `cert-manager`) + raw kubectl manifests (Topology CRs, `cw-ib` ResourceFlavor, `iris-cq` ClusterQueue, ClusterIssuers) + CRD-establishment waits + namespace-scoped webhook config. NodePools and RBAC are k8s objects.

### CoreWeave provider reality

CoreWeave ships an official **Terraform provider** (`coreweave/coreweave`, GA Feb 2025) with a first-class `coreweave_cks_cluster` resource (+ VPC). **NodePools are managed as Kubernetes manifests** through the cluster kubeconfig, *not* as native provider resources ([CKS Terraform docs](https://docs.coreweave.com/docs/products/cks/terraform/about), [`coreweave_cks_cluster`](https://registry.terraform.io/providers/coreweave/coreweave/latest/docs/resources/cks_cluster)). So on either tool: cluster+VPC = provider resource; NodePools + add-ons = k8s API. **Pulumi bridges any TF provider** (`pulumi package add terraform-provider coreweave/coreweave`), so a Pulumi program reuses the CoreWeave (and Cloudflare) TF providers from Python — Pulumi rides on the TF ecosystem rather than replacing it.

### The comparison, weighted for *our* situation

| Axis | Pulumi | Terraform / OpenTofu | Verdict for us |
|---|---|---|---|
| Language | Python (real) | HCL | **Pulumi** — reuse `rigging.filesystem.s3_data_buckets`, read `config/<cluster>.yaml` `provisioning:` directly, compute the served CRD apiVersion in-code (exactly what `install_kueue.py` does today). No re-expression of config. |
| CRD-then-CR in one apply | Yes | **No** — `kubernetes_manifest` validates schema at plan time (two-apply problem); needs `kubectl` provider or split applies ([TF k8s CRD FAQ](https://developer.hashicorp.com/terraform/tutorials/kubernetes/kubernetes-crd-faas), [avoid kubernetes_manifest](https://medium.com/@danieljimgarcia/dont-use-the-terraform-kubernetes-manifest-resource-6c7ff4fe629a)) | **Pulumi** — our stack is CRD-heavy (cert-manager, Kueue, CoreWeave NodePool); scripts already loop on `wait_for_crd_established`. |
| Secrets | First-class, encrypted in state by default | **Plaintext in state**; mitigate with Vault/backend | **Pulumi** — we're secret-heavy (SA keys, R2/CW access keys, IAP secrets, finelog delegation secret) and the admin design uses `gcp-secret://` refs. |
| Embed in a CLI | **Automation API** (drive up/preview/destroy from Python) | None equivalent | **Pulumi** — lets `marin-cluster admin` call the engine in-process, matching "IaC is the engine behind `provisioning:`". |
| Ecosystem / battle-testing | Smaller; ~45%/yr growth | ~76% market share; huge corpus; leaner at scale | **Terraform** — more GCLB/IAP examples, more contributors know HCL. |
| Provider source | Bridges TF providers (CoreWeave/Cloudflare) — extra layer | Native | **Terraform** — CoreWeave/Cloudflare providers are TF-first; Pulumi adds a bridge that can lag. |
| Licensing / backend | Open-core SDK; default backend is Pulumi Cloud (self-host `s3://` supported) | BSL since 2023; **OpenTofu** is the truly-OSS fork | Wash — both self-hostable; OpenTofu removes the BSL concern for TF. |

Sources: [Pulumi vs Terraform (Pulumi)](https://www.pulumi.com/docs/iac/comparisons/terraform/), [spacelift](https://spacelift.io/blog/pulumi-vs-terraform), [env0](https://www.env0.com/blog/pulumi-vs-terraform-an-in-depth-comparison).

**Net:** the axes that favor Pulumi (Python reuse of our config/rigging, CRD ergonomics our code already fights, first-class secrets, Automation API to back the umbrella) are the ones that map to *our specific stack*; the axes that favor Terraform (bigger ecosystem, native CoreWeave provider) are real but general. Recommend **Pulumi**, eyes open to the bridge-layer dependency on TF providers.

## What's harder under IaC (either tool) — honest list

- **R2 / CoreWeave object-storage bucket lifecycle.** No first-class provider; today it's a botocore S3 reconciler. Options: keep the botocore reconciler out of IaC (pragmatic), or drive it through the AWS S3 provider pointed at the CW/R2 endpoint (works for generic S3, non-AWS quirks). Lean: **leave bucket lifecycle as-is initially**; don't force it into the first cut.
- **IAP OAuth brand/clients** are create-only declaratively — but we already treat them as external inputs, so no regression *if we hold that boundary*.
- **Bridged-provider drift.** CoreWeave/Cloudflare are TF-native; a Pulumi bridge can lag upstream provider releases.
- **CKS cluster-create maturity.** The provider is new (2025); how robust `coreweave_cks_cluster` create/update is at our fleet sizes is unproven by us — may start IaC at the NodePool/add-on layer and leave cluster-create a console step initially.

## State & secrets backend

Pulumi supports self-managed backends, so we avoid a Pulumi-Cloud dependency: **state in a GCS bucket** (`pulumi login gs://marin-iac-state`) and the **secrets provider a GCP KMS key** (`pulumi stack init --secrets-provider gcpkms://…`). GCS is the logical choice — we already run on GCP, it's the same trust domain as our other admin tooling, and it works uniformly for both the CoreWeave and GCP stacks (the state bucket need not live in the cloud a stack provisions; a CoreWeave stack keeping its state in GCS is fine and keeps one backend for everything). The backend + secrets provider are **stack configuration**, not baked into the program — a third party points their stacks at their own bucket/KMS key.

## Deployability by others (monorepo; detailed)

Decision: IaC lives in the monorepo (`infra/pulumi/`). The tradeoff, made concrete:

- **For the monorepo:** the Pulumi program sits next to the `config/<cluster>.yaml` it reads and the CI that runs it; the `provisioning:` schema is versioned together with the Iris code that consumes it (no cross-repo version skew); a third party gets a working, runnable reference implementation by forking one tree.
- **Against (and the mitigation):** our concrete values — project ids, `oa.dev` DNS, reserved fleet sizes, SA emails, bucket names — are checked into `config/*.yaml` and the per-stack `Pulumi.<stack>.yaml`. The risk is the program accidentally hard-coding *our* specifics. **Mitigation, load-bearing:** every concrete value lives in the `provisioning:` config or stack config; the Pulumi program is generic over the schema and contains no Marin-specific constants. This is the same discipline the admin-unification doc already imposes on `setup_iam.py`/`iap_gclb.py` (kill the `DEFAULT_PROJECT`/`DEFAULT_ZONE` constants).

**The real portability contract is not the IaC — it's the set of prerequisites `iris cluster start` now *verifies*** (namespace + RBAC it can use, NodePools matching the config's scale groups, Kueue with namespace-scoped webhooks + the bound ClusterQueue, an ingress publishing `/proxy`, an object-storage bucket + creds). Because Iris verifies rather than creates these, the contract is explicit and testable, and a deployer can satisfy it *however they like*. Our Pulumi program is a reference implementation of that contract, not a requirement.

Two personas:

1. **"Run Iris as-is on my own CoreWeave/GCP account"** (topology matches ours). Workflow: copy a `config/cw-*.yaml` (or the GCP `marin.yaml`), edit its `identity:` + `provisioning:` to their project/region/fleet/DNS, `pulumi stack init <their-cluster>` pointing at *their* state bucket + KMS key, `pulumi up`, then `iris --cluster=<their-cluster> cluster start`. They reuse our program unchanged; only config differs. This is the happy path and the reason monorepo + schema-generic program matters.
2. **"Run Iris on infra we don't support"** (AWS EKS, on-prem k8s, a different ingress/object store). Our Pulumi program doesn't fit their topology. Workflow: they provision the documented prerequisite contract with their own IaC (their own Pulumi program, Terraform, Helm, or by hand), then `iris cluster start` verifies the invariants and runs. They write their own IaC; the recommendation is *not* to bend ours to every backend. The Iris cluster config schema is what they target.

So the recommendation to a would-be deployer is a decision tree: *topology matches → fork `infra/pulumi/`, bring your own `provisioning:` config + backend; topology differs → satisfy the prerequisite contract with your own IaC.* Either way the Iris cluster config is the stable boundary, and the monorepo location makes the reference implementation and the contract it satisfies discoverable in one place.
