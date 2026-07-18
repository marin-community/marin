# IaC coverage gaps — what a live Iris cluster needs beyond pt1

Pt1 ([the CoreWeave RBAC + NodePool cut](spec.md)) provisions two of the prerequisites an
Iris CoreWeave cluster actually needs. This doc inventories the rest: every static
prerequisite of a live deployment, who owns it today, whether IaC covers it, and — for the
gaps — exactly where it lands. It answers the "analyze the remaining gaps … egress IPs,
finelog auth secrets, etc." ask on PR #7153, and it is the running checklist for the
follow-up slices named in the spec.

Scope note: this is the **static / install-once substrate** only. Everything dynamic —
per-job Pod dispatch, the GCP TPU-slice autoscaler, NHC preemption — stays in Iris by design
(research.md, class 2) and is not a gap.

## Inventory

Owner legend: **IaC-pt1** = provisioned by this PR · **IaC-next** = a deferred component
named in [spec.md §3](spec.md) whose config schema already exists in
[`config.py`](../../../infra/iac/src/iac/config.py) · **IaC-gcp** = belongs to the deferred
GCP arm · **Iris** = stays in `start_controller()` by design · **manual** = console / script,
not yet modeled anywhere.

| # | Prerequisite | Today | IaC status | Lands in |
|---|---|---|---|---|
| 1 | Reserved NodePools (per scale group) | `ensure_nodepools()` | **IaC-pt1** | `CoreweaveCluster` ✅ |
| 2 | Namespace + controller RBAC (SA, ClusterRole, Binding) | `ensure_rbac()` | **IaC-pt1** | `IrisRbac` ✅ |
| 3 | CKS cluster object + VPC + kubeconfig | manual (console / CW TF provider) | **decided: permanently manual** | `CoreweaveCluster` documents it (`CksClusterSpec`, exported not managed — [cluster.py](../../../infra/iac/src/iac/coreweave/cluster.py)); no CoreWeave TF provider bridged, no CoreWeave API credentials — see below |
| 4 | Kueue: `cks-kueue` chart, Topology CRs, `cw-ib` ResourceFlavor, `iris-cq` ClusterQueue, **namespace-scoped webhooks** | `install_kueue.py --with-queues` | **IaC-landed** | `KueueAddon` ✅ |
| 5 | Traefik + cert-manager + HTTP-01 ClusterIssuers | `install_cw_network.py` | **IaC-landed** | `TraefikAddon` ✅ |
| 6 | **Federation ingress**: IP-locked `Ingress` + `ipAllowList` Middleware over the whole controller host | `install_cw_network.py` | **IaC-landed** | `TraefikAddon` ✅ — reads `IngressSpec.federation_allow_sources` (see §Egress IPs) |
| 7 | Object-storage buckets + access keys (`s3://marin-<region>`) | manual console + `configure_buckets.py` (lifecycle) | IaC-next | `ObjectStorage` (`ObjectStorageSpec` exists); bucket *lifecycle* stays out (spec §7) |
| 8 | **iris controller signing key** (`iris-<cluster>-signing-key`) | `iris cluster init-keys` → GCP Secret Manager | **decided: stays manual** | `init-keys` does the whole thing (create-if-absent + write + IAM); Pulumi only warns (non-fatal, `pulumi.log.warn` in `__main__.py`) when `auth.signing_key` is unset, with the exact command to run (see §Signing secrets) |
| 9 | **finelog forwarding signing key** (`finelog-<cluster>-signing-key`) | minted by hand → GCP Secret Manager | **gap** | same posture as row 8 once a cluster needs finelog forwarding (see §Signing secrets) |
| 10 | Federation **egress** IP reservations (`34.27.183.11`, `35.254.13.19` = `iris-marin-fed-egress` / `iris-marin-dev-fed-egress`) | reserved by hand in `hai-gcp-models` | **IaC-landed** (GCP arm) | `GcpStaticAddresses` ✅ ([gcp/addresses.py](../../../infra/iac/src/iac/gcp/addresses.py)); the CoreWeave-side allowlist is `IngressSpec.federation_allow_sources` (see §Egress IPs) |
| 11 | DNS: `iris-cw-<cluster>.oa.dev` CNAME → Traefik LB FQDN | manual (Cloudflare) | **decided: stays manual for now** | considered and deferred — see §DNS CNAME below |
| 12 | finelog server Deployment (in-cluster) | `finelog deploy up <cluster>` | IaC-next (planned) | `FinelogServer` component (a later CoreWeave slice; needs the finelog signing key) |
| 13 | Iris runtime objects: ConfigMap, `iris-task-env` Secret, LocalQueue, PriorityClasses, controller Deployment + Service, state PVC | `start_controller()` | **Iris (by design)** | stays in Iris (spec §4) |

Rows 1–2, 4–6, 8, 10 are done. Rows 3 and 11 are resolved decisions (deferred, not undecided —
see below). Row 7 is the remaining sequenced CoreWeave follow-up already in the design; row 12
is a planned CoreWeave slice. Row 9 is still open. Row 13 is deliberately *not* IaC.

## The two gaps rjpower named

### Egress IPs (row 6 + row 10)

"Egress IPs for the iris controller" is really three coupled resources, split across clouds —
two now modeled, one deferred:

- **The GCP reservations (row 10) — landed.** The marin / marin-dev controllers federate
  *into* each CoreWeave controller, and their egress IPs (`34.27.183.11`, `35.254.13.19`) are
  reserved as `iris-marin-fed-egress` / `iris-marin-dev-fed-egress` in project
  `hai-gcp-models`. These are now `google_compute_address` resources in the GCP arm —
  `GcpStaticAddresses` ([gcp/addresses.py](../../../infra/iac/src/iac/gcp/addresses.py)), the
  GCP arm's first slice, on the `marin` stack. Each pins its IP so adoption imports the live
  reservation without ever reassigning an IP baked into a CoreWeave allowlist. (Confirmed
  against the live reservations: both are EXTERNAL, `us-central1`, in use.)

- **The CoreWeave-side allowlist (part of row 6) — landed as a config input.** Which sources
  the CoreWeave federation route admits is `IngressSpec.federation_allow_sources`, defaulting
  to the `MARIN_FEDERATION_EGRESS_SOURCES` constant in
  [`config.py`](../../../infra/iac/src/iac/config.py) (the same values as the
  `FEDERATION_ALLOW_SOURCES` constant in
  [`install_cw_network.py`](../../../lib/iris/scripts/install_cw_network.py)).

- **The enforcement (CoreWeave side, row 6) — deferred.** The `ipAllowList` Middleware +
  IP-locked Ingress that admits only those sources over the whole controller host. This is a
  k8s object currently applied by `install_cw_network.py`, folded into `TraefikAddon` when
  that lands — it reads `federation_allow_sources`.

**One home, eventually.** Until the federation-ingress component consumes the field, the
allowlist has two copies (the config constant and the script constant); the config docstring
flags the sync obligation, and the follow-up deletes the script constant so the input is the
sole source.

### finelog auth secrets (rows 8–9)

Two Ed25519 signing keys per cluster, both GCP Secret Manager secrets referenced by
`gcp-secret://projects/748532799086/secrets/…`:

- `iris-<cluster>-signing-key` — the controller signs its worker / `/proxy` / federation
  tokens with it (`auth.signing_key` in the cluster config). Minted by
  `iris cluster init-keys`.
- `finelog-<cluster>-signing-key` — the finelog server signs the rows it forwards to the
  `marin` hub (`forwarding.signing_key` in the finelog config).

The **private** halves are the Secret Manager secrets, minted entirely by `iris cluster
init-keys` (create-if-absent, write the version, optionally grant `--accessor`). Considered and
rejected: having Pulumi provision the empty secret first. The only real benefit — an operator
running `init-keys` needing just `secretAccessor`/`versions.add` on one secret instead of
project-wide `secretmanager.secrets.create` — is entirely latent today (every operator who runs
it is a project Owner already), so it wasn't worth a new component. Pulumi's only involvement is
a non-fatal `pulumi.log.warn` in `__main__.py` when a cluster's `auth.signing_key` is unset,
naming the exact `init-keys` command to run. The **public** halves are config, hand-registered
in two places: the cluster's `auth.federation_peers` (peer controllers' keys) and the hub's
[`marin.yaml`](../../../lib/finelog/config/marin.yaml) finelog `auth` (accepted forwarder keys).

Row 9 (finelog) is still a **hard gate**, not just a to-do: `test_every_bundled_sender_names_a_cluster_some_bundled_hub_trusts`
in [`lib/finelog/tests/test_config.py`](../../../lib/finelog/tests/test_config.py) fails a
bundled finelog config that forwards as a cluster no hub's jwt layer trusts. So a forwarding
finelog server for a cluster cannot even be committed until its key is minted and registered.

**Remaining work:** once a cluster needs finelog forwarding, run `init-keys --gcp-secret
projects/hai-gcp-models/secrets/finelog-<cluster>-signing-key`, register the public halves, and
add the finelog deploy config + `finelog:` block to the Iris config.

### DNS CNAME (row 11)

`iris-cw-<cluster>.oa.dev` must be CNAME'd to the Traefik LoadBalancer's `*.coreweave.app`
FQDN (`install_cw_network.py`'s `read_traefik_fqdn`) so the federation ingress (`TraefikAddon`,
row 6) is actually reachable. Considered automating this with a bridged Cloudflare Terraform
provider (`pulumi package add terraform-provider cloudflare/cloudflare`) and rejected for now:

- **No Cloudflare API token exists anywhere in this codebase** (checked; same gap as the
  CoreWeave provider in row 3) — a new credential would need to be provisioned first.
- **The CNAME target isn't a static, declarable value.** CoreWeave's External Hostname
  Controller allocates the LoadBalancer's `*.coreweave.app` FQDN *asynchronously* after Traefik
  comes up — `read_traefik_fqdn` polls the Service's `status.conditions[ExternalRecords]` field
  for up to 90s, then rewrites the wildcard label to the real hostname (confirmed live: applying
  `TraefikAddon` for `cw-us-west-04a` took ~20s before
  `iris-cw-us-west-04a.208261-marin.coreweave.app` was allocated and readable). Pulumi's
  declarative model has no clean way to express "create this DNS record once another resource's
  async status field appears" — it would need a custom Dynamic Provider replicating that same
  poll-and-rewrite logic, or an `Output.apply()` reading the Traefik Service's live status
  (which `TraefikAddon`'s `helm.v3.Release` doesn't expose directly; a separate `Service.get()`
  read would be needed on top).

Comparable in size to `TraefikAddon` itself once a Cloudflare token exists — not a quick
bolt-on, but bounded. The DNS CNAME stays a manual, printed-instruction step
(`install_cw_network.py`'s own "Done. To finish wiring..." output) until then.

### Pulumi Helm chart resolution — resolved (2026-07-17), workaround adopted

`TraefikAddon`'s two Helm `Release`s (`traefik`, `cert-manager`) intermittently failed
`pulumi preview`/`up` with `chart "coreweave/<x>" version "<y>" not found in
https://charts.core-services.ingress.coreweave.com repository`, even though the chart and
version genuinely exist and resolve fine via the real `helm` CLI. `KueueAddon`'s `cks-kueue`
Release (same repo, same `repository_opts` pattern) never once failed across the whole
investigation below.

**Decision: drop `repository_opts` on `traefik`/`cert-manager` only** (`src/iac/coreweave/traefik.py`).
Verified clean across 8 consecutive `pulumi preview` runs after the change (0 failures), versus
5/5 failures the same session with `repository_opts` present. Requires `helm repo add coreweave
<url>` registered locally — now documented as a hard prerequisite in `infra/iac/README.md`, not
optional. `KueueAddon`'s `cks-kueue` Release is untouched (still uses `repository_opts`) since it
was never observed to fail.

**Confirmed, via direct testing (not documentation, not speculation):**
- Not caching. Pointing `HELM_CACHE_HOME` at an empty directory and re-running: the directory
  was never touched. Deleting and regenerating the real `helm` CLI's own cache
  (`~/Library/Caches/helm`) made no difference either. There is no local cache being read for
  this code path at all.
- Not chart-specific. Forcing `traefik` to resolve before `cert-manager` (via `depends_on`):
  `traefik` failed 5/5 runs. Flipping the order: `cert-manager` failed 5/5. Whichever one goes
  first in a forced pair fails, every time — not a property of one chart.
- Not a simple ordering/position rule either. Making both Releases depend on `KueueAddon`'s
  (always-succeeding) Release, so they'd start after it but race each other: exactly one of the
  pair failed each run, but *which* one varied — not deterministic. Chaining them into a fully
  serial `kueue -> cert-manager -> traefik` sequence (zero concurrency, one resolution in
  flight at a time) made `cert-manager` fail deterministically, 8/8 runs — worse than the
  original intermittent behavior, not better. No dependency-graph shape tried produced a
  reliably clean run.

**Two confirmed-working alternatives; here's why one was picked:**
- **Adopted: drop `repository_opts`.** Pulumi falls back to the local `helm` CLI's own repo
  config (`~/Library/Preferences/helm/repositories.yaml`, populated by `helm repo add coreweave
  <url>`). Reliable in every test. Real, accepted cost: `pulumi preview`/`up` now silently
  depends on that local, undeclared-in-code prerequisite — breaks on any fresh checkout or CI
  runner that hasn't run `helm repo add`. Mitigated, not eliminated, by making it an explicit,
  documented step in `infra/iac/README.md`'s Prerequisites; still needs to be added to any CI
  workflow that runs `pulumi preview`/`up` against a CoreWeave stack (none exists yet — see §9
  Phase 1 in `spec.md`, not yet built). Also: resolution now trusts whatever `coreweave` is
  aliased to locally, with no URL pinned in code to verify against (low-likelihood risk, but
  real — accepted).
- **Rejected: vendor the chart locally** (download + untar once, reference via Pulumi's `path`
  option instead of `repositoryOpts` + remote chart name) — a validated pattern from the Pulumi
  community for this exact gap (see [pulumi-kubernetes#935](https://github.com/pulumi/pulumi-kubernetes/issues/935)
  comment thread). Removes the live-fetch dependency entirely, the most durable of the three
  options. Rejected for this cluster (2026-07-17): adds a re-vendoring maintenance step on every
  chart version bump; revisit if the local-repo-registration prerequisite proves too fragile in
  practice (e.g., if CI adoption of `helm repo add` turns out harder than expected).
- **Tried and abandoned: forcing resolution order via `depends_on`.** Not a third alternative,
  a dead end — see below. Recorded so nobody re-attempts it expecting a different result.

**Root cause (upstream, confirmed via Pulumi's own issue tracker, not fixed):**
[pulumi-kubernetes#935](https://github.com/pulumi/pulumi-kubernetes/issues/935) — Pulumi's Helm
resources do not cache the fetched/rendered chart at all; every `preview`/`up` re-fetches live.
Open since 2020. A maintainer comment on
[#1504](https://github.com/pulumi/pulumi-kubernetes/issues/1504) confirms this causes exactly
this class of failure ("network issues" during resolution). Neither upstream issue names the
specific "whichever resolves in a given slot fails" behavior found here — worth filing as a new,
more precise report if this becomes a recurring blocker.

**Current state: mitigated.** `pulumi preview`/`up` on `cw-us-west-04a` resolves cleanly with the
`repository_opts` drop in place and `helm repo add coreweave <url>` registered locally (README
Prerequisites). The live cluster was never at risk from the underlying bug either way — every
failure observed during the investigation happened during Pulumi's diff computation, before any
resource was actually created, updated, or deleted (verified via `iris cluster status` / live
`kubectl` checks after every failed attempt).

### Traefik/cert-manager CRD-registration race — investigated, accepted

`TraefikAddon` applies the `ClusterIssuer`/`Middleware`/`Ingress` CustomResources right after
their Helm `Release` (`depends_on=[cert_manager_release]` / `[traefik_release]`), with no
explicit CRD-readiness wait — unlike `install_cw_network.py`'s `wait_for_crd` (up to 120s,
hard-fails with a clear error if the CRD never shows up). Flagged in code review as a
theoretical race: the CRD a CustomResource needs might not be registered in the API server yet
even though the Helm Release that ships it has been created.

**Checked, not assumed:** [pulumi-kubernetes#1446](https://github.com/pulumi/pulumi-kubernetes/issues/1446)
confirms the provider does retry a CustomResource create when its CRD isn't found yet — but only
5 times, hardcoded, not configurable via `custom_timeouts` (a user-reported attempt to raise it
that way was ignored by the provider). Still an open enhancement request upstream, not fixed.

**Why this is an accepted risk, not a bug to fix here:** `depends_on=[cert_manager_release]`
already orders CR creation after the Release's own readiness check, which waits for
cert-manager's Deployments to have healthy pods — image pull + container start + probe passes
takes far longer in practice than API-server CRD registration, so by the time the Release is
"complete" the CRD is essentially always already Established. The bounded 5-retry provider
behavior is a safety net under that for the remaining edge case. This combination has never
failed across all `pulumi up`/`preview` runs against `cw-us-west-04a` this session,
including the real adoption apply. Reproducing `install_cw_network.py`'s imperative
`wait_for_crd` polling loop inside a declarative Pulumi resource graph would need a custom
Dynamic Provider — real, ongoing complexity for a risk that's small and already has a
provider-level safety net. Revisit only if this actually manifests in practice (retry a
`pulumi up` if it ever does — the fix upstream is a higher retry budget, not a wait added here).

## "Easy to land now" vs deferred — the call

Three things land now:

1. **The GB200 cluster config** — a new cluster's `provisioning:` + `scale_groups` flow
   straight through `derive_nodepools` + `IrisRbac` with no new code (below).
2. **The federation egress IP reservations** — `GcpStaticAddresses`, the GCP arm's first
   slice: the two `google_compute_address` reservations, adopted by import so the pinned IPs
   are never reassigned (row 10 / §Egress IPs).
3. **The CoreWeave-side allowlist** — the config input `IngressSpec.federation_allow_sources`
   with a constant default, read later by the deferred federation-ingress enforcement.

Everything else needs a deferred component (rows 3–7, 12) or a later GCP slice (rows 8–9): the
CKS cluster object, Kueue, Traefik/cert-manager, object storage, the finelog server, and the
Secret Manager signing keys. Those are **documented, not added** — a bucket name or secret with
no component to create it would be dead config. Each row above names its exact landing site so
the follow-up slices are turnkey.

## What landed now: `cw-us-east-08a` (GB200)

Added in this PR, using only the existing schema + pt1 components:

- [`lib/iris/config/cw-us-east-08a.yaml`](../../../lib/iris/config/cw-us-east-08a.yaml) —
  the cluster: `cpu-erapids` pool (4× `cd-gp-i64-erapids`) + `gb200` pool (216× `gb200-4x`,
  4 GB200 GPUs each = 864 Blackwell GPUs = 12 NVL72 racks), both pinned warm.
- [`infra/iac/Pulumi.cw-us-east-08a.yaml`](../../../infra/iac/Pulumi.cw-us-east-08a.yaml) —
  the Pulumi stack pointer.

No finelog config ships for it — that's blocked on the signing key + hub registration (see
§finelog auth secrets), so the Iris config omits `finelog:` and the controller uses MemStore
until the key exists.

Also landed: the **GCP address stub** — [`iac/gcp/addresses.py`](../../../infra/iac/src/iac/gcp/addresses.py)
(`GcpStaticAddresses`), the `GcpProvisioning` schema, the `Provider.GCP` dispatch in
`__main__.py`, the `provisioning:` block on [`lib/iris/config/marin.yaml`](../../../lib/iris/config/marin.yaml),
and the [`Pulumi.marin.yaml`](../../../infra/iac/Pulumi.marin.yaml) stack. `pulumi-gcp` joins
the deps. This is the GCP arm's first slice (§Egress IPs); its live `pulumi preview --import`
is operator-run.

`derive_nodepools` yields `cw-use08a-cpu-erapids` (min=max=4) and `cw-use08a-gb200`
(min=max=72). 72 is a multiple of 18, which the GB200 NVL72 rack constraint requires
(instances deploy in whole racks of 18 nodes; a NodePool count must be a multiple of 18 —
[CoreWeave docs](https://docs.coreweave.com/platform/instances/gpu/gb200-4x)).

### Confirmed / still to confirm

Confirmed **one cluster** — one CKS cluster / one Iris cluster / one Pulumi stack holds the
whole delivery (mirroring how `cw-us-east-02a` = 256 H100 and `cw-rno2a` = 512 H100 are each
one cluster). Still to confirm before it goes live:

1. **Placeholders to confirm against the console** once the hardware lands: the CKS cluster
   name (`marin-gb200`), `kube_context` (`marin-gb200_US-EAST-08A`), and `US-EAST-08A` as the
   exact region string. The `gb200-4x` SKU, 4 GPUs/node, and 144 vCPU / 960 GB / 30.72 TB
   node spec are from CoreWeave's published `gb200-4x` page.
2. **Secrets + registrations still needed** (rows 8–9): mint `iris-cw-us-east-08a-signing-key`
   and `finelog-cw-us-east-08a-signing-key`, register this cluster's public halves in the peer
   `federation_peers` blocks and the finelog hub `marin.yaml`, then add the finelog deploy
   config + the `finelog:` block to the Iris config.
