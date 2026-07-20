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
[`config.py`](../../../infra/pulumi/src/iac/config.py) · **IaC-gcp** = belongs to the deferred
GCP arm · **Iris** = stays in `start_controller()` by design · **manual** = console / script,
not yet modeled anywhere.

| # | Prerequisite | Today | IaC status | Lands in |
|---|---|---|---|---|
| 1 | Reserved NodePools (per scale group) | `ensure_nodepools()` | **IaC-pt1** | `CoreweaveCluster` ✅ |
| 2 | Namespace + controller RBAC (SA, ClusterRole, Binding) | `ensure_rbac()` | **IaC-pt1** | `IrisRbac` ✅ |
| 3 | CKS cluster object + VPC + kubeconfig | manual (console / CW TF provider) | **Manual (permanent)** | `CoreweaveCluster` records it as config (`CksClusterSpec`, exported outputs — [cluster.py](../../../infra/pulumi/src/iac/coreweave/cluster.py)); no CoreWeave TF provider bridged, no CoreWeave API credentials — see below |
| 4 | Kueue: `cks-kueue` chart, Topology CRs, `cw-ib` ResourceFlavor, `iris-cq` ClusterQueue, **namespace-scoped webhooks** | `install_kueue.py --with-queues` | **IaC-landed** | `KueueAddon` ✅ |
| 5 | Traefik + cert-manager + HTTP-01 ClusterIssuers | `install_cw_network.py` | **IaC-landed** | `TraefikAddon` ✅ |
| 6 | **Federation ingress**: IP-locked `Ingress` + `ipAllowList` Middleware over the whole controller host | `install_cw_network.py` | **IaC-landed** | `TraefikAddon` ✅ — reads `IngressSpec.federation_allow_sources` (see §Egress IPs) |
| 7 | Object-storage buckets + access keys (`s3://marin-<region>`) | manual console + `configure_buckets.py` (lifecycle) | IaC-next | `ObjectStorage` (`ObjectStorageSpec` exists); bucket *lifecycle* stays out (spec §7) |
| 8 | **iris controller signing key** (`iris-<cluster>-signing-key`) | `iris cluster init-keys` → GCP Secret Manager | **Manual** | `init-keys` does the whole thing (create-if-absent + write + IAM); Pulumi only warns (non-fatal, `pulumi.log.warn` in `__main__.py`) when `auth.signing_key` is unset, with the exact command to run (see §Signing secrets) |
| 9 | **finelog forwarding signing key** (`finelog-<cluster>-signing-key`) | minted by hand → GCP Secret Manager | **Open** | same posture as row 8 once a cluster needs finelog forwarding (see §Signing secrets) |
| 10 | Federation **egress** IP reservations (`34.27.183.11`, `35.254.13.19` = `iris-marin-fed-egress` / `iris-marin-dev-fed-egress`) | reserved by hand in `hai-gcp-models` | **IaC-landed** (GCP arm) | `GcpStaticAddresses` ✅ ([gcp/addresses.py](../../../infra/pulumi/src/iac/gcp/addresses.py)); the CoreWeave-side allowlist is `IngressSpec.federation_allow_sources` (see §Egress IPs) |
| 11 | DNS: `iris-cw-<cluster>.oa.dev` CNAME → Traefik LB FQDN | manual (Cloudflare) | **Manual (deferred)** | considered and deferred — see §DNS CNAME below |
| 12 | finelog server Deployment (in-cluster) | `finelog deploy up <cluster>` | IaC-next (planned) | `FinelogServer` component (a later CoreWeave slice; needs the finelog signing key) |
| 13 | Iris runtime objects: ConfigMap, `iris-task-env` Secret, LocalQueue, PriorityClasses, controller Deployment + Service, state PVC | `start_controller()` | **Iris (by design)** | stays in Iris (spec §4) |
| 14 | AR pull-through caches: `ghcr-mirror` (ghcr.io), `docker-mirror` (Docker Hub), each in `us` + `europe`, plus 30d-delete / keep-16 cleanup | manual console / `gcloud artifacts` | **IaC-landed** (GCP arm) | `GcpArtifactRegistries` ✅ ([gcp/registries.py](../../../infra/pulumi/src/iac/gcp/registries.py)); declared in `provisioning.gcp.registries`, consumed by `GcpWorkerProvider.resolve_image` (see [image-push.md](../../../lib/iris/docs/image-push.md)) |

Rows 1–2, 4–6, 8, 10, 14 are done. Rows 3 and 11 are deferred (see below). Row 7 is the
remaining sequenced CoreWeave follow-up already in the design; row 12 is a planned CoreWeave
slice. Row 9 is still open. Row 13 stays in Iris by design.

## The two gaps rjpower named

### Egress IPs (row 6 + row 10)

"Egress IPs for the iris controller" is really three coupled resources, split across clouds —
two now modeled, one deferred:

- **The GCP reservations (row 10) — landed.** The marin / marin-dev controllers federate
  *into* each CoreWeave controller, and their egress IPs (`34.27.183.11`, `35.254.13.19`) are
  reserved as `iris-marin-fed-egress` / `iris-marin-dev-fed-egress` in project
  `hai-gcp-models`. These are now `google_compute_address` resources in the GCP arm —
  `GcpStaticAddresses` ([gcp/addresses.py](../../../infra/pulumi/src/iac/gcp/addresses.py)), the
  GCP arm's first slice, on the `marin` stack. Each pins its IP so adoption imports the live
  reservation without ever reassigning an IP baked into a CoreWeave allowlist. (Confirmed
  against the live reservations: both are EXTERNAL, `us-central1`, in use.)

- **The CoreWeave-side allowlist (part of row 6) — landed as a config input.** Which sources
  the CoreWeave federation route admits is `IngressSpec.federation_allow_sources`, defaulting
  to the `MARIN_FEDERATION_EGRESS_SOURCES` constant in
  [`config.py`](../../../infra/pulumi/src/iac/config.py) (the same values as the
  `FEDERATION_ALLOW_SOURCES` constant in
  [`install_cw_network.py`](../../../lib/iris/scripts/install_cw_network.py)).

- **The enforcement (CoreWeave side, row 6) — deferred.** The `ipAllowList` Middleware +
  IP-locked Ingress that admits only those sources over the whole controller host. This is a
  k8s object currently applied by `install_cw_network.py`, folded into `TraefikAddon` when
  that lands — it reads `federation_allow_sources`.

Until the federation-ingress component consumes the field, the allowlist has two copies (the
config constant and the script constant); the config docstring flags the sync obligation, and
the follow-up deletes the script constant so the input is the sole source.

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

Row 9 (finelog) is a **hard gate**: `test_every_bundled_sender_names_a_cluster_some_bundled_hub_trusts`
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

Comparable in size to `TraefikAddon` itself once a Cloudflare token exists. The DNS CNAME stays
a manual, printed-instruction step (`install_cw_network.py`'s own "Done. To finish wiring..."
output) until then.

### Pulumi Helm chart resolution (2026-07-17)

`TraefikAddon`'s two Helm `Release`s (`traefik`, `cert-manager`) intermittently failed
`pulumi preview`/`up` with `chart "coreweave/<x>" version "<y>" not found in
https://charts.core-services.ingress.coreweave.com repository`, though the chart and version
exist and resolve via the real `helm` CLI. `KueueAddon`'s `cks-kueue` Release (same repo, same
`repository_opts` pattern) never failed across the investigation.

**Current mitigation: drop `repository_opts` on `traefik`/`cert-manager`** (`src/iac/coreweave/traefik.py`).
Verified clean across 8 consecutive `pulumi preview` runs (0 failures), versus 5/5 failures the
same session with `repository_opts` present. Without `repository_opts`, `chart="coreweave/traefik"`
resolves through the local `helm` CLI's repo config (populated by `helm repo add coreweave <url>`),
so that registration becomes a prerequisite — documented in one place, `infra/pulumi/README.md`
Prerequisites. `KueueAddon`'s `cks-kueue` Release is untouched (still `repository_opts`), never
having failed.

**Empirical findings (direct testing):**
- No local cache is involved. Pointing `HELM_CACHE_HOME` at an empty directory left it untouched;
  deleting and regenerating the `helm` CLI's own cache (`~/Library/Caches/helm`) changed nothing.
- Chart-independent. Forcing `traefik` to resolve before `cert-manager` (via `depends_on`) failed
  `traefik` 5/5; flipping the order failed `cert-manager` 5/5. Whichever resolves first in a
  forced pair fails.
- No positional rule. Making both Releases depend on `KueueAddon`'s (always-succeeding) Release so
  they race each other failed exactly one per run, but which one varied. A fully serial
  `kueue -> cert-manager -> traefik` chain failed `cert-manager` 8/8 — worse than the intermittent
  baseline. No dependency-graph shape produced a reliably clean run, so ordering via `depends_on`
  is a dead end.

**Root cause (upstream, open):** [pulumi-kubernetes#935](https://github.com/pulumi/pulumi-kubernetes/issues/935)
— Pulumi's Helm resources cache nothing; every `preview`/`up` re-fetches the chart live. Open
since 2020. A maintainer comment on [#1504](https://github.com/pulumi/pulumi-kubernetes/issues/1504)
attributes this failure class to "network issues" during resolution. Neither issue names the
"whichever resolves first in a forced pair fails" behavior seen here; worth a precise upstream
report if it recurs.

**Structuring the out-of-band `helm repo add`.** The `repository_opts` drop trades a flaky
in-program resolution for a local prerequisite: `pulumi preview`/`up` needs `helm repo add
coreweave <url>` run first, which breaks on a fresh checkout or an ephemeral CI runner. The Helm
`Release` resolves its chart during `preview` (the engine's diff render, per #935), so any fix
has to make the repo available at preview time. Three ways to handle it, in preference order:

1. **Fold it into the program with `command.local.run`** (recommended; not yet adopted). The
   `pulumi-command` provider's `local.run` is an invoke that executes during program evaluation —
   on every `preview` and `up`, before the engine resolves the Release charts — so calling
   `helm repo add coreweave <url> --force-update` (idempotent) at the top of the CoreWeave build
   registers the repo ahead of chart resolution. The out-of-band step and its duplicated
   instructions collapse into one place: the program. Needs the `helm` binary on PATH (already
   required) and the `pulumi-command` dependency. Use `local.run` (the invoke), not
   `command.local.Command` (the resource): the resource's `create` runs only at `up`, after chart
   resolution, so it would not help `preview`
   ([pulumi-command#49](https://github.com/pulumi/pulumi-command/issues/49)). Confirm with a live
   `pulumi preview` on `cw-us-west-04a` before adopting.
2. **Vendor the charts as local paths.** `helm.v3.Release`'s `chart` field accepts a path to an
   unpacked chart or a `.tgz` (local-path support added in provider v0.65,
   [pulumi-kubernetes#1732](https://github.com/pulumi/pulumi-kubernetes/issues/1732)), so
   `chart="./charts/traefik"` drops the remote repo and network entirely, makes `preview`
   self-contained, and sidesteps the #935 re-fetch flakiness. Cost: re-vendor (`helm pull`) on
   every chart bump, and the tarballs live in-tree.
3. **Keep it out-of-band** (current). One documented step in `infra/pulumi/README.md` Prerequisites,
   plus an explicit step in any CI workflow that runs a CoreWeave preview/up (none exists yet —
   spec.md §9 Phase 1). Residual risk: resolution trusts whatever `coreweave` is aliased to
   locally, with no URL pinned in code.

Option 1 answers the review question directly (fold the step into `pulumi up`); option 2 is the
most robust. Both are larger than doc-only option 3 and want a live preview to confirm, so this PR
keeps option 3 and records 1–2 as the next step.

**Current state.** `pulumi preview`/`up` on `cw-us-west-04a` resolves cleanly with the
`repository_opts` drop and `helm repo add coreweave <url>` registered locally. The live cluster
was never at risk: every failure occurred during Pulumi's diff computation, before any resource
was created, updated, or deleted (verified via `iris cluster status` / `kubectl` after each).

### Traefik/cert-manager CRD-registration race

`TraefikAddon` applies the `ClusterIssuer`/`Middleware`/`Ingress` CustomResources right after
their Helm `Release` (`depends_on=[cert_manager_release]` / `[traefik_release]`), with no
explicit CRD-readiness wait — `install_cw_network.py` uses `wait_for_crd` (up to 120s,
hard-fails if the CRD never shows up). Code review flagged the theoretical race: a
CustomResource's CRD might not be registered in the API server yet even though the Helm Release
that ships it has been created.

[pulumi-kubernetes#1446](https://github.com/pulumi/pulumi-kubernetes/issues/1446) confirms the
provider retries a CustomResource create when its CRD isn't found — 5 times, hardcoded, not
configurable via `custom_timeouts`. Still an open upstream enhancement request.

The race is an accepted, bounded risk. `depends_on=[cert_manager_release]` orders CR creation
after the Release's readiness check, which waits for cert-manager's Deployments to have healthy
pods; image pull, container start, and probe passes take longer than API-server CRD
registration, so the CRD is Established by the time the Release completes. The 5-retry provider
behavior covers the remaining edge case. This held across every `pulumi up`/`preview` run against
`cw-us-west-04a` this session, including the adoption apply. Replicating `wait_for_crd`'s polling
loop inside a declarative Pulumi graph would need a custom Dynamic Provider. Revisit if it
manifests in practice; the upstream fix is a higher retry budget.

## What lands now vs. what's deferred

Three things land now:

1. **The GB200 cluster config** — a new cluster's `provisioning:` + `scale_groups` flow
   straight through `derive_nodepools` + `IrisRbac` with no new code (below).
2. **The federation egress IP reservations** — `GcpStaticAddresses`, the GCP arm's first
   slice: the two `google_compute_address` reservations, adopted by import so the pinned IPs
   are never reassigned (row 10 / §Egress IPs).
3. **The CoreWeave-side allowlist** — the config input `IngressSpec.federation_allow_sources`
   with a constant default, read later by the deferred federation-ingress enforcement.

The rest is deferred or manual: the CKS cluster object (row 3), object storage (row 7), the
finelog server (row 12), and the Secret Manager signing keys (rows 8–9). These are documented
here with no component to create them — a bucket name or secret with no creator would be dead
config. Each row above names its exact landing site so the follow-up slices are turnkey.

## What landed now: `cw-us-east-08a` (GB200)

Added in this PR, using only the existing schema + pt1 components:

- [`lib/iris/config/cw-us-east-08a.yaml`](../../../lib/iris/config/cw-us-east-08a.yaml) —
  the cluster: `cpu-erapids` pool (4× `cd-gp-i64-erapids`) + `gb200` pool (216× `gb200-4x`,
  4 GB200 GPUs each = 864 Blackwell GPUs = 12 NVL72 racks), both pinned warm.
- [`infra/pulumi/Pulumi.cw-us-east-08a.yaml`](../../../infra/pulumi/Pulumi.cw-us-east-08a.yaml) —
  the Pulumi stack pointer.

No finelog config ships for it — that's blocked on the signing key + hub registration (see
§finelog auth secrets), so the Iris config omits `finelog:` and the controller uses MemStore
until the key exists.

Also landed: the **GCP address stub** — [`iac/gcp/addresses.py`](../../../infra/pulumi/src/iac/gcp/addresses.py)
(`GcpStaticAddresses`), the `GcpProvisioning` schema, the `Provider.GCP` dispatch in
`__main__.py`, the `provisioning:` block on [`lib/iris/config/marin.yaml`](../../../lib/iris/config/marin.yaml),
and the [`Pulumi.marin.yaml`](../../../infra/pulumi/Pulumi.marin.yaml) stack. `pulumi-gcp` joins
the deps. This is the GCP arm's first slice (§Egress IPs); its live `pulumi preview --import`
is operator-run.

`derive_nodepools` yields `cw-use08a-cpu-erapids` (min=max=4) and `cw-use08a-gb200`
(min=max=216 = 12 racks). 216 is a multiple of 18, which the GB200 NVL72 rack constraint
requires (instances deploy in whole racks of 18 nodes; a NodePool count must be a multiple of
18 — [CoreWeave docs](https://docs.coreweave.com/platform/instances/gpu/gb200-4x)).

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
