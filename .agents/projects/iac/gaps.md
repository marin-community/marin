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
| 3 | CKS cluster object + VPC + kubeconfig | manual (console / CW TF provider) | IaC-next | `CoreweaveCluster` (TODO in [cluster.py](../../../infra/iac/src/iac/coreweave/cluster.py); `CksClusterSpec` exists) |
| 4 | Kueue: `cks-kueue` chart, Topology CRs, `cw-ib` ResourceFlavor, `iris-cq` ClusterQueue, **namespace-scoped webhooks** | `install_kueue.py --with-queues` | IaC-next | `KueueAddon` (`KueueProvisioningSpec` exists) |
| 5 | Traefik + cert-manager + HTTP-01 ClusterIssuers | `install_cw_network.py` | IaC-next | `TraefikAddon` (`IngressSpec` exists) |
| 6 | **Federation ingress**: IP-locked `Ingress` + `ipAllowList` Middleware over the whole controller host | `install_cw_network.py` | enforcement IaC-next; **allowlist input landed** | fold into `TraefikAddon`; reads `IngressSpec.federation_allow_sources` (see §Egress IPs) |
| 7 | Object-storage buckets + access keys (`s3://marin-<region>`) | manual console + `configure_buckets.py` (lifecycle) | IaC-next | `ObjectStorage` (`ObjectStorageSpec` exists); bucket *lifecycle* stays out (spec §7) |
| 8 | **iris controller signing key** (`iris-<cluster>-signing-key`) | `iris cluster init-keys` → GCP Secret Manager | **gap** | IaC-gcp Secret Manager (see §Signing secrets) |
| 9 | **finelog forwarding signing key** (`finelog-<cluster>-signing-key`) | minted by hand → GCP Secret Manager | **gap** | IaC-gcp Secret Manager (see §Signing secrets) |
| 10 | Federation **egress** IP reservations (`34.27.183.11`, `35.254.13.19` = `iris-marin-fed-egress` / `iris-marin-dev-fed-egress`) | reserved by hand in `hai-gcp-models` | **IaC-landed** (GCP arm) | `GcpStaticAddresses` ✅ ([gcp/addresses.py](../../../infra/iac/src/iac/gcp/addresses.py)); the CoreWeave-side allowlist is `IngressSpec.federation_allow_sources` (see §Egress IPs) |
| 11 | DNS: `iris-cw-<cluster>.oa.dev` CNAME → Traefik LB FQDN | manual (Cloudflare) | gap | out of scope now; bridged Cloudflare provider later, or stay manual |
| 12 | finelog server Deployment (in-cluster) | `finelog deploy up <cluster>` | IaC-next (planned) | `FinelogServer` component (a later CoreWeave slice; needs the finelog signing key) |
| 13 | Iris runtime objects: ConfigMap, `iris-task-env` Secret, LocalQueue, PriorityClasses, controller Deployment + Service, state PVC | `start_controller()` | **Iris (by design)** | stays in Iris (spec §4) |

Rows 1–2, 10 are done. Rows 3–7 are the sequenced CoreWeave follow-ups already in the design;
row 12 is a planned CoreWeave slice. Rows 8–11 are the newly-surfaced gaps this analysis adds;
rows 8–10 are the ones rjpower called out. Row 13 is deliberately *not* IaC.

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

The **private** halves are the Secret Manager secrets (the gap: creating them). The
**public** halves are config, hand-registered in two places: the cluster's
`auth.federation_peers` (peer controllers' keys) and the hub's
[`marin.yaml`](../../../lib/finelog/config/marin.yaml) finelog `auth` (accepted forwarder
keys).

This gap is a **hard gate**, not just a to-do: `test_every_bundled_sender_names_a_cluster_some_bundled_hub_trusts`
in [`lib/finelog/tests/test_config.py`](../../../lib/finelog/tests/test_config.py) fails a
bundled finelog config that forwards as a cluster no hub's jwt layer trusts. So a forwarding
finelog server for `cw-us-east-08a` cannot even be committed until its key is minted and
registered — which is why **this PR ships no finelog config for the cluster** and omits the
`finelog:` block from its Iris config (the controller uses the in-process MemStore until
then). This is the same posture as the `ci-*` cluster configs.

**Why not "easy to land now":** GCP Secret Manager is the **GCP arm** (deferred), and the
key material is generated by `init-keys`, not declared. The clean IaC shape is *provision
the empty secret + IAM* in Pulumi and keep `init-keys` as the generator that writes a
version — Pulumi should not hold private key bytes even encrypted. **Landing plan:** with the
GCP arm, add a `SecretsSpec`/`GcpSecrets` component that declares the two Secret Manager
secrets + accessor IAM per cluster; then mint the keys, register the public halves, and add
the finelog deploy config + `finelog:` block to the Iris config.

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
