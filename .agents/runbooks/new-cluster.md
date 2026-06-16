---
name: new-cluster
description: Stand up a brand-new Iris cluster in a new GCP region from scratch — the bootstrap order.
---

# Runbook: Stand up a brand-new Iris cluster from scratch (the bootstrap order)

**When you're here:** You're bringing up a Marin/Iris cluster in a region (or
project) that has none yet, and the from-scratch ordering lives in people's
heads and scattered docstrings rather than one place. This runbook is that
ordering. For the CoreWeave/k8s variant see `stand-up-coreweave-cluster`.

**TL;DR:** Bottom-up, fail-fast. Each step depends on the one before:
1. Create the main `marin-<region>` GCS bucket **out-of-band** (the one unscripted prerequisite).
2. `setup_iam.py` — service accounts, project roles, APIs, impersonation.
3. Provision the `ghcr-mirror` Artifact Registry pull-through repo for the region — **currently unscripted** for new multi-regions.
4. `configure_buckets.py` (lifecycle/soft-delete) + `configure_gcp_registry.py` (cleanup policy).
5. Author `lib/iris/config/<cluster>.yaml` + `lib/finelog/config/<cluster>.yaml`, **same region**.
6. `iris --config=... cluster log-server up`, then `iris --config=... cluster start`.
7. Verify `/health`, controller git-hash, `iris cluster status`.

## Before you touch anything

- **This creates real, billed GCP resources** — a GCS bucket, a controller GCE
  VM, an Artifact Registry repo, TPU/CPU autoscale groups. Don't stand one up
  speculatively; have a reason and a region.
- **Bucket creation is deliberately out-of-band.** `configure_buckets.py`
  *refuses* to create the main bucket and only manages lifecycle/soft-delete on
  one that already exists (`infra/configure_buckets.py:170` —
  *"main buckets must be created out-of-band; skipping."*). Create it yourself
  in the target region first. This is intentional: bucket location, class, and
  naming are load-bearing and not safe to auto-create.
- **Do not proceed past a failing step.** The order is a dependency chain — IAM
  before bootstrap, bucket before lifecycle, mirror before `cluster start`. A
  half-applied step leaves a cluster that comes up but misbehaves later (bad
  pulls, missing roles). Fix the failing step, don't skip it.
- **Cross-region cost guardrail:** the bucket, the finelog `remote_log_dir`, and
  the iris `remote_state_dir` must all be in the **same** region as the cluster.
  Nothing cross-checks this today (see Steps 5). Co-locating avoids per-read
  cross-region egress charges.

## Steps

Least-destructive / most-foundational first. Each step points at the runnable
artifact that owns the *how*; don't restate its flags here — they drift.

**1. Create the main `marin-<region>` GCS bucket (out-of-band, unscripted).**
Create `gs://marin-<region>` in the target region, single-region, matching the
existing fleet's storage class. This is the one step with no script — by design
(see guardrail above). Everything downstream derives bucket paths from
`rigging.filesystem.REGION_TO_DATA_BUCKET`, so the region's entry must exist
there and the bucket name must match it.

**2. Service accounts, roles, APIs, impersonation —** `lib/iris/scripts/setup_iam.py`.
```bash
uv run lib/iris/scripts/setup_iam.py --help
```
Creates the `iris-controller` and `iris-worker` SAs if missing, grants their
project roles, enables the required GCP APIs, and wires impersonation/act-as for
CI and operator principals. The SA ids, role sets, and API list are the script's
module constants (`setup_iam.py:26` onward) — read them there, do not copy them
into config. If the cluster is in a **new GCP project** (not just a new region),
this is where the project gets its Iris identity; a same-project new region
mostly re-confirms existing bindings.

**3. Provision the `ghcr-mirror` Artifact Registry pull-through repo — UNSCRIPTED.**
GCP VMs do not pull from `ghcr.io` directly; the bootstrap rewrites every
`ghcr.io/...` image tag to `<multi-region>-docker.pkg.dev/<project>/ghcr-mirror/...`,
a pull-through cache (`bootstrap.py:56` `rewrite_ghcr_to_ar_remote`). That AR
remote repo must already exist for the cluster's multi-region (`us` or
`europe`). **There is no script for this step** — it's provisioned by hand.
A zone whose region prefix has no mirror raises at bootstrap:
`bootstrap.py:48` rejects `asia`/`me` prefixes with *"has no AR remote repo
provisioned"*. If you're standing up in `us` or `europe`, the mirror already
exists fleet-wide and you can skip; if you're in a genuinely new multi-region,
provision the `ghcr-mirror` remote repo (pull-through cache for `ghcr.io`) there
first and extend `_ZONE_PREFIX_TO_MULTI_REGION` in `bootstrap.py:31`.

**4. Bucket lifecycle + registry cleanup policy.**
```bash
uv run infra/configure_buckets.py --bucket marin-<region>
uv run infra/configure_gcp_registry.py marin --region=<region>
```
`configure_buckets.py` disables soft-delete and applies the `tmp/ttl=Nd/`
lifecycle rules (docstring at `infra/configure_buckets.py:5`; it preserves
hand-curated rules and is safe to re-run). `configure_gcp_registry.py` sets the
30-day / keep-16 cleanup policy on the region's `marin` AR repo (docstring at
`infra/configure_gcp_registry.py:5`; also `infra/README.md:69`). Both read the
region list from `REGION_TO_DATA_BUCKET`, so Step 1's entry must be in place.
Use `--dry-run` on either to preview.

**5. Author the two cluster configs — same region.**
- `lib/iris/config/<cluster>.yaml` — model on `lib/iris/config/marin.yaml`:
  `remote_state_dir` (e.g. `gs://marin-<region>/iris/<cluster>/state`),
  `controller` (zone/machine_type/image), `tpu_pools` / `scale_groups` for the
  region's accelerators, `user_budgets`, and `log_server_config: <name>`.
- `lib/finelog/config/<cluster>.yaml` — model on `lib/finelog/config/marin.yaml`:
  `name`, `remote_log_dir` (e.g. `gs://marin-<region>/finelog/<cluster>`),
  deployment zone/SA.

**Region-match requirement (no automated cross-check):** the iris
`remote_state_dir`, the finelog `remote_log_dir`, and the Step-1 bucket must all
point at the *same* `marin-<region>`. Nothing validates this — a mismatched
region silently bills cross-region egress on every state/log access. The iris
config's `log_server_config:` value must equal the finelog config's `name:`
(in `marin.yaml`, `log_server_config: marin` ↔ `name: finelog-marin`), or
`iris cluster log-server up` will not find the deployment.

**6. Bring up the log server, then the controller.**
```bash
iris --config=lib/iris/config/<cluster>.yaml cluster log-server up
iris --config=lib/iris/config/<cluster>.yaml cluster start
```
`log-server up` provisions/refreshes the finelog deployment named by
`log_server_config` (idempotent; `cluster.py:464`). `cluster start` pins
`:latest` to the current git sha, builds and pushes the controller/worker/task
images, creates the controller GCE VM, and bootstraps it
(`cluster.py:258`). The connection selector (`--config=` / `--cluster=`) is a
global flag *before* the subcommand (lib/iris/OPS.md:11 "Connection selectors").

## Verify

- **`/health`** on the controller returns OK (dashboard at the config's
  `dashboard_url`; same probe the deploy runbook uses).
- **Controller git-hash** matches the commit you stood up from — confirm the
  running image tag, not just that it came up. See
  [deploy-controller-fix](deploy-controller-fix.md) "Verify" for how.
- **`iris --config=... cluster status`** shows the controller healthy and the
  autoscaler reporting the pools you declared (lib/iris/OPS.md:19 "Cluster
  Lifecycle"). Worker groups at zero replicas is expected on a cold cluster —
  they scale up on first job.
- **First image pull is slow on cache miss** through the AR mirror — expected,
  not a failure (see deploy-controller-fix "Why").

## Why this ordering

Each step is a hard prerequisite for the next, which is why fail-fast matters:

- **Bucket before lifecycle/state:** `configure_buckets.py`, `remote_state_dir`,
  and `remote_log_dir` all resolve `gs://marin-<region>` — a missing bucket
  makes Step 4 skip and Step 6 fail to checkpoint.
- **IAM before bootstrap:** the controller VM boots *as* `iris-controller` and
  pulls images as `artifactregistry.reader`; without the Step-2 roles the VM
  comes up but can't pull or provision TPUs.
- **Mirror before start:** GCP VMs never pull `ghcr.io` directly. With no
  `ghcr-mirror` remote repo for the multi-region, bootstrap either rewrites to a
  non-existent repo (pull fails) or raises outright for unsupported prefixes
  (`bootstrap.py:48`).
- **Two unscripted seams to remember:** main-bucket creation (Step 1, deliberate)
  and AR-mirror provisioning for a new multi-region (Step 3, a real gap). Both
  are easy to forget because every existing `us`/`europe` cluster already has
  them.

## See also

- `infra/README.md` — the cluster/region list, `configure_buckets.py` and
  `configure_gcp_registry.py` usage, the cleanup-policy "when to use".
- `infra/configure_buckets.py:5` / `infra/configure_gcp_registry.py:5` — script
  docstrings (own the flags).
- `lib/iris/scripts/setup_iam.py` — SA/role/API/impersonation bootstrap (Step 2).
- `lib/iris/src/iris/cluster/backends/gcp/bootstrap.py:31` — GHCR→AR mirror map
  and the unsupported-prefix raise (Step 3).
- `lib/iris/config/marin.yaml` / `lib/finelog/config/marin.yaml` — the
  per-cluster YAML shapes to copy (Step 5).
- lib/iris/OPS.md:11 "Connection selectors", lib/iris/OPS.md:19 "Cluster
  Lifecycle" — selector flag + status command reference.
- [deploy-controller-fix](deploy-controller-fix.md) — controller image/git-hash
  verify, AR slow-pull behavior.
- `stand-up-coreweave-cluster` — the CoreWeave/k8s variant of this procedure.
- `.agents/projects/ops-runbooks.md` — the runbook batch this belongs to.
