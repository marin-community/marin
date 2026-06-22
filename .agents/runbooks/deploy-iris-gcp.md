---
name: deploy-iris-gcp
description: Deploy Iris on GCP — push a merged controller/iris fix to a running cluster, or stand one up from scratch in a new region.
---

# Runbook: Deploy Iris on GCP

**When you're here:** You're deploying Iris to a GCP cluster — either pushing a
merged controller/iris fix to a cluster that is already running (the common
case), or bringing a cluster up from scratch in a region that has none yet. For
the CoreWeave/k8s variant, see [deploy-iris-coreweave](deploy-iris-coreweave.md).

## Before you touch anything

- **Never restart a running controller without an explicit human yes.** A restart
  drops in-flight scheduling state; controller-only vs full restart is the
  operator's call, not the script's.
- **A restart alone ships nothing.** marin pins `iris-controller:latest`, so a
  restart only re-pulls whatever `:latest` already is. Rebuild the image from the
  merged commit *first*, then restart, then verify the running git-hash. Skipping
  the rebuild is the classic trap — it once cost ~5 red-canary days.
- **A from-scratch stand-up creates real, billed resources** — a GCS bucket, a
  controller VM, an Artifact Registry repo, TPU/CPU autoscale groups. Have a
  reason and a region; don't stand one up speculatively.
- **Co-locate everything in one region.** The bucket, the iris `remote_state_dir`,
  and the finelog `remote_log_dir` must share a region. Nothing cross-checks this,
  and a mismatch silently bills cross-region egress on every state/log access.

## Push a merged fix to a running cluster

Build and verify are mechanical and scripted (`scripts/iris/controller_deploy.py`);
the restart between them is the one human-gated step.

**1. Rebuild the image from the merged commit.**

```bash
uv run scripts/iris/controller_deploy.py build --ref main
```

Triggers the "Ops - Docker Images" workflow and streams it to completion. Until
it finishes, `:latest` still resolves to the old code.

**2. Restart the controller — human-gated.** Get an explicit yes, then restart on
the freshly-built image. Controller-only vs full restart and `--skip-checkpoint`
recovery are the operator's judgment; command reference in lib/iris/OPS.md
"Controller Restart". Do not automate this step.

**3. Verify the controller is running the new code.**

```bash
uv run scripts/iris/controller_deploy.py verify --cluster marin --ref main
```

Compares the controller's reported git-hash to the ref's **tree** hash — the
image bakes in `HEAD^{tree}`, not the commit sha, so eyeballing `iris cluster
status` against `git log` never matches. Exit 0 = match, 1 = mismatch (rebuild
didn't run, restart shipped a stale `:latest`, or the pull was cached), 2 =
controller reports an unknown hash.

## Stand up a new cluster from scratch (the bootstrap order)

Bottom-up, fail-fast — each step is a hard prerequisite for the next, so don't
proceed past a failing one. Each points at the artifact that owns the *how*;
don't restate its flags here, they drift.

**1. Create the main `marin-<region>` GCS bucket (out-of-band, unscripted).**
Create `gs://marin-<region>` in the target region, single-region, matching the
existing fleet's storage class. This is the one step with no script — by design:
`configure_buckets.py` refuses to create the main bucket (location/class/naming
are load-bearing and not safe to auto-create). Everything downstream derives
bucket paths from `rigging.filesystem.REGION_TO_DATA_BUCKET`, so the region's
entry must exist there and the bucket name must match it.

**2. Service accounts, roles, APIs, impersonation —** `lib/iris/scripts/setup_iam.py`.

```bash
uv run lib/iris/scripts/setup_iam.py --help
```

Creates the `iris-controller` and `iris-worker` SAs if missing, grants their
project roles, enables the required GCP APIs, and wires impersonation/act-as for
CI and operator principals. The SA ids, role sets, and API list are the script's
module constants — read them there, do not copy them into config. A **new GCP
project** (not just a new region) gets its Iris identity here; a same-project new
region mostly re-confirms existing bindings.

**3. Provision the `ghcr-mirror` Artifact Registry pull-through repo — UNSCRIPTED.**
GCP VMs do not pull from `ghcr.io` directly; the bootstrap rewrites every
`ghcr.io/...` tag to `<multi-region>-docker.pkg.dev/<project>/ghcr-mirror/...`, a
pull-through cache. That AR remote repo must already exist for the cluster's
multi-region. **There is no script for this step.** If you're standing up in `us`
or `europe`, the mirror already exists fleet-wide and you can skip; for a
genuinely new multi-region, provision the `ghcr-mirror` remote repo there first
and extend `_ZONE_PREFIX_TO_MULTI_REGION` in the GCP `bootstrap.py` (it raises at
bootstrap for an unsupported zone prefix).

**4. Bucket lifecycle + registry cleanup policy.**

```bash
uv run infra/configure_buckets.py --bucket marin-<region>
uv run infra/configure_gcp_registry.py marin --region=<region>
```

`configure_buckets.py` disables soft-delete and applies the `tmp/ttl=Nd/`
lifecycle rules; it preserves hand-curated rules and is safe to re-run.
`configure_gcp_registry.py` sets the 30-day / keep-16 cleanup policy on the
region's `marin` AR repo. Both read the region list from `REGION_TO_DATA_BUCKET`,
so Step 1's entry must be in place. `--dry-run` previews either.

**5. Author the two cluster configs — same region.**
- `lib/iris/config/<cluster>.yaml` — model on `lib/iris/config/marin.yaml`:
  `remote_state_dir`, `controller` (zone/machine_type/image), `tpu_pools` /
  `scale_groups` for the region's accelerators, `user_budgets`, and
  `log_server_config: <name>`.
- `lib/finelog/config/<cluster>.yaml` — model on `lib/finelog/config/marin.yaml`:
  `name`, `remote_log_dir`, deployment zone/SA.

The iris config's `log_server_config:` must equal the finelog config's `name:`
(in `marin.yaml`, `log_server_config: marin` ↔ `name: finelog-marin`), or
`iris cluster log-server up` won't find the deployment.

**6. Bring up the log server, then the controller.**

```bash
iris --config=lib/iris/config/<cluster>.yaml cluster log-server up
iris --config=lib/iris/config/<cluster>.yaml cluster start
```

`log-server up` provisions/refreshes the finelog deployment named by
`log_server_config` (idempotent). `cluster start` pins `:latest` to the current
git sha, builds and pushes the controller/worker/task images, creates the
controller GCE VM, and bootstraps it. The connection selector (`--config=` /
`--cluster=`) is a global flag *before* the subcommand.

## Verify

- **`/health`** on the controller returns OK.
- **Git-hash** matches what you deployed — use the `verify` command above, not a
  glance at `iris cluster status` (it shows the tree hash, not a commit sha).
- **`iris cluster status`** shows the controller healthy and the autoscaler
  reporting the pools you declared. Worker groups at zero replicas is expected on
  a cold cluster — they scale up on first job.
- **First image pull is slow on a cache miss** through the AR mirror — expected,
  not a failure.

## Why the bootstrap order matters

Each step is a hard prerequisite for the next: the bucket backs lifecycle/state,
IAM lets the controller VM boot and pull, and the GHCR→AR mirror must exist
before `cluster start`. Two seams are unscripted and easy to forget because every
existing `us`/`europe` cluster already has them: main-bucket creation (Step 1,
deliberate) and AR-mirror provisioning for a genuinely new multi-region (Step 3,
a real gap).

## See also

- `lib/iris/scripts/setup_iam.py`, `infra/configure_buckets.py`,
  `infra/configure_gcp_registry.py` — the bootstrap scripts (own the flags).
- `lib/iris/config/marin.yaml` / `lib/finelog/config/marin.yaml` — the YAML
  shapes to copy.
- [deploy-iris-coreweave](deploy-iris-coreweave.md) — the CoreWeave/k8s variant.
