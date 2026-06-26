---
name: tpu-ci-fleet
description: Stand up, rotate the PAT for, or recover the self-hosted TPU CI runner fleet that backs GitHub Actions TPU jobs.
---

# Runbook: Operate the self-hosted TPU CI runner fleet

**When you're here:** TPU CI jobs are queuing with no runner to pick them up, a
runner went unresponsive, the GitHub PAT expired, or you're standing the fleet
up from scratch. The fleet is a controller VM running a `vm_manager` daemon that
keeps a fixed count of preemptible v5litepod-4 TPU VMs alive across three zones;
each VM auto-registers as a self-hosted GitHub Actions runner.

**TL;DR:**
- The scripts under `infra/tpu-ci/` are the gold-standard authority; this
  runbook just routes you to the right one. There is no README and no OPS.md
  coverage — this is it.
- **Runners won't register (expired/under-permissioned PAT)** →
  `MARIN_CI_TOKEN=<new-pat> uv run infra/tpu-ci/setup.py update-token`
  (recreates all VMs; ~10 min).
- **Runners stuck/unresponsive** → `uv run infra/tpu-ci/setup.py restart-runners`
  (deletes all TPU VMs; controller recreates within ~10 min).
- **One VM is failing setup** → diagnose with `vm_manager.py check-logs` /
  `debug-setup`; reset a wedged TPU device with `clean-tpu.sh`.
- Fleet shape (controller name/zone, per-zone TPU counts, accelerator type,
  runner labels) lives in `infra/tpu-ci/config.py` — read it, don't restate it.

## Before you touch anything

- **`update-token`, `restart-runners`, and `teardown` all DELETE every TPU VM in
  the fleet.** Any TPU CI job running at that moment dies and gets re-queued.
  These VMs are preemptible and the daemon recreates them, so this is routine
  recovery — but check the Actions queue first and avoid bouncing the fleet
  mid-merge-train if you can wait.
- **`teardown` destroys the controller too** (`infra/tpu-ci/setup.py teardown`).
  Only run it for a full decommission/rebuild — after it, nothing recreates the
  VMs until you re-run `setup-controller`.
- **PAT permission requirement:** the token in `MARIN_CI_TOKEN` must have the
  fine-grained **Administration: read & write** repository permission (or classic
  `repo` scope). `update-token` verifies this against the GitHub API *before*
  storing it and exits non-zero if it's missing
  (`infra/tpu-ci/setup.py:444` docstring + verification block at 459–489) — so a
  bad token fails fast without deleting anything.
- **Baseline:** before recovering, confirm the symptom is the fleet and not a
  workflow bug — check the runner list in the repo's Actions settings and the
  controller logs (`controller-logs` below). A red TPU job with healthy runners
  is a code problem, not a fleet problem.

## Diagnose

**Is it the PAT, the daemon, or one bad VM?**

1. **No runners online at all / runners "offline" in GitHub settings.** The
   controller daemon recreates VMs but they can't register → almost always an
   expired or under-permissioned PAT. Confirm by reading the controller's
   monitor logs:
   ```bash
   uv run infra/tpu-ci/setup.py controller-logs -f
   ```
   (Shows the controller VM's startup-script and `tpu-monitor` service logs;
   `infra/tpu-ci/setup.py:514`.) Registration failures surface here. Go to
   **Resolve → PAT expired.**

2. **Some runners present but jobs hang / runner unresponsive.** A VM is up but
   its runner process or TPU is wedged. Pull per-VM diagnostics — runner
   journald logs, `_diag` logs, runner/docker process state:
   ```bash
   uv run infra/tpu-ci/vm_manager.py check-logs <vm-name> -n 200
   ```
   (`infra/tpu-ci/vm_manager.py:677`; VM names follow the `tpu-ci` prefix +
   zone, see `config.py`.) If only one VM is bad, go to **Resolve → single VM**;
   if several are, **Resolve → restart the fleet.**

3. **A VM never finishes setup (no runner ever appears for it).** Re-run its
   startup script in place to see where it dies:
   ```bash
   uv run infra/tpu-ci/vm_manager.py debug-setup <vm-name>
   ```
   (`infra/tpu-ci/vm_manager.py:818`.)

## Resolve

Least-destructive first.

**PAT expired (the common one).** Rotate the token. This stores the new PAT in
Secret Manager and deletes all TPU VMs so they re-register with it:
```bash
MARIN_CI_TOKEN=<new-pat> uv run infra/tpu-ci/setup.py update-token
```
The controller recreates VMs within ~10 minutes (`infra/tpu-ci/setup.py:444`).
The new PAT needs the Administration read&write permission noted above; the
command refuses a token that lacks it.

**Single VM wedged.** First try diagnostics (`check-logs`) and re-running setup
(`debug-setup`) from Diagnose. If the TPU device itself is stuck — held
`/dev/vfio` handles, stale lockfile, jobs failing to acquire the chip — reset it
on that VM with the cleanup script (kills TPU-holding processes, clears
`/tmp/libtpu_lockfile`, resets the PCI device, waits for `/dev/vfio/0`):
`infra/tpu-ci/clean-tpu.sh`. If the VM stays bad, just let it be replaced —
deleting it (or restarting the fleet) makes the daemon recreate a clean one.

**Several runners stuck / want a clean fleet.** Recreate every TPU VM:
```bash
uv run infra/tpu-ci/setup.py restart-runners
```
Deletes all TPU VMs; the controller's monitor loop rebuilds them within ~10 min
(`infra/tpu-ci/setup.py:501`). Watch with `controller-logs -f`.

**First-time bring-up / full rebuild.** Create the controller VM and supporting
infra (stores PAT in Secret Manager, builds+pushes the `tpu-ci` image, creates
the controller, which then spins up the TPU fleet from `config.py`):
```bash
MARIN_CI_TOKEN=<pat> uv run infra/tpu-ci/setup.py setup-controller
```
(`infra/tpu-ci/setup.py:400`.) To rebuild just the CI Docker image without
touching VMs: `uv run infra/tpu-ci/setup.py build-image`
(`infra/tpu-ci/setup.py:436`).

## Verify

Coming back up is not the same as registering. Confirm runners are actually
serving jobs:

- **Controller logs show VMs created and registering** —
  `uv run infra/tpu-ci/setup.py controller-logs -n 100`; look for the monitor
  recreating the expected per-zone counts (`config.py` `TPU_ZONES_CONFIG`) with
  no registration errors.
- **Runners appear online in GitHub** — the repo's Actions → Runners list should
  show the fleet's runners (labels `tpu`, `self-hosted`, `tpu-ci` from
  `config.py`) as idle/online, not offline.
- **A real TPU job picks up and goes green** — re-run the queued/failed TPU
  workflow, or push a trivial change that triggers it. Tag-matching and
  "VM exists" are necessary but not sufficient; only a passing job proves the
  runner serves work.

## Why this happens

The fleet is preemptible by design: `TPU_ACCELERATOR_TYPE = "v5litepod-4"` VMs
get reclaimed by GCP, so a `vm_manager` daemon on the controller continuously
reconciles actual VM count against the desired per-zone counts in
`config.py:TPU_ZONES_CONFIG`. That's why "delete the VMs and wait ~10 min" is
the standard recovery for almost everything — the daemon's job is to bring them
back. Each VM registers itself against a GitHub registration token derived from
the PAT in Secret Manager, so a PAT that expired or lost the Administration
permission breaks *registration* without breaking the VMs — you get VMs that
exist but no runners online, which is why the PAT path looks different from a
wedged-TPU path. `config.py` is the single source of truth for fleet shape;
changing counts/zones there and re-running the relevant command is how the fleet
is resized.

## See also

- `infra/tpu-ci/setup.py` — controller/fleet lifecycle CLI: `setup-controller`,
  `teardown`, `build-image`, `update-token`, `restart-runners`,
  `controller-logs`. Command docstrings are the authority for flags.
- `infra/tpu-ci/vm_manager.py` — per-VM CLI: `check-logs`, `debug-setup`,
  `debug-tpu` (rsync repo + run pytest in the CI Docker image on a VM).
- `infra/tpu-ci/config.py` — single source of truth for controller VM, zone→count
  map, accelerator type, runner labels, image coordinates.
- `infra/tpu-ci/clean-tpu.sh` — wedged-TPU device reset (run on the affected VM).
- lib/iris/OPS.md "CI Workflows" — broader CI/Docker-image workflow reference
  (the GitHub-side workflows these runners execute).
