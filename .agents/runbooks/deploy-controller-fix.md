---
name: deploy-controller-fix
description: Ship a merged controller/iris fix, or restart the controller, without shipping a stale :latest.
---

# Runbook: Deploy a merged controller/iris fix (rebuild the image, then restart)

**When you're here:** A controller/iris fix is merged to `main` but the cluster
still behaves like the old code — or you just need to bounce the controller. The
trap is that "merged" does not mean "deployed."

**TL;DR:**
- marin pins `iris-controller:latest` (`config/marin.yaml:33`). A restart only
  re-pulls whatever `:latest` *currently* is — it does **not** rebuild from your
  merged commit.
- To deploy a merged fix: **rebuild the image first** (`gh workflow run "Ops -
  Docker Images"`), wait for it to push, **then** `iris cluster controller
  restart`. Restarting against a stale `:latest` ships nothing.
- **Verify** the running controller is the `:<git-short-hash>` you expect — not
  merely that it came back up.

## Before you touch anything

- **`iris cluster controller restart` is the safe one** — controller only,
  seconds of downtime, workers and running jobs survive.
- **`iris cluster restart` tears down EVERYTHING** — controller *and* all
  workers; every running job dies. This is the load-bearing guardrail:
  **NEVER run a full `iris cluster restart` without explicit user approval.**
  (This runbook is the canonical home for that policy; AGENTS.md and the
  `restart-iris` skill carry only the bare prohibition and point here.) You do
  not need a full restart to deploy a controller fix — the controller restart is
  sufficient.
- **Capture a baseline before restarting** so you can confirm workers/jobs
  survived: `iris cluster status` (see lib/iris/OPS.md:19 "Cluster Lifecycle").
- **Dry-run the new controller config locally first**: `iris cluster controller
  serve --dry-run` (see lib/iris/OPS.md:27 "Controller Restart"). Catches a
  broken config before you take downtime.

## Diagnose

**Hypothesis: the fix is merged but the controller is running stale code.**
This is the default and most common case. Don't assume the merge deployed.

1. **Is the fix's commit actually in a built `:latest`?** The "Ops - Docker
   Images" workflow only rebuilds on `workflow_dispatch` or Sunday 02:00 UTC
   (see lib/iris/OPS.md:289 "CI Workflows", the `ops-docker-images.yaml` row).
   If your fix merged Monday and it's Tuesday, no rebuild has run — `:latest` is
   last Sunday's image. Check recent runs:
   `gh run list --workflow "Ops - Docker Images" -R marin-community/marin -L 5`
   and compare the run's commit against the one carrying your fix.

2. **What is the controller actually running right now?** `iris cluster status`
   reports the controller image/tag. If it shows `:latest` resolved to an old
   digest (or you redeployed and the behavior is unchanged), the image was never
   rebuilt — go to Resolve, rebuild first.

   - If a rebuild *did* run and pushed, but the controller still runs old code,
     suspect a **slow/cached AR pull** rather than a missing build (see "Why").

**Look-alike:** weird scheduling, jobs stuck PENDING, reservation-taint
behavior. A stale controller is a common root cause of these — if you've ruled
out scheduler/capacity causes in [diagnose-stuck-pending-job](diagnose-stuck-pending-job.md),
come back here and confirm the controller is actually on the fixed image.

## Resolve

Least-destructive first. You almost never need more than steps 1–3.

1. **Rebuild the image.** Trigger the build that pushes
   `iris-{controller,worker,task}:latest` to GHCR:
   ```bash
   gh workflow run "Ops - Docker Images" -R marin-community/marin --ref main
   ```
   (syntax: lib/iris/OPS.md:301). Or, if it's not urgent, just wait for the
   Sunday 02:00 UTC build. Do **not** skip this step and restart — that ships the
   stale `:latest`.

2. **Wait for the build to finish and push.** Watch it:
   `gh run watch <run-id> -R marin-community/marin`, or view a failed one with
   `gh run view <run-id> -R marin-community/marin --log-failed | tail -50`
   (lib/iris/OPS.md:303). The job pushes to `ghcr.io/marin-community/...`; GCP
   VMs pull through the Artifact Registry mirror (see lib/iris/docs/image-push.md).

3. **Restart the controller** (after baseline + dry-run from above):
   ```bash
   iris --cluster=marin cluster controller restart
   ```
   Controller only — workers and jobs survive (lib/iris/OPS.md:27). The
   connection selector (`--cluster`/`--config`) is a global flag, before the
   subcommand (lib/iris/OPS.md:11 "Connection selectors").

4. **Fallback — checkpoint times out.** If the restart hangs checkpointing
   controller state, use `iris cluster controller restart --skip-checkpoint`
   (lib/iris/OPS.md:34). This restores from the last *periodic* checkpoint, so
   some recent in-memory state may be lost — acceptable to unstick a deploy,
   not a default.

## Verify

The restart returning is **not** proof the fix shipped. Confirm the image:

- `iris cluster status` should show the controller running the
  `:<git-short-hash>` (or `:latest` resolved to the digest) that matches your
  merged fix — not the previous one. If it still shows the old code, the rebuild
  didn't run or the pull was cached; loop back to Resolve step 1.
- **Confirm the blast radius stayed small:** compare against the baseline you
  captured — workers still present, running jobs still RUNNING. If workers
  vanished, you ran a full restart, not a controller restart.
- **Confirm the behavior is actually fixed** — re-run the thing that exposed the
  bug (e.g. the canary ferry, the stuck submission). Image tag matching is
  necessary but not sufficient.

## Why this happens

The pin `iris-controller:latest` (`config/marin.yaml:33`) **decouples "merged"
from "deployed."** `:latest` is a moving tag that only advances when the "Ops -
Docker Images" workflow rebuilds it (`ops-docker-images.yaml`, dispatch or
Sunday). A controller restart re-pulls `:latest`; if nothing rebuilt it, the
restart faithfully redeploys the *old* code. The merge changed the source tree,
not the tag your cluster pulls.

This bit hard: on 2026-06-08, fix #6141 for the reservation-holder taint merged
Jun 3 but **the marin controller was never redeployed**, so the bug recurred and
the daily TPU canary stayed red for ~5 days
(`.agents/ops/2026-06-08-canary-ferry-reservation-taint-timeouts.md:18`). The
redeploy is what finally cleared mode 1.

A second-order trap: even after a correct rebuild+restart, the AR pull-through
mirror caches per continent. A **slow first pull** on cache miss is expected and
not a failure (lib/iris/docs/image-push.md:136); but a *stale* cached tag can
mask a fresh build. If status shows old code despite a confirmed build, suspect
the pull, not the workflow.

## See also

- lib/iris/OPS.md:19 "Cluster Lifecycle" and lib/iris/OPS.md:27 "Controller
  Restart" — restart command syntax, dry-run/baseline workflow,
  `--skip-checkpoint`.
- lib/iris/OPS.md:289 "CI Workflows" — the `ops-docker-images.yaml` row and the
  `gh workflow run` / `gh run view` commands.
- lib/iris/docs/image-push.md — GHCR push auth, the AR pull-through mirror, and
  slow-pull troubleshooting.
- `restart-iris` skill — thin pointer here; carries only the never-full-restart
  prohibition.
- [diagnose-stuck-pending-job](diagnose-stuck-pending-job.md) — a stale
  controller is a common root cause of weird scheduling; rule that in here.
- `.agents/ops/2026-06-08-canary-ferry-reservation-taint-timeouts.md` — the
  ~5-day red-canary recurrence from skipping the redeploy.
