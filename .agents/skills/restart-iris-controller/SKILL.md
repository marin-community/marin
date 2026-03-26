---
name: restart-iris-controller
description: Restart the Iris controller with state preservation. Use when asked to restart, redeploy, or upgrade the Iris controller or cluster.
---

# Skill: Restart Iris Controller

Restart the Iris controller process on a remote cluster (GCP) with full state
preservation. Workers survive the restart — only the controller VM is cycled.

Read first: @lib/iris/AGENTS.md

## Required Info

1. `config` — Iris config path. Resolve shorthand names:
   - `marin` / `marin_prod` -> `lib/iris/examples/marin.yaml`
   - `marin_dev` / `marin-dev` -> `lib/iris/examples/marin-dev.yaml`
   - `coreweave` -> `lib/iris/examples/coreweave.yaml`

If the user says "restart the controller" without specifying a cluster, ask which one.

## Concepts: Controller Restart vs Cluster Restart

| Operation | Command | Scope | Downtime |
|---|---|---|---|
| **Controller restart** | `iris cluster controller restart` | Controller VM only; workers unaffected | Seconds (checkpoint → new VM → restore) |
| **Full cluster restart** | `iris cluster restart` | Controller + all workers torn down, cold start | Minutes; all running jobs killed |

**Default to controller restart** unless the user explicitly asks for a full cluster restart.
A full cluster restart (`iris cluster restart`) kills all running jobs cluster-wide — always
confirm with the user before doing this.

## Workflow

### Phase 1: Pre-flight — Local Dry-Run Validation

Before touching the remote controller, validate locally that the controller
code boots and restores state correctly. This catches config errors, migration
issues, and import failures before they take down production.

```bash
uv run iris --config=<CONFIG> cluster controller serve --dry-run
```

This will:
- Load the cluster config
- Download the latest checkpoint from remote storage (e.g. `gs://marin-us-central2/iris/marin/state`)
- Restore the SQLite database (jobs, tasks, workers, autoscaler state)
- Start the scheduling loop (computes assignments but suppresses dispatch)
- Start the dashboard on `localhost:10000`

**What dry-run suppresses:** task dispatch, VM creation/termination, checkpoint
writes, worker kill requests, autoscaler refresh, profile collection.

**What dry-run runs:** full startup, DB restore, scheduling loop, dashboard + RPC.

Wait for the log line `Controller started successfully` before proceeding.

### Phase 2: Dashboard Verification with Playwright

With the dry-run controller running locally, use Playwright to verify state
was restored correctly. The dashboard is a Vue 3 SPA at `http://localhost:10000`.
Routes use hash-based routing (`/#/`, `/#/fleet`, etc.).

**Navigate and verify each critical tab:**

1. **Jobs tab** (`/#/`)
   - Confirm job count is non-zero and matches expectations
   - Check that RUNNING jobs still show as RUNNING
   - Check that PENDING jobs have sensible diagnostics
   - Look for unexpected FAILED or UNSCHEDULABLE states

2. **Workers tab** (`/#/fleet`)
   - Confirm worker count matches expectations
   - All workers should show "Healthy" with recent heartbeats ("Xs ago")
   - Note: in dry-run mode, heartbeats will eventually go stale since the
     controller isn't dispatching — check within the first ~60s

3. **Autoscaler tab** (`/#/autoscaler`) — **skip in dry-run mode**.
   The autoscaler and provider bundle are not created in dry-run
   (`main.py:106`), so this tab will be empty. Verify it post-restart instead.

4. **Health endpoint**
   ```bash
   curl -s http://localhost:10000/health
   # Expected: {"status": "ok"}
   ```

**Capture a pre-restart snapshot** of job/worker counts for post-restart comparison:

```bash
# Via CLI (preferred — works without dashboard)
uv run iris --config=<CONFIG> cluster status
```

Once verified, Ctrl-C the dry-run process.

### Phase 3: Capture Pre-Restart Baseline from Production

Before restarting, record the current production state for comparison:

```bash
# Get current job and worker counts from the live controller
uv run iris --config=<CONFIG> cluster status
```

Note:
- Total jobs (especially RUNNING count)
- Total workers and healthy count
- Any jobs in PENDING with autoscaler reasons

### Phase 4: Execute Controller Restart

```bash
uv run iris --config=<CONFIG> cluster controller restart
```

This performs three steps automatically:
1. **Checkpoint** — RPC call to the running controller (default 300s timeout)
2. **Build** — Fresh controller image pinned to current git SHA
3. **Restart** — Terminates old controller VM, launches new one

If the checkpoint is timing out, use `--skip-checkpoint`:
```bash
uv run iris --config=<CONFIG> cluster controller restart --skip-checkpoint
```

The new controller auto-discovers and restores from the latest checkpoint.

### Phase 5: Post-Restart Verification

Wait ~30-60s for the new controller to boot and restore, then verify:

1. **Health check:**
   ```bash
   uv run iris --config=<CONFIG> cluster status
   ```
   - Controller should report Running: True, Healthy: True
   - Worker count should match pre-restart baseline
   - No workers should have gone unhealthy

2. **Dashboard verification with Playwright:**
   The controller exposes the dashboard directly. Establish the SSH tunnel:
   ```bash
   uv run iris --config=<CONFIG> cluster dashboard
   ```
   Then use Playwright to navigate the controller URL printed by the tunnel and verify:
   - Jobs tab: job count matches pre-restart baseline; RUNNING jobs still RUNNING
   - Workers tab: all workers healthy with recent heartbeats
   - Autoscaler tab: scale groups restored, no unexpected backoff states

3. **Compare against baseline:**
   - Job count should be >= pre-restart count (new jobs may have arrived)
   - Worker count should match exactly
   - No previously-RUNNING jobs should have become FAILED or WORKER_FAILED

### Phase 6: Report

Print a summary:
```
Controller restart complete.
  Config:          <CONFIG>
  Pre-restart:     <N> jobs, <M> workers (<H> healthy)
  Post-restart:    <N'> jobs, <M'> workers (<H'> healthy)
  Jobs preserved:  yes/no
  Workers healthy: yes/no
```

## Error Recovery

- **Checkpoint timeout:** Use `--skip-checkpoint`. The controller will restore from
  the last periodic checkpoint (hourly by default). Some recent state may be lost.
- **Controller won't start:** Check bootstrap logs:
  ```bash
  uv run iris --config=<CONFIG> cluster vm logs iris-controller-<LABEL_PREFIX>
  ```
- **Workers show unhealthy after restart:** Workers reconnect automatically via
  heartbeat. Wait 2-3 heartbeat cycles (~30s). If still unhealthy, the controller
  may have lost worker tracking — check autoscaler state.
- **Job count mismatch:** If jobs were lost, the checkpoint may have been stale.
  Check the checkpoint timestamp in the controller startup logs.

## Rules

- **NEVER do a full cluster restart** (`iris cluster restart`) without explicit user approval.
  This kills all running jobs.
- **Always dry-run locally first** before restarting the remote controller.
- **Always capture pre-restart baseline** so you can verify state preservation.
- **Do not skip the Playwright verification** — visual inspection catches issues
  (like blank zones, stale heartbeats) that CLI checks miss.
