---
name: babysit-job
description: Monitor/babysit a job continuously and recover on failure. Use when asked to babysit, monitor, or watch a job, pipeline, workflow, or training run.
---

# Skill: Babysit Job

Monitor a job continuously and recover on failure. For **Zephyr pipelines**,
delegate to **babysit-zephyr** instead. Otherwise, follow this skill — Iris is
the execution backend.

## Required Info

1. `job_id` — Iris job ID in canonical format `/<user>/<job>` (e.g., `/dlwh/iris-run-train_tiny_model_tpu-20260302-185630`)
2. `config` — Iris config path (e.g., `lib/iris/examples/marin.yaml`). When the user
   refers to a cluster by shorthand name (e.g., "marin_dev", "marin-dev", "marin",
   "coreweave"), resolve it to the matching config file under `lib/iris/examples/`.
   Common mappings:
   - `marin` / `marin_prod` -> `lib/iris/examples/marin.yaml`
   - `marin_dev` / `marin-dev` -> `lib/iris/examples/marin-dev.yaml`
   - `coreweave` -> `lib/iris/examples/coreweave.yaml`
3. `resubmit_command` — exact Iris submit command for resubmission; must include `--no-wait`
4. For Marin TPU training jobs, use `--extra marin:tpu` (not `--extra marin:cpu`)
5. For TPU jobs, the resubmit command must request TPU resources with `--tpu <variant>`.
   `--reserve <variant>` only holds capacity; it does not attach TPU devices to the task container.

Example resubmit command:
`uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --extra marin:tpu --tpu v5litepod-16 -- python experiments/tutorials/train_tiny_model_tpu.py`

If any required field is missing, ask for it before proceeding.

## Scope

- Recovery is stop then resubmit at the job level.
- Cluster-level actions are out of scope. Do not restart, recreate, or otherwise
  mutate the cluster unless the user gives explicit consent in the current thread.
- For TPU bad-node errors, escalate to **debug-tpu**.

## Monitoring Ownership and Duration

- Assign a single monitoring owner when the loop starts.
- Keep this loop running until one of the following:
  - the job reaches a terminal state and the user has acknowledged next action
  - a user-specified stopping point is reached
  - an unrecoverable error is found and reported to the user
- Do not stop early after seeing first loss lines, first eval, or first W&B link.
- Expect monitoring to commonly take 4-5 hours for ferry-scale runs.
- For GPT Codex specifically: unless otherwise directed, do not end your turn just
  to give a status update; keep monitoring until terminal state or until the user's
  goal is reached.
- If the user requests continuous monitoring, do not end the turn while monitoring
  is active; continue until terminal state, a user-specified stopping point, or an
  unrecoverable error.
- If handoff is needed, transfer ownership explicitly with: current `job_id`,
  latest error/signal, W&B link(s), and resubmission metadata.

## Cadence and Tooling Notes

- Cadence default after startup stabilization is `sleep 570`.
- Startup stabilization sequence (after submit/resubmit):
  - once the job is submitted, sleep `120` and check for immediate failure
  - if still alive, switch to the normal `570` cadence
- Tool-runtime workaround:
  - keep one long-running monitor process/session
  - poll the same session in ~30 second chunks as needed by tool runtime limits
  - repeated no-output polls are expected while waiting for the next 570-second check
- Single monitor process rule:
  - run only one active monitor loop per job to avoid duplicate SSH tunnel and
    port-binding conflicts
- Sleep must be foreground (max ~10 min due to tool timeout).
- Loop control is at agent level, not bash.

## MCP-Assisted Monitoring

When testing or using `marin-mcp-babysitter`, keep the MCP server resident and
verify the job through MCP tools, not only through Iris CLI commands.

- Keep the controller tunnel and MCP server in named, restartable sessions
  (`screen`, `tmux`, or one long-running exec session). Record session names,
  ports, and log paths in the state file.
- Start MCP with a stable local controller URL and streamable HTTP transport:
  `uv run --package marin marin-mcp-babysitter --controller-url <URL> --cluster <CLUSTER> --transport streamable-http --host 127.0.0.1 --port <PORT>`
- Verify with `iris_job_summary` and `iris_tail_logs`. For heartbeat-style
  monitoring, report: job state, latest progress/tick/log line, timestamp, and
  error signal.
- If the MCP server is reachable but tool calls fail with connection refused to
  the controller URL, restart only the smoke-test tunnel/session. Do not restart
  or mutate the Iris cluster.
- If a sandbox blocks direct localhost TCP probes, run the probe inside an
  existing long-lived session and write a small JSON result under `scratch/`.
- For bounded smoke tests, create a thread heartbeat only after the job is
  submitted, MCP is reachable, and at least one expected log/progress line has
  appeared. Delete the heartbeat and stop the smoke-test sessions/listeners when
  the job reaches the expected terminal state.

## State File

Write to `scratch/<create_timestamp>_monitoring_state.json`, create the `scratch`
directory if needed. `<create_timestamp>` should have format `YYYYMMDD-HHMM`.
Track `restart_count` to detect flapping. Add MCP fields when a resident MCP
server is part of the monitoring setup. State file allows resume after context reset.

```json
{
  "ts": <timestamp_ms>,
  "job_id": "<JOB_ID>",
  "config": "<IRIS_CONFIG_PATH>",
  "mcp_url": "http://127.0.0.1:<PORT>/mcp",
  "tunnel_session": "<SESSION_NAME>",
  "server_session": "<SESSION_NAME>",
  "tunnel_log": "scratch/<TUNNEL_LOG>",
  "server_log": "scratch/<SERVER_LOG>",
  "resubmit_command": "<IRIS_JOB_RUN_COMMAND_WITH_NO_WAIT>",
  "restart_count": 0
}
```

## Loop

```
1. SLEEP
   - if just submitted/restarted: sleep 120 once
   - otherwise: sleep 570

2. CHECK LOGS
   uv run iris --config <CONFIG> job logs --since-seconds 900 --include-children <JOB_ID> | rg -i -e "loss|error|traceback|exception|resource_exhausted|oom|compiler_base\.cc:2587|program hbm requirement|largest program allocations|ownerdiederror|dead node|node death|autoscaler unsatisfied resources|no accelerator found|failed_precondition|device or resource busy"

3. CHECK STATUS
   uv run iris --config <CONFIG> job list --json --prefix <JOB_ID>

   Terminal success: JOB_STATE_SUCCEEDED
   Terminal non-success: JOB_STATE_FAILED, JOB_STATE_KILLED, JOB_STATE_WORKER_FAILED, JOB_STATE_UNSCHEDULABLE
   Non-terminal: JOB_STATE_PENDING, JOB_STATE_BUILDING, JOB_STATE_RUNNING

   If `pending_reason` indicates worker scale-up/capacity wait, treat as scheduler
   capacity wait — do not run cluster update/recreate/restart actions. Continue
   waiting on cadence, or stop+resubmit only if user explicitly asks.

   Treat RUNNING as controller-level signal only; confirm allocation via expected
   W&B run when possible.

3a. ON TERMINAL STATE / OOM-LIKE SIGNAL — get a structured per-task summary
   (final state, exit, duration, peak memory) instead of grepping logs:

   uv run iris --config <CONFIG> job summary --json <JOB_ID>

   Fast postmortem: e.g. "13/14 shards peaked near the container memory limit
   and failed with exit 137" → cgroup OOM, raise `--memory` on resubmit.

4. PRINT W&B RUN IDS/LINKS (once per training run)
5. REPORT PROGRESS (format: ~<current>/<exact_max>)
6. EVALUATE (terminal? error? stalled? -> recover or continue)

7. RECOVER (STOP -> RESUBMIT)
   - If current job is still non-terminal, stop it first:
     uv run iris --config <CONFIG> job stop <JOB_ID>
   - Then resubmit:
     <RESUBMIT_COMMAND>
   - Capture `job_id` from output (line like `Job submitted: /<user>/<job>`).
   - Iris nuance:
     - if `resubmit_command` omits `--job-name`, Iris auto-generates a fresh id each resubmission.
     - if `resubmit_command` uses a fixed `--job-name`, Iris may reuse the same id
       after terminal completion by replacing the finished job.
   - Update state file: `job_id=<NEW_JOB_ID>`, `restart_count += 1`.
   - Go to step 1.
```

## Fixing Small Bugs

When EVALUATE detects an error, before recovery:

1. Analyze logs — look for `Traceback`, `Error`, `Exception`. Identify file and line.
2. If small fix (typo, missing import, wrong variable name): fix it, then RECOVER.
3. If complex (architectural, unclear cause, broad investigation): report to user, exit loop.

Small-fix examples: `NameError`, `ImportError`, `SyntaxError`, obvious `KeyError`.

Complex examples: OOM, TPU/XLA HBM exhaustion, distributed training failures,
data loading issues, unclear multi-file stack traces.

## Error Patterns

- Treat TPU/XLA HBM reports as failure even without literal OOM:
  - `Program hbm requirement ...`
  - `Largest program allocations in hbm`
- If progress stalls across multiple intervals with `OwnerDiedError`, dead node,
  or unsatisfied resources -> mark `degraded` and notify user.
- If same error repeats after one fix attempt, do not retry blindly; report to user.

## When to Escalate

- Debug Zephyr pipeline issues -> **debug-zephyr-job**
- Debug TPU bad-node errors -> **debug-tpu**
- Debug running tasks with `iris task exec` -> **debug-iris-job**

## Notes

- Iris `job list --prefix` requires canonical job names (`/<user>/<job>`), not short names.
- Iris monitoring is job-level; cluster updates are not part of normal recovery.
