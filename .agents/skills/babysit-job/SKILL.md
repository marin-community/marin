---
name: babysit-job
description: Monitor and, when authorized, recover a specified non-Zephyr Iris job by stopping and resubmitting it.
---

# Babysit an Iris job

Use `babysit-zephyr` only when the requested target is a Zephyr pipeline.

## Required Info

1. `job_id` in canonical `/<user>/<job>` format.
2. Iris config path. Resolve shorthand against `lib/iris/config/`; common mappings:
   - `marin` / `marin_prod` -> `lib/iris/config/marin.yaml`
   - `marin_dev` / `marin-dev` -> `lib/iris/config/marin-dev.yaml`
   - `coreweave` / `cw-us-east-02a` -> `lib/iris/config/cw-us-east-02a.yaml`; `cw-rno2a` -> `lib/iris/config/cw-rno2a.yaml`
3. Exact resubmit command, including `--no-wait`.
4. For Marin TPU training jobs, use `--extra marin-core:tpu` (not `--extra marin-core:cpu`)
5. For TPU jobs, the resubmit command must request TPU resources with `--tpu <variant>`.
   `--reserve <variant>` only holds capacity; it does not attach TPU devices to the task container.

Example resubmit command:
`uv run iris --config lib/iris/config/marin.yaml job run --no-wait --extra marin-core:tpu --tpu v5litepod-16 -- python -m experiments.tutorials.train_tiny_model --device v5litepod-16 --dataset tinystories`

Ask for any missing field before monitoring.

## Scope

- Recovery is stop then resubmit at the job level.
- Cluster-level actions are out of scope. Do not restart, recreate, or otherwise
  mutate the cluster unless the user gives explicit consent in the current thread.
- For TPU bad-node errors, escalate to **debug**.

## Monitoring Ownership and Duration

- Assign a single monitoring owner when the loop starts.
- Continue until terminal state and acknowledged next action, a requested stop,
  or a reported unrecoverable error. First loss, eval, or W&B link is not an exit.
- For handoff, transfer ownership explicitly with: current `job_id`, latest
  error/signal, W&B link(s), and resubmission metadata.

## Cadence and Tooling Notes

- After submit/resubmit: sleep `120` once, check for immediate failure; if still
  alive, switch to the normal `570` cadence.
- Keep one long-running monitor session. Resume that session across tool yields;
  repeated no-output waits are expected.
- Run only one active monitor loop per job (duplicate loops cause SSH tunnel and
  port-binding conflicts).
- Sleep in the foreground; keep loop control at the agent level.
- Screen/process alive is not enough. Check state-file freshness plus
  stdout/event-log mtime when a monitor writes them; if no monitor state or
  event update occurs for more than 2 cadences, report `monitor stale`
  separately from `run unhealthy`.
- If an Iris/orchestrator query is blocked or inconclusive, do not assume job
  failure. Cross-check W&B freshness, live logs, checkpoint movement,
  worker/TPU health, and latest monitor state.

## MCP-Assisted Monitoring

When using `marin-mcp-babysitter`, keep the MCP server resident and verify the
job through MCP tools, not only Iris CLI commands.

- Keep the controller tunnel and MCP server in named, restartable sessions and
  record their names, ports, and logs in the state file.
- Start MCP with a stable local controller URL and streamable HTTP transport:
  `uv run --package marin-core marin-mcp-babysitter --controller-url <URL> --cluster <CLUSTER> --transport streamable-http --host 127.0.0.1 --port <PORT>`
- Verify with `iris_job_summary` and `iris_tail_logs`. For heartbeat monitoring,
  report: job state, latest progress/tick/log line, timestamp, error signal.
- If the MCP server is reachable but tool calls fail with connection refused to
  the controller URL, restart only the smoke-test tunnel/session — do not mutate
  the Iris cluster.
- If a sandbox blocks localhost TCP probes, run the probe inside an existing
  long-lived session and write a small JSON result under `scratch/`.
- For a bounded smoke test, start a heartbeat only after submission, MCP
  reachability, and one expected progress line. Delete it at terminal state.

## State File

Write to `scratch/<create_timestamp>_monitoring_state.json` (create `scratch/`
if needed); `<create_timestamp>` has format `YYYYMMDD-HHMM`. Track
`restart_count` to detect flapping. Add MCP fields when a resident MCP server is
part of the setup. The state file allows resume after context reset.

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
   uv run iris --config <CONFIG> job logs --since-seconds 900 <JOB_ID> | rg -i -e "loss|error|traceback|exception|resource_exhausted|oom|compiler_base\.cc:2587|program hbm requirement|largest program allocations|ownerdiederror|dead node|node death|autoscaler unsatisfied resources|no accelerator found|failed_precondition|device or resource busy"

   `iris job logs <JOB_ID>` includes child-job task logs by default.

3. CHECK STATUS
   uv run iris --config <CONFIG> job list --prefix <JOB_ID>

   If `pending_reason` indicates worker scale-up/capacity wait, treat as scheduler
   capacity wait — do not run cluster update/recreate/restart actions. Continue
   waiting on cadence, or stop+resubmit only if user explicitly asks.

   Treat RUNNING as controller-level signal only; confirm allocation via expected
   W&B run when possible.

3a. ON TERMINAL STATE / OOM-LIKE SIGNAL — get a structured per-task summary
   (final state, exit, duration, peak memory) instead of grepping logs:

   uv run iris --config <CONFIG> job describe <JOB_ID>

   Fast postmortem: e.g. "13/14 shards peaked near the container memory limit
   and failed with exit 137" → cgroup OOM, raise `--memory` on resubmit.

4. PRINT W&B RUN IDS/LINKS (once per training run)
   - For normal runs, record the active W&B run id/display name/link when W&B is
     available; many runs use autoassigned ids.
   - When the launch workflow provides an intended W&B identity, validate the
     active run id/display name, state, `_timestamp`, `global_step`, and key
     losses against it. Do not rely only on a stored URL.
   - During resume catch-up, W&B and checkpoint progress may be stale. Live
     training-progress log lines with advancing timestamps are sufficient
     liveness until W&B appears; once W&B is active, require W&B
     timestamps/steps to keep moving.
5. REPORT PROGRESS (format: ~<current>/<exact_max>)
   - Resolve `<exact_max>` from the launched config/code, not from progress-bar display text.
6. EVALUATE (terminal? error? stalled? -> recover or continue)

7. RECOVER (STOP -> RESUBMIT)
   - If current job is still non-terminal, stop it first:
     uv run iris --config <CONFIG> job cancel --exact <JOB_ID>
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

## Error handling

Before recovery, identify the failing file and line. Fix and recover only a
small obvious `NameError`, `ImportError`, `SyntaxError`, or `KeyError`. Report
OOM/HBM, distributed, data-loading, or unclear multi-file failures and stop.

- Treat TPU/XLA HBM reports as failure even without literal OOM:
  - `Program hbm requirement ...`
  - `Largest program allocations in hbm`
- If progress stalls across multiple intervals with `OwnerDiedError`, dead node,
  or unsatisfied resources -> mark `degraded` and notify user.
- If an error repeats after one fix attempt, report it; do not retry blindly.
- Noisy shutdown traces are not decisive by themselves. Terminal Iris/orchestrator
  status, driver/process exit code, final checkpoint state, and W&B state
  determine whether a run succeeded.

## Completion

Before declaring the job complete:

- Verify terminal state is successful.
- Verify W&B is finished or has the expected final state and metrics when W&B is
  part of the run.
- Verify the final checkpoint has `metadata.json` when the run is expected to
  write a checkpoint.
- Capture final metrics, final step, W&B run id/display name, output root, final
  checkpoint path, and caveats in the monitoring state or handoff note.
- Stop/delete monitor heartbeats and resident monitoring sessions that are no
  longer needed.

## Notes

- Iris `job list --prefix` requires canonical job names (`/<user>/<job>`), not short names.
- Iris monitoring is job-level; cluster updates are not part of normal recovery.
- Use `debug` only for a stated TPU bad-node fault or a request to inspect a
  running task; use `babysit-zephyr` for Zephyr.
