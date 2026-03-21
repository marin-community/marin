---
name: babysit-iris-job
description: Monitor an Iris job continuously and recover on failure. Use when asked to babysit, monitor, or watch an Iris training job (not Zephyr pipelines — use babysit-zephyr for those).
---

# Skill: Babysit Iris Job

Monitor an Iris job continuously and recover on failure. Recovery is stop then resubmit.

For Zephyr pipeline jobs, use **babysit-zephyr** instead.
For TPU bad-node errors, escalate to **debug-tpu**.
For Ray jobs, use **babysit-job** with `track=ray`.

Cluster-level actions are out of scope. Do not restart, recreate, or otherwise mutate the cluster unless the user gives explicit consent in the current thread.

## Before Starting

### Required Info

1. `job_id` — Iris job ID in canonical format `/<user>/<job>` (e.g., `/dlwh/iris-run-train_tiny_model_tpu-20260302-185630`)
2. `config` — Iris config path (e.g., `lib/iris/examples/marin.yaml`)
3. `resubmit_command` — exact Iris submit command for resubmission; must include `--no-wait`
4. For Marin TPU training jobs, use `--extra marin:tpu` (not `--extra marin:cpu`)
5. For TPU jobs, the resubmit command must request TPU resources with `--tpu <variant>`.
   `--reserve <variant>` only holds capacity; it does not attach TPU devices to the task container.

Example Iris resubmit command:
`uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --extra marin:tpu --tpu v5litepod-16 -- python experiments/tutorials/train_tiny_model_tpu.py`

If any required field is missing, ask for it before proceeding.

## Monitoring Ownership and Duration

- Assign a single monitoring owner when the loop starts.
- Keep this loop running until one of the following:
  - the job reaches a terminal state and the user has acknowledged next action
  - a user-specified stopping point is reached
  - an unrecoverable error is found and reported to the user
- Do not stop early after seeing first loss lines, first eval, or first W&B link.
- Expect monitoring to commonly take 4-5 hours for ferry-scale runs.
- For GPT Codex specifically: unless otherwise directed, do not end your turn just to give a status update; keep monitoring until terminal state or until the user's goal is reached.
- If the user requests continuous monitoring, do not end the turn while monitoring is active; continue until terminal state, a user-specified stopping point, or an unrecoverable error.
- If handoff is needed, transfer ownership explicitly with: track, current `job_id`, latest error/signal, W&B link(s), and resubmission metadata.

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
  - run only one active monitor loop per job to avoid duplicate SSH tunnel and port-binding conflicts

## State File

Write to `scratch/<create_timestamp>_monitoring_state.json`, create the `scratch` directory if needed. `<create_timestamp>` should have format `YYYYMMDD-HHMM`.

```json
{
  "ts": <timestamp_ms>,
  "track": "iris",
  "job_id": "<JOB_ID>",
  "config": "<IRIS_CONFIG_PATH>",
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

4. PRINT W&B RUN IDS/LINKS
   - Inspect logs for W&B URL/ID (`wandb: View run`, `/runs/<RUN_ID>`).
   - Print run id(s) and full link(s) once per training run.
   - If resubmission creates a new run, print new id/link once.

5. REPORT PROGRESS
   - Report as `~<current>/<exact_max>`.
   - Example: `~4600/5155`.
   - Never present rounded max as exact.
   - If max unknown, report `~<current>/unknown`.

6. EVALUATE
   - If user stop point reached -> summarize and exit.
   - If terminal success -> report completion and exit.
   - If terminal non-success -> go to step 7 unless user says otherwise.
   - If output contains `error`/`Error`/`Traceback` -> go to step 7.
   - Treat TPU/XLA HBM reports as failure even without literal OOM:
     - `Program hbm requirement ...`
     - `Largest program allocations in hbm`
   - If progress stalls across multiple intervals with `OwnerDiedError`, dead node,
     or unsatisfied resources -> mark `degraded` and notify user.
   - If loss/progress lines are present and status non-terminal -> go to step 1.
   - If no output and status non-terminal -> go to step 1.

7. RECOVER (STOP -> RESUBMIT)
   - Recover only the job (never mutate cluster without explicit consent).
   - For TPU bad-node errors, invoke **debug-tpu** skill first, then continue here.
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

## Task Exec

Use `iris task exec` to run commands inside a running task's container.

### Basic usage

```bash
# Run a one-shot command
iris task exec <TASK_ID> -- ls /app

# Run with a custom timeout (default: 60s)
iris task exec <TASK_ID> --timeout 300 -- python -c "import torch; print(torch.cuda.device_count())"

# Run with no timeout
iris task exec <TASK_ID> --timeout -1 -- tail -f /var/log/app.log
```

### Running a background command that survives disconnect

The exec session is non-interactive and buffers output. To kick off a long-running
command that continues after the exec returns, use `nohup` + `&` inside a bash wrapper:

```bash
# Start a background process that writes to a file
iris task exec <TASK_ID> -- bash -c "nohup bash -c 'while true; do date >> /tmp/heartbeat.txt; sleep 10; done' > /dev/null 2>&1 &"

# Check on it later
iris task exec <TASK_ID> -- cat /tmp/heartbeat.txt

# Check if the process is still running
iris task exec <TASK_ID> -- pgrep -f heartbeat
```

### Task ID format

Task IDs follow the pattern `/<user>/<job>/<task_index>`, e.g., `/rav/my-job/0`.

## Fixing Small Bugs

When EVALUATE detects an error, before recovery:

1. Analyze logs:
   - Look for `Traceback`, `Error`, `Exception`.
   - Identify file and line number.
2. If it is a small fix (typo, missing import, wrong variable name):
   - Read the file.
   - Make the fix.
   - Proceed to RECOVER.
3. If it is complex (architectural, unclear cause, broad investigation):
   - Do not auto-fix.
   - Report to user and exit loop.

Small-fix examples:
- `NameError: name 'foo' is not defined`
- `ImportError: cannot import 'bar'`
- `SyntaxError` from missing comma/bracket/colon
- obvious wrong `KeyError` key

Complex examples:
- OOM errors
- TPU/XLA HBM exhaustion signatures (`Program hbm requirement`, `Largest program allocations in hbm`)
- distributed training failures
- data loading issues
- unclear stack traces spanning multiple files

## Notes

- Sleep must be foreground (max ~10 min due to tool timeout).
- Loop control is at agent level, not bash.
- Track `restart_count` to detect flapping.
- State file allows resume after context reset.
- Iris states: `JOB_STATE_PENDING`, `JOB_STATE_BUILDING`, `JOB_STATE_RUNNING`, `JOB_STATE_SUCCEEDED`, `JOB_STATE_FAILED`, `JOB_STATE_KILLED`, `JOB_STATE_WORKER_FAILED`, `JOB_STATE_UNSCHEDULABLE`
- Iris `job list --prefix` requires canonical job names (`/<user>/<job>`), not short names.
- Iris monitoring is job-level; cluster updates are not part of normal recovery.
- Loop duration can be hours to days.
- If same error repeats after one fix attempt, do not retry blindly; report to user.

## When to Escalate

- Zephyr pipeline issues -> **debug-zephyr**
- Controller issues (RPCs timing out, stuck scheduling) -> **debug-iris-controller**
- TPU bad-node errors -> **debug-tpu**
