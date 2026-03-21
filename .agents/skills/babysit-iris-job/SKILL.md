---
name: babysit-iris-job
description: Monitor an Iris job continuously and recover on failure. Use when asked to babysit, monitor, or watch a job, pipeline, workflow, or training run.
---

# Skill: Babysit Iris Job

Monitor an Iris job continuously and recover on failure. Follows the shared
practices in **babysit-job** — read that first for monitoring ownership, cadence,
fixing small bugs, and escalation procedures.

## Required Info

1. `job_id` — Iris job ID in canonical format `/<user>/<job>` (e.g., `/dlwh/iris-run-train_tiny_model_tpu-20260302-185630`)
2. `config` — Iris config path (e.g., `lib/iris/examples/marin.yaml`)
3. `resubmit_command` — exact Iris submit command for resubmission; must include `--no-wait`
4. For Marin TPU training jobs, use `--extra marin:tpu` (not `--extra marin:cpu`)
5. For TPU jobs, the resubmit command must request TPU resources with `--tpu <variant>`.
   `--reserve <variant>` only holds capacity; it does not attach TPU devices to the task container.

Example resubmit command:
`uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --extra marin:tpu --tpu v5litepod-16 -- python experiments/tutorials/train_tiny_model_tpu.py`

If any required field is missing, ask for it before proceeding.

## State File

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

4-6. Follow generic PRINT W&B / REPORT PROGRESS / EVALUATE from babysit-job.

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

## Task Exec

Use `iris task exec` to run commands inside a running task's container ONLY if you must, prefer existing
logs, metrics, CLI commands and RPC calls for monitoring and diagnosis.

### Basic usage

```bash
# Run a one-shot command
iris task exec <TASK_ID> -- ls /app

# Run with a custom timeout (default: 60s)
iris task exec <TASK_ID> --timeout 300 -- python -c "import torch; print(torch.cuda.device_count())"

# Run with no timeout - last resort for long-running commands
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

## Notes

- Iris `job list --prefix` requires canonical job names (`/<user>/<job>`), not short names.
- Iris monitoring is job-level; cluster updates are not part of normal recovery.
