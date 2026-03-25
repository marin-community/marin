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

## Notes

For debugging running tasks with `iris task exec`, see **debug-iris-job**.

- Iris `job list --prefix` requires canonical job names (`/<user>/<job>`), not short names.
- Iris monitoring is job-level; cluster updates are not part of normal recovery.
