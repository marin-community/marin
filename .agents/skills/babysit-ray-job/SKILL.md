---
name: babysit-ray-job
description: Monitor a Ray job continuously and recover on failure. Use when asked to babysit, monitor, or watch a Ray training job.
---

# Skill: Babysit Ray Job

Monitor a Ray job continuously and recover on failure. Follows the shared
practices in **babysit-job** — read that first for monitoring ownership, cadence,
fixing small bugs, and escalation procedures.

## Required Info

1. `job_id` — Ray job ID (e.g., `ray-run-held-isoflop_sweep-20260131-051716`)
2. `cluster` — cluster name/alias for `scripts/ray/cluster.py` (e.g., `us-east5-a`, `us-central2`)
3. `experiment` — script path used for resubmission (e.g., `experiments/isoflop_sweep.py`)

If any required field is missing, ask for it before proceeding.

## State File

```json
{
  "ts": <timestamp_ms>,
  "track": "ray",
  "job_id": "<JOB_ID>",
  "cluster": "<CLUSTER>",
  "experiment": "<EXPERIMENT_PATH>",
  "restart_count": 0
}
```

## Loop

```
1. SLEEP
   - if just submitted/restarted: sleep 120 once
   - otherwise: sleep 570

2. CHECK LOGS
   uv run scripts/ray/cluster.py --cluster <CLUSTER> job-logs -n 200 -g "loss\|error\|Error\|Traceback\|RESOURCE_EXHAUSTED\|compiler_base.cc:2587\|Program hbm requirement\|Largest program allocations" <JOB_ID>

3. CHECK STATUS
   - terminal success: SUCCEEDED
   - terminal non-success: FAILED, STOPPED
   - non-terminal: PENDING, RUNNING
   - Treat RUNNING as controller-level signal only; confirm allocation via
     expected W&B run when possible.

4-6. Follow generic PRINT W&B / REPORT PROGRESS / EVALUATE from babysit-job.

7. RECOVER (STOP -> RESUBMIT)
   - If current job is still non-terminal, stop it first:
     uv run scripts/ray/cluster.py --cluster <CLUSTER> stop-job <JOB_ID>
   - Then resubmit:
     uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster <CLUSTER> -- python <EXPERIMENT_PATH>
   - Capture `job_id` from submit output.
   - Update state file: `job_id=<NEW_JOB_ID>`, `restart_count += 1`.
   - Go to step 1.
```

## Notes

- Ray states: `PENDING`, `RUNNING`, `STOPPED`, `SUCCEEDED`, `FAILED`
- Ray `STOPPED` is equivalent to Iris `JOB_STATE_KILLED`.
