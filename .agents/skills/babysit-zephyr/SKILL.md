---
name: babysit-zephyr
description: Launch or continuously monitor a specified Zephyr pipeline on Iris only when asked to babysit it; restart only after explicit approval.
---

# Babysit a Zephyr job

## Zephyr Job Structure

A job has a one-task `*-coord` coordinator and a `*-workers` pool. Sequential
pipelines produce different `p<N>` child names; retries produce different hashes
with the same `p0`. Child names follow
`<hash>-p<pipeline>-a<attempt>-{coord,workers}`. Old coordinators may linger.

## Iris Config

Resolve the requested cluster to its file under `lib/iris/config/`; substitute
that file for `<CONFIG>` below. See `lib/zephyr/OPS.md` for the dashboard and
coordinator query reference.

## Starting a Job

Get the run command from the user:
```bash
uv run iris --config <CONFIG> job run --region <REGION> --no-wait -- python <SCRIPT>
```

The entrypoint container defaults to 1GB memory. For long-running pipelines that accumulate state (GCS clients, logging), increase with `--memory`:
```bash
uv run iris --config <CONFIG> job run --region <REGION> --memory 5GB --no-wait -- python <SCRIPT>
```

The command prints a job ID on success. Note it for monitoring.

## Stopping a Job

Always ask the user before stopping. Stopping kills all child jobs (coordinators, workers).
```bash
uv run iris --config <CONFIG> job cancel <JOB_ID>
```

## Monitoring

### Health Checks

Check child job states via the Iris CLI (returns per-task state and resourceUsage):
```bash
# diskMb is updated every ~60s. On K8s it is always 0 (workdir lives inside the pod).
uv run iris --config <CONFIG> rpc controller list-tasks --job-id <JOB_ID>
```

A healthy zephyr job has:
- Coordinator: RUNNING, 1 task running
- Workers: RUNNING, tasks ramping up toward target count

### Stage Progress

The coordinator logs stage, completed, in-flight, queued, and worker counts:
```bash
uv run iris --config <CONFIG> rpc controller get-task-logs \
  --id <COORD_JOB_ID> --max-total-lines 5000 --attempt-id -1 --tail
```

Large pools can flood the log with `pull_task`, `Started operation`,
`report_result`, `registered`, and `tasks completed`; filter those entries.

### Coordinator Thread Dump

When logs are flooded, a thread dump tells you if the coordinator is alive and working:
```bash
uv run iris --config <CONFIG> rpc controller profile-task \
  --json '{"target":"<COORD_JOB_ID>/0","durationSeconds":1,"profileType":{"threads":{}}}'
```

Key patterns:
- `actor-method_0` in `_wait_for_stage` → pipeline active, waiting for current stage to complete
- `_coordinator_loop` thread present → heartbeat/dispatch loop running
- All threads in `_worker` (thread pool idle) → pipeline exited, coordinator is a zombie

## Monitoring Lifecycle

After submitting, monitor in escalating stages:

1. **Smoke check (first 2-5 minutes)**: Confirm coordinator and workers child jobs appear and reach RUNNING. Check coordinator logs for early errors. Failure here is likely a code bug, config issue, or bundle fetch timeout.

2. **Steady-state monitoring**: Check stage progress via coordinator logs. Confirm (a) shards complete within the current stage, and (b) stages advance. Calibrate check-in interval so you see at least one stage transition between checks — every few minutes for many short stages, every 15-30 minutes for few long stages.

3. **Failure detection**: If workers get KILLED or the coordinator goes zombie, the `StepRunner` may retry automatically (new child jobs with a different hash). Check the latest attempt. Stale coordinators from previous attempts may accumulate (#3705). If retries keep failing, escalate to **debug**.

**"Terminated by user" is misleading**: This does not necessarily mean a human killed the job. The system uses this message for various internal termination reasons. Always check the actual logs at each level (parent job, coordinator, workers) to find the real cause.

## Restarting After Failure

1. Ask the user if it's okay to stop and restart.
2. Stop the job.
3. Get the run command (or reuse the previous one).
4. Submit and resume monitoring.

## When to Escalate

Escalate to **debug** when:
- A stage is stuck (no shard progress for an extended period)
- Stragglers are holding up a stage (few in-flight, 0 queued, most workers idle)
- Workers are failing repeatedly with the same error
- Controller issues (e.g., RPCs timing out)
