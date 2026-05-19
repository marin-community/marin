---
name: babysit-zephyr
description: Launch and babysit Zephyr pipeline jobs on Iris.
---

# Skill: Babysit Zephyr Job

Start, monitor, and keep Zephyr pipeline jobs running on Iris. Escalate deeper investigations to **debug**.

## Zephyr Job Structure

A zephyr pipeline job spawns child Iris jobs:
- **`*-coord`** — coordinator (1 task). Orchestrates pipeline stages, queues tasks, tracks progress.
- **`*-workers`** — worker pool (many tasks). Workers poll the coordinator for shards.

A single job may execute **multiple pipelines sequentially** (e.g. fuzzy dedup runs connected components iteratively, each iteration a separate pipeline). These show as different `p<N>` values in child job names — normal, not failed retries.

Failed retries show as different **hashes** with the same `p0`. Stale coordinators from previous attempts may linger (#3705).

Child job naming: `<hash>-p<pipeline>-a<attempt>-{coord,workers}`.

## Iris Config

All Iris commands below use `--config <CONFIG>`. Resolve the cluster name the user gives to the matching file under `lib/iris/config/` (see **babysit-job** for the full mapping). Examples use `marin.yaml` — substitute the actual config (e.g., `marin-dev.yaml`) as needed.

## Dashboard

```bash
# Connect to the Iris dashboard (establishes SSH tunnel, prints URL with port)
uv run iris --config lib/iris/config/marin.yaml cluster dashboard
```

## Starting a Job

Get the run command from the user. Typical pattern:
```bash
uv run iris --config lib/iris/config/marin.yaml job run --region <REGION> --no-wait -- python <SCRIPT>
```

The entrypoint container defaults to 1GB memory. For long-running pipelines that accumulate state (GCS clients, logging), increase with `--memory`:
```bash
uv run iris --config lib/iris/config/marin.yaml job run --region <REGION> --memory 5GB --no-wait -- python <SCRIPT>
```

The command prints a job ID on success. Note it for monitoring.

## Stopping a Job

Always ask the user before stopping. Stopping kills all child jobs (coordinators, workers).
```bash
uv run iris --config lib/iris/config/marin.yaml job stop <JOB_ID>
```

## Monitoring

### Health Checks

Check child job states via the Iris CLI (returns per-task state and resourceUsage):
```bash
# diskMb is updated every ~60s. On K8s it is always 0 (workdir lives inside the pod).
uv run iris --config lib/iris/config/marin.yaml rpc controller list-tasks --job-id <JOB_ID>
```

A healthy zephyr job has:
- Coordinator: RUNNING, 1 task running
- Workers: RUNNING, tasks ramping up toward target count

### Stage Progress

The coordinator logs a progress line every 5s:
```
[stage0-Map → Scatter] 347/1964 complete, 1617 in-flight, 0 queued, 1828/1891 workers alive, 63 dead
```

Fetch via the Iris CLI:
```bash
uv run iris --config lib/iris/config/marin.yaml rpc controller get-task-logs \
  --id <COORD_JOB_ID> --max-total-lines 5000 --attempt-id -1 --tail
```

**Caveat**: With large worker pools, `pull_task` operations flood the log buffer (#3707). Filter when parsing:
```python
for entry in task_logs:
    msg = entry.get('data', '')
    if 'pull_task' in msg or 'Started operation' in msg or 'report_result' in msg or 'registered' in msg or 'tasks completed' in msg:
        continue
    print(msg)
```

### Coordinator Thread Dump

When logs are flooded, a thread dump tells you if the coordinator is alive and working:
```bash
uv run iris --config lib/iris/config/marin.yaml rpc controller profile-task \
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
