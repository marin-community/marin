---
name: babysit-zephyr
description: Start, monitor, and babysit Zephyr pipeline jobs on Iris. Use when launching a zephyr job, watching it run, or restarting after failures.
---

# Skill: Babysit Zephyr Job

Start, monitor, and keep Zephyr pipeline jobs running on Iris. If something needs deeper investigation, escalate to debug-zephyr-job.

## Zephyr Job Structure

A zephyr pipeline job spawns child Iris jobs:
- **`*-coord`** — coordinator (1 task). Orchestrates the pipeline stages, queues tasks, tracks progress.
- **`*-workers`** — worker pool (many tasks). Workers poll the coordinator for shards to execute.

A single job may execute **multiple pipelines sequentially** (e.g. fuzzy dedup runs connected components iteratively, each iteration is a separate pipeline). These show as different `p<N>` values in child job names. This is normal — don't confuse sequential pipelines with failed retries.

Failed retries show as different **hashes** with the same `p0`. Stale coordinators from previous attempts may linger (#3705).

Child job naming: `<hash>-p<pipeline>-a<attempt>-{coord,workers}`.

## Dashboard

```bash
# Connect to the Iris dashboard (establishes SSH tunnel, prints URL with port)
uv run iris --config lib/iris/examples/marin.yaml cluster dashboard
```

## Starting a Job

Get the run command from the user. Typical pattern:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --region <REGION> --no-wait -- python <SCRIPT>
```

The entrypoint container defaults to 1GB memory. For long-running pipelines that accumulate state (GCS clients, logging), increase with `--memory`:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --region <REGION> --memory 5GB --no-wait -- python <SCRIPT>
```

The command prints a job ID on success. Note this ID for monitoring.

## Stopping a Job

Always ask the user before stopping. Stopping kills all child jobs (coordinators, workers).
```bash
uv run iris --config lib/iris/examples/marin.yaml job stop <JOB_ID>
```

## Monitoring

### Health Checks

Check child job states via the Iris CLI (returns per-task state and resourceUsage):
```bash
# diskMb is updated every ~60s. On K8s it is always 0 (workdir lives inside the pod).
uv run iris --config lib/iris/examples/marin.yaml rpc controller list-tasks --job-id <JOB_ID>
```

A healthy zephyr job has:
- Coordinator: RUNNING, 1 task running
- Workers: RUNNING, tasks ramping up toward target count

### Stage Progress

The coordinator logs a progress line every 5s:
```
[stage0-Map → Scatter] 347/1964 complete, 1617 in-flight, 0 queued, 1828/1891 workers alive, 63 dead, counters: bytes_written=4831838208 documents_processed=1200000
```

### User-Defined Counters

Zephyr pipelines can report user-defined counters via `zephyr.counters.increment()`. Counters are aggregated across all workers and appear in:
- **Coordinator progress logs**: appended to the periodic status line (grep for `counters:`)
- **`get_status()` RPC**: `JobStatus.counters` dict, accessible programmatically from the entrypoint process

To emit counters from task code:
```python
from zephyr import counters
counters.increment("documents_processed", batch_size)
```

Counters are sent to the coordinator via the worker heartbeat (every 5s) and only transmitted when values change — no overhead for idle workers.

### Accessing Counters Remotely (for babysitting agents)

Agents running remotely (e.g. via `claude-code-action` or scheduled triggers) access counter state through **coordinator logs**. The coordinator emits a progress line every 5s that includes counter values:

```
[exec-id] [stage0-Map → Scatter] 347/1964 complete, 1617 in-flight, 0 queued, ..., counters: bytes_written=4831838208 documents_processed=1200000
```

**Step 1: Find the coordinator job ID.** The coordinator is a child job of the parent Zephyr job, named `*-coord`:
```bash
uv run iris --config lib/iris/examples/marin.yaml rpc controller list-tasks \
  --job-id <PARENT_JOB_ID> --json | python3 -c "
import json, sys
for t in json.load(sys.stdin):
    print(t['taskId'])
" | grep coord
```

**Step 2: Fetch coordinator logs and extract counter lines:**
```bash
uv run iris --config lib/iris/examples/marin.yaml rpc controller get-task-logs \
  --id <COORD_JOB_ID> --max-total-lines 5000 --attempt-id -1 --tail
```

**Step 3: Parse counters from log lines.** Grep for `counters:` to find the latest values:
```python
import re

# From the fetched log entries
for entry in task_logs:
    msg = entry.get("data", "")
    match = re.search(r"counters: (.+)$", msg)
    if match:
        pairs = match.group(1).split()
        counters = {k: int(v) for k, v in (p.split("=") for p in pairs)}
        print(counters)
```

The last `counters:` line in the logs gives the most recent aggregate values across all workers.

**Caveat**: With large worker pools, `pull_task` operations flood the log buffer (#3707). Filter when parsing:
```python
for entry in task_logs:
    msg = entry.get('data', '')
    if 'pull_task' in msg or 'Started operation' in msg or 'report_result' in msg or 'registered' in msg or 'tasks completed' in msg:
        continue
    print(msg)
```

**Note**: `get_status()` on the coordinator actor returns `JobStatus.counters` as a dict, but this is only callable from within the entrypoint process (same fray cluster). Remote agents should use the log-parsing approach above.

### Coordinator Thread Dump

When logs are flooded, a thread dump tells you if the coordinator is alive and working:
```bash
uv run iris --config lib/iris/examples/marin.yaml rpc controller profile-task \
  --json '{"target":"<COORD_JOB_ID>/0","durationSeconds":1,"profileType":{"threads":{}}}'
```

Key patterns:
- `actor-method_0` in `_wait_for_stage` → pipeline active, waiting for current stage to complete
- `_coordinator_loop` thread present → heartbeat/dispatch loop running
- All threads in `_worker` (thread pool idle) → pipeline exited, coordinator is a zombie

## Monitoring Lifecycle

After submitting, monitor in escalating stages:

1. **Smoke check (first 2-5 minutes)**: Confirm coordinator and workers child jobs appear and reach RUNNING state. Check coordinator logs for early errors. If it fails here — likely a code bug, config issue, or bundle fetch timeout.

2. **Steady-state monitoring**: Check stage progress via coordinator logs. Confirm two things: (a) shards are completing within the current stage, and (b) stages are advancing. Calibrate check-in interval to the pipeline — you want to see at least one stage transition between checks. For pipelines with many short stages, check every few minutes. For pipelines with few long stages, every 15-30 minutes may suffice.

3. **Failure detection**: If workers get KILLED or the coordinator goes zombie, the `StepRunner` may retry automatically (new child jobs with a different hash appear). Check the latest attempt. Stale coordinators from previous attempts may accumulate (#3705). If retries keep failing, escalate to debug-zephyr-job.

**"Terminated by user" is misleading**: This diagnostic does not necessarily mean a human killed the job. The system uses this message for various internal termination reasons. Always check the actual logs at each level (parent job, coordinator, workers) to determine the real cause.

## Restarting After Failure

1. Ask the user if it's okay to stop and restart.
2. Stop the job.
3. Get the run command (or reuse the previous one).
4. Submit and resume monitoring.

## Monitoring Counters

When babysitting a Zephyr job, check coordinator logs for counter lines. Counters give you insight into pipeline throughput (e.g. `documents_processed`, `bytes_written`, `validation_errors`). If counters stop advancing while shards are still in-flight, this may indicate a straggler or stuck worker — escalate to debug-zephyr-job.

## When to Escalate

Escalate to **debug-zephyr-job** when:
- A stage is stuck (no shard progress for an extended period)
- Stragglers are holding up a stage (few in-flight, 0 queued, most workers idle)
- Workers are failing repeatedly with the same error
- Counters stop advancing while tasks remain in-flight
- For controller issues (e.g., RPCs timing out), use the **debug-iris-controller** skill
