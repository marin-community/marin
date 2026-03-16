---
name: zephyr-debugger
description: Debug Zephyr pipeline execution issues (stuck stages, stragglers, idle workers). Use when zephyr workers are misbehaving or pipelines are slow.
---

# Skill: Zephyr Pipeline Debugger

## Dashboard Setup

```bash
# Connect to the Iris dashboard (establishes SSH tunnel, prints URL with port)
uv run iris --config lib/iris/examples/marin.yaml cluster dashboard

# Local dashboard proxy — serves a locally-built dashboard against the remote controller.
# Default port 8080 (configurable via args). Restart after each dashboard code change.
uv run iris --config lib/iris/examples/marin.yaml cluster dashboard-proxy
```

You can modify the dashboard frontend locally and serve via `dashboard-proxy` to build custom views on top of the existing API (e.g. aggregate worker memory into a histogram, render stage progress charts). The backend is remote and cannot be changed — only the frontend. If the backend API doesn't expose data you need, file an issue. If you build a useful frontend component during debugging, also file an issue to upstream it. You need to restart the `dashboard-proxy` after making frontend changes.

## Architecture

Pull-based coordinator/worker model. Coordinator queues tasks per stage; workers poll `pull_task()`, execute shards, report results. Stages are sequential barriers — all shards in a stage must complete before the next starts (`_wait_for_stage`).

Key files:
- `lib/zephyr/src/zephyr/execution.py` — coordinator loop, worker poll loop, shard execution
- `lib/zephyr/src/zephyr/plan.py` — pipeline plan, scatter/reduce, k-way merge

Child job naming: `<hash>-p<pipeline>-a<attempt>-{coord,workers}`. Focus on the latest attempt.

## Observability

### Dashboard API (ConnectRPC, JSON)

The Iris dashboard at `http://127.0.0.1:<DASHBOARD_PORT>` proxies to the controller. The proxy target is shown in the dashboard header (`Proxy → http://127.0.0.1:<port>`).

```bash
# Task logs (coordinator or workers)
# Response includes taskLogs with timestamp, source, data, level; plus truncated flag and cursor
uv run iris --config lib/iris/examples/marin.yaml rpc controller get-task-logs \
  --id <JOB_ID> --max-total-lines 5000 --attempt-id -1 --tail

# List tasks (memory, CPU, disk, state per worker)
# Each task's resourceUsage includes: memoryMb, diskMb, cpuPercent, memoryPeakMb, processCount
# diskMb is updated every ~60s. On K8s it is always 0 (workdir lives inside the pod).
uv run iris --config lib/iris/examples/marin.yaml rpc controller list-tasks --job-id <JOB_ID>
```

### Iris CLI

```bash
# Job logs (uses batch log fetching via SSH tunnel)
uv run iris --config lib/iris/examples/marin.yaml job logs --level info --since-seconds 300 <JOB_ID>

# Or with direct controller URL (check dashboard Proxy header for port)
uv run iris --controller-url http://127.0.0.1:10001 job logs --level info <JOB_ID>
```

### On-Demand Profiling (py-spy / memray)

The dashboard CPU, MEM, and THR buttons trigger profiling via the `ProfileTask` RPC (controller → worker → task container). All profiling is on-demand — nothing is collected automatically.

| Button | Tool | Output | Duration |
|---|---|---|---|
| **THR** | `py-spy dump` | Thread dump (text) | Instant |
| **CPU** | `py-spy record` | Speedscope JSON flamegraph | 10s default |
| **MEM** | `memray attach` | Flamegraph HTML | 10s default |

The CLI supports profiling **worker processes** (not task containers):
```bash
uv run iris process profile threads --worker <WORKER_ID>
uv run iris process profile cpu --duration 10 --output /tmp/profile.speedscope.json --worker <WORKER_ID>
uv run iris process profile mem --duration 10 --output /tmp/profile.html --worker <WORKER_ID>
```

**Task-level profiling** (inside the task container) is not supported by the CLI scalar flags yet. Use the dashboard buttons or call the `ProfileTask` RPC via `--json`:

```bash
# CPU profile a specific task (10s, speedscope JSON)
uv run iris --config lib/iris/examples/marin.yaml rpc controller profile-task \
  --json '{"target":"<TASK_ID>","durationSeconds":10,"profileType":{"cpu":{"format":"SPEEDSCOPE"}}}'

# Target format: /user/job-name/child-job/task-index
# Profile types: {"threads":{}}  {"cpu":{"format":"SPEEDSCOPE"}}  {"memory":{"format":"FLAMEGRAPH"}}
```

### Poor Man's Profiling

Take 5-10 THR samples from the same worker with ~2s intervals. The `zephyr-poll-*` thread stack shows where time is spent. Look for `active+gil` vs `idle`:

| Thread state | Location | Meaning |
|---|---|---|
| `active+gil` in `_execute_shard` | `_reduce_gen`, `_scatter_items`, user fn | Doing work |
| `idle` at `_poll_loop:1163` | — | Waiting for tasks |
| `idle` at `write_table` (pyarrow) | — | I/O bound |

Coordinator: `actor-method_0` in `_wait_for_stage` means it's blocked waiting for the current stage to finish.

## Diagnostic Patterns

### Stage Progress

The coordinator logs a progress line every 5s during active execution:
```
[stage1-Reduce → Scatter] 1869/1964 complete, 95 in-flight, 0 queued, 1964/1964 workers alive, 0 dead
```

**Caveat**: idle workers flood coordinator logs with `pull_task` operations (~thousands/sec with large worker pools). Filter when querying:
```python
for entry in task_logs:
    msg = entry.get('data', '')
    if 'pull_task' in msg or 'Started operation' in msg:
        continue
    print(msg)
```

### Straggler Detection

1. **Progress line**: `in-flight >> 0` with `queued == 0` means stragglers — no new work to assign, waiting on slow shards.
2. **Memory/disk distribution**: Query `ListTasks` and bucket by `memoryMb` and `diskMb`. Idle workers: <200 MB memory. Finished workers: 1-5 GB. Stragglers: >5 GB memory (or high CPU/disk).
3. **THR on high-memory workers**: Confirm they're in `_execute_shard` with `active+gil`. The user function in the stack identifies the bottleneck.

### Data Skew

Symptom: most shards complete fast, a few take orders of magnitude longer.

Diagnosis: THR on the straggler worker shows the user-level reduce function holding the GIL. Compare memory across workers — skewed keys produce 10-100x memory outliers.

### Worker Failures / Reassignment

Workers that failed and got reassigned show in the task table with an error like `Worker ... failed: Request timed out`. The replacement worker starts fresh (low memory) and must re-pull a task — if no tasks remain queued, it idles.

### Misleading Diagnostics

**"Terminated by user"** does not necessarily mean a human killed the job. The system uses this message for various internal termination reasons. Always check the actual logs at each level (parent job, coordinator, workers) to determine the real cause.

## Requesting New Tools

If debugging reveals a need for capabilities not exposed by the existing API or CLI — e.g. you find yourself wanting to SSH into a worker to inspect something — do not work around it. File an issue requesting the capability be exposed as a proper RPC endpoint or CLI command. The platform should provide the tools; agents and users should not need to SSH into machines to debug.

## After a Fix

When debugging leads to a code fix, use the **zephyr-babysit** skill to stop the current job, resubmit, and monitor the new run. Then spot-check the fix:

- THR-sample active workers to confirm the hot function is no longer dominating the stack
- Check memory distribution — are stragglers still present? How do they compare to the old run?
- If new anomalies appear, continue debugging with the techniques above
