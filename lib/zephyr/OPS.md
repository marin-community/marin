# Zephyr Operations

## Dashboard

See `lib/iris/OPS.md` → "Cluster Lifecycle" for `iris cluster dashboard` and `dashboard-proxy`. The proxy serves a locally-built frontend against the remote controller — restart it after frontend changes.

## Architecture

Pull-based coordinator/worker model. Coordinator queues tasks per stage; workers poll `pull_task()`, execute shards, report results. Stages are sequential barriers — all shards in a stage must complete before the next starts (`_wait_for_stage`).

Key files:
- `src/zephyr/execution.py` — coordinator loop, worker poll loop, shard execution
- `src/zephyr/plan.py` — pipeline plan, scatter/reduce, k-way merge

Child job naming: `<hash>-p<pipeline>-a<attempt>-{coord,workers}`. Focus on the latest attempt.

## Observability

### Iris CLI

See `lib/iris/OPS.md` → "Job Management" and "Process Inspection & Profiling" for general log/profiling commands. Zephyr-specific notes:

```bash
# Zephyr task logs (coordinator or workers) — use --attempt-id -1 for latest attempt
uv run iris rpc controller get-task-logs --id <JOB_ID> --max-total-lines 5000 --attempt-id -1 --tail

# List tasks with resource usage (memoryMb, diskMb, cpuPercent, memoryPeakMb, processCount)
# diskMb updated ~60s. On K8s it is always 0 (workdir lives inside the pod).
uv run iris rpc controller list-tasks --job-id <JOB_ID>
```

### On-Demand Profiling

Dashboard buttons (THR/CPU/MEM) trigger profiling via the `ProfileTask` RPC — nothing is collected automatically.

Task-level profiling via RPC (when CLI flags don't cover it):
```bash
uv run iris rpc controller profile-task \
  --json '{"target":"<TASK_ID>","durationSeconds":10,"profileType":{"cpu":{"format":"SPEEDSCOPE"}}}'
# Profile types: {"threads":{}}  {"cpu":{"format":"SPEEDSCOPE"}}  {"memory":{"format":"FLAMEGRAPH"}}
```

### Actor RPC (Coordinator Queries)

`iris actor call` calls methods on the zephyr coordinator through the controller proxy. Find the coordinator endpoint from the coord child job's logs:

```bash
# The coord child job is named <JOB_ID>/<hash>-p<N>-a<N>-coord (see "Child job naming" above)
uv run iris --config <CONFIG> job logs <COORD_JOB_ID> | grep "host_actor.*coord"
```

The full path before `->` is the ENDPOINT argument.

```bash
# Global counters (accumulated across all stages)
uv run iris --config <CONFIG> actor call <ENDPOINT> get_counters

# Per-worker counters (in-flight heartbeat snapshot)
uv run iris --config <CONFIG> actor call <ENDPOINT> get_counters '{"worker_id": "<WORKER_ID>"}'

# Pipeline status (stage, completed/total shards, worker states)
uv run iris --config <CONFIG> actor call <ENDPOINT> get_status
```

Worker IDs follow `zephyr-<hash>-p<N>-workers-<INDEX>`. Compare per-worker counters to spot stragglers.

## Diagnostic Patterns

### Stage Progress

The coordinator logs a progress line every 5s:
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
2. **Memory/disk distribution**: Query `ListTasks` and bucket by `memoryMb` and `diskMb`. Idle workers: <200 MB. Finished: 1-5 GB. Stragglers: >5 GB (or high CPU/disk).
3. **THR on high-memory workers**: Confirm they're in `_execute_shard` with `active+gil`. The user function in the stack identifies the bottleneck.

### Data Skew

Symptom: most shards complete fast, a few take orders of magnitude longer.

Diagnosis: THR on the straggler worker shows the user-level reduce function holding the GIL. Compare memory across workers — skewed keys produce 10-100x memory outliers.

### Worker Failures / Reassignment

Workers that failed and got reassigned show in the task table with `Worker ... failed: Request timed out`. The replacement worker starts fresh (low memory) and must re-pull a task — if no tasks remain queued, it idles.

### Misleading Diagnostics

**"Terminated by user"** does not necessarily mean a human killed the job. The system uses this message for various internal termination reasons. Always check the actual logs at each level (parent job, coordinator, workers) to determine the real cause.

### Poor Man's Profiling

Take 5-10 THR samples from the same worker with ~2s intervals. The `zephyr-poll-*` thread stack shows where time is spent:

| Thread state | Location | Meaning |
|---|---|---|
| `active+gil` in `_execute_shard` | `_reduce_gen`, `_scatter_items`, user fn | Doing work |
| `idle` at `_poll_loop:1163` | — | Waiting for tasks |
| `idle` at `write_table` (pyarrow) | — | I/O bound |

Coordinator: `actor-method_0` in `_wait_for_stage` means it's blocked waiting for the current stage to finish.

## Requesting New Tools

If debugging reveals a need for capabilities not exposed by the existing API or CLI — e.g. you find yourself wanting to SSH into a worker — do not work around it. File an issue requesting the capability as a proper RPC endpoint or CLI command.
