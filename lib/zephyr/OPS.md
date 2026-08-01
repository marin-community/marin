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

### Repeated Worker Pools

Several coordinator and worker-group pairs with different pool IDs are
independent `ZephyrContext` instances, not retries. Concurrent pipelines that
use plain contexts repeat worker startup and can leave capacity idle.

List the complete topology beneath the root job:

```bash
uv run iris --config <CONFIG> job list --prefix <ROOT_JOB_ID> --limit 500
```

Entering one context starts a retained pool. Pass that entered context to the
functions that should share it; a serialized child receives a borrowed
coordinator handle. The expected topology is then one coordinator actor group
and one worker actor group for all submitted pipelines. Additional actor jobs,
such as a memory store, are expected and are not worker pools.

If a pipeline unexpectedly creates another pool, confirm that the entered
context is still alive and was passed to that stage. Do not add a federation
target to repair this: child jobs already run on their federated parent's peer.

One shared coordinator retains state for every active pipeline. Size its task
for their aggregate plans, queues, counters, and RPCs.

### External-Sort Memory Budgets

Worker container RAM and active task RAM are different limits. Shuffle fan-in
and external-sort buffers use the active map or reduce task budget when one is
declared; the worker cgroup limit is only the fallback. Leave headroom between
the task budget and container request for the worker process and runtime.

The worker log records the selected memory budget and external-sort fan-in. If
a reducer OOMs despite a small declared task budget, verify those log values
before increasing the worker request. A large cgroup-derived fan-in means the
task budget was not propagated to the shard.

Use completed `zephyr.stage` finelog rows for the result: compare
`mem_peak_bytes_max`, `mem_bytes_avg`, and `cpu_time_total` between equivalent
runs. Requested RAM shows capacity, not actual use. Inspect the live cgroup of
a known skew worker only as supporting evidence because a point sample can
miss the true peak.

### CoreWeave Parquet HEAD Returns HTTP 400

If every Parquet-shuffle reducer fails with a URL shaped like this, the client
is using path-style S3 addressing:

```text
Generic S3 HEAD http://cwlota.com/<bucket>/...: 400 Bad Request
```

CoreWeave LOTA requires a virtual-hosted URL such as
`http://<bucket>.cwlota.com/...`. Confirm the request in the worker logs:

```bash
uv run iris --cluster marin job logs <WORKER_JOB_ID> --level error | \
  rg 'Generic S3 HEAD|400 Bad Request'
```

Zephyr qualifies the endpoint in `zephyr.shuffle._scan_scatter_parquet` before
calling Polars. Check `AWS_ENDPOINT_URL_S3` and `AWS_ENDPOINT_URL` if the bad
URL persists. `FSSPEC_S3` does not configure Polars' Rust `object_store`
client, so changing fsspec retries, credentials, or worker RAM does not repair
this addressing error. See the [incident record](https://echo.oa.dev/wiki/59)
for the reproduced failure and validation.

### Stale Pipeline Warning

Grafana evaluates `ZephyrPipelineProgressStalled` once each minute. The rule
becomes pending after 45 minutes without a shard completion. The warning
becomes active after five more minutes.

The coordinator publishes `progress_time_seconds` through direct `service=zephyr` telemetry. The
metric resets at each stage start and after each shard completion. The metric
includes the root job ID and the Zephyr execution ID. Grafana removes a
producer when its most recent row is more than 90 seconds old.

This rule is a passive warning. It does not send a notification, restart a job,
or kick a task. A valid long shard can activate the warning. The warning means
that no shard completed during the time limit. It does not mean that item,
byte, CPU, or memory values stayed constant.

Use `get_status` to confirm the completed, in-flight, and queued shard counts.
Then compare the per-worker counters and collect thread profiles from the
in-flight workers. Do not restart or kick a task before you collect this
evidence.

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
