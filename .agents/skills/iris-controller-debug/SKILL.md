---
name: iris-controller-debug
description: Debug Iris controller state using the live SQLite database. Use when investigating stuck jobs, resource leaks, scheduling failures, or worker issues.
---

# Skill: Iris Controller Debug

Debug Iris controller issues by querying the live SQLite database on the controller VM.

## Prerequisites

Read first:

@lib/iris/AGENTS.md

## Access Pattern

The controller DB is inside a Docker container on a GCP VM. Run queries by
piping a Python script through SSH:

```bash
cat <<'PYEOF' | gcloud compute ssh <VM> --zone=<ZONE> --project=<PROJECT> \
  --command="cat > /tmp/query.py && docker cp /tmp/query.py <CONTAINER>:/tmp/ && docker exec <CONTAINER> python3 /tmp/query.py"
import sqlite3
conn = sqlite3.connect("/tmp/iris/controller-logs/controller.sqlite3")
conn.row_factory = sqlite3.Row
# ... queries ...
PYEOF
```

**Finding the container ID**: Run `gcloud compute ssh <VM> --zone=<ZONE> --project=<PROJECT> --command="docker ps"` and look for the iris controller container.

**Default production values** (verify before use):
- VM: `iris-controller-marin`, zone `us-central1-a`, project `hai-gcp-models`
- DB path inside container: `/tmp/iris/controller-logs/controller.sqlite3`

**Always confirm the VM name, zone, and container ID at the start of a debug session.**

## Database Schema Reference

### Core Tables

| Table | Purpose |
|---|---|
| `jobs` | Job hierarchy (parent/child). `state` = JOB_STATE_*, `request_proto` = serialized LaunchJobRequest |
| `tasks` | One row per task. `state` = TASK_STATE_*, `current_attempt_id` links to active attempt |
| `task_attempts` | Attempt history per task. Links task to `worker_id`. Multiple attempts on retry |
| `workers` | Worker nodes. Tracks health, committed resources, metadata proto |
| `worker_attributes` | Key-value pairs per worker (zone, region, device_type, etc.) |
| `scaling_groups` | Autoscaler state: backoff, quota, last scale times |
| `slices` | VM slices managed by autoscaler, lifecycle state |
| `tracked_workers` | Maps worker_id to slice_id and scale_group |
| `dispatch_queue` | Pending run/kill commands for workers |
| `endpoints` | Service endpoints registered by tasks |
| `reservation_claims` | Workers claimed by reservation-holding jobs |
| `logs` | Job/task output logs |
| `txn_log` / `txn_actions` | Audit trail of state mutations (last 1000 entries) |

### Task States

| State | Value | Meaning |
|---|---|---|
| PENDING | 1 | Waiting for worker assignment |
| BUILDING | 2 | Being set up on worker |
| RUNNING | 3 | Executing |
| SUCCEEDED | 4 | Completed OK |
| FAILED | 5 | Non-zero exit |
| KILLED | 6 | User-terminated |
| WORKER_FAILED | 7 | Worker died/preempted |
| UNSCHEDULABLE | 8 | Cannot be scheduled |
| ASSIGNED | 9 | Assigned but not yet started |

**Active states** (task is occupying resources): 2, 3, 9
**Terminal states**: 4, 5, 6, 7, 8

### Job States

Same integer mapping as task states (PENDING=1 through UNSCHEDULABLE=8).

### Worker Fields

```
worker_id           TEXT PRIMARY KEY
healthy             INTEGER (0/1)  -- 0 if heartbeat timeout or consecutive failures
active              INTEGER (0/1)  -- 0 if being deprovisioned
committed_cpu_millicores  INTEGER  -- CPU reserved for active tasks
committed_mem_bytes       INTEGER  -- Memory reserved for active tasks
committed_gpu             INTEGER  -- GPUs reserved
committed_tpu             INTEGER  -- TPUs reserved
metadata_proto            BLOB     -- WorkerMetadata (cpu_count, memory_bytes, device info)
```

Available resources = metadata capacity - committed.

### Worker Attributes

```
worker_id   TEXT
key         TEXT     -- e.g. "zone", "region", "device_type", "device_variant"
value_type  TEXT     -- "str", "int", or "float"
str_value   TEXT
int_value   INTEGER
float_value REAL
```

Well-known keys: `zone`, `region`, `device_type`, `device_variant`, `device_count`, `preemptible`.

## Diagnostic Queries

### 1. Job Status

Find a job and its tasks:

```python
# Find jobs by user or name pattern
rows = conn.execute("""
    SELECT job_id, state, submitted_at_ms, finished_at_ms, error, num_tasks
    FROM jobs
    WHERE job_id LIKE ?
    ORDER BY submitted_at_ms DESC LIMIT 10
""", ('%PATTERN%',)).fetchall()

for r in rows:
    print(f"  {r['job_id']} state={r['state']} tasks={r['num_tasks']} err={r['error']}")
```

### 2. Task Status for a Job

```python
rows = conn.execute("""
    SELECT t.task_id, t.state, t.error, a.worker_id, a.attempt_id
    FROM tasks t
    LEFT JOIN task_attempts a
        ON t.task_id = a.task_id AND t.current_attempt_id = a.attempt_id
    WHERE t.job_id = ?
""", (JOB_ID,)).fetchall()

for r in rows:
    print(f"  {r['task_id']} state={r['state']} worker={r['worker_id']} attempt={r['attempt_id']}")
```

### 3. Stuck Pending Tasks

```python
import time
now_ms = int(time.time() * 1000)

rows = conn.execute("""
    SELECT t.task_id, t.job_id, t.submitted_at_ms,
           (? - t.submitted_at_ms) / 1000 as wait_sec
    FROM tasks t
    WHERE t.state = 1
    ORDER BY t.submitted_at_ms ASC
    LIMIT 20
""", (now_ms,)).fetchall()

for r in rows:
    print(f"  {r['task_id']} job={r['job_id']} waiting={r['wait_sec']}s")
```

### 4. Worker Resource Utilization

Show committed vs total resources per worker in a region:

```python
rows = conn.execute("""
    SELECT w.worker_id, w.healthy, w.active,
           w.committed_cpu_millicores, w.committed_mem_bytes,
           w.committed_gpu, w.committed_tpu
    FROM workers w
    JOIN worker_attributes wa ON w.worker_id = wa.worker_id
        AND wa.key = 'region' AND wa.str_value = ?
    WHERE w.healthy = 1 AND w.active = 1
""", (REGION,)).fetchall()

for r in rows:
    gb = round(r["committed_mem_bytes"] / (1024**3), 1)
    print(f"  {r['worker_id']} cpu={r['committed_cpu_millicores']}m mem={gb}GB "
          f"gpu={r['committed_gpu']} tpu={r['committed_tpu']}")
```

### 5. Committed Resource Leak Detection

Find workers with committed resources but no active tasks:

```python
rows = conn.execute("""
    SELECT w.worker_id, w.committed_cpu_millicores, w.committed_mem_bytes,
           w.committed_gpu, w.committed_tpu,
           (SELECT COUNT(*) FROM tasks t
            JOIN task_attempts a ON t.task_id = a.task_id
                AND t.current_attempt_id = a.attempt_id
            WHERE a.worker_id = w.worker_id AND t.state IN (2, 3, 9)
           ) as active_tasks
    FROM workers w
    WHERE w.healthy = 1 AND w.active = 1
      AND (w.committed_cpu_millicores > 0 OR w.committed_mem_bytes > 0
           OR w.committed_gpu > 0 OR w.committed_tpu > 0)
""").fetchall()

leaked = [r for r in rows if r["active_tasks"] == 0]
print(f"Workers with committed resources: {len(rows)}")
print(f"Workers with leak (committed but no active tasks): {len(leaked)}")
for r in leaked:
    gb = round(r["committed_mem_bytes"] / (1024**3), 1)
    print(f"  {r['worker_id']}: cpu={r['committed_cpu_millicores']}m mem={gb}GB "
          f"tpu={r['committed_tpu']} active_tasks={r['active_tasks']}")
```

### 6. Reset Leaked Committed Resources

**This modifies live state. Confirm with the user before running.**

```python
cur = conn.execute("""
    UPDATE workers
    SET committed_cpu_millicores = 0, committed_mem_bytes = 0,
        committed_gpu = 0, committed_tpu = 0
    WHERE worker_id NOT IN (
        SELECT DISTINCT a.worker_id FROM tasks t
        JOIN task_attempts a ON t.task_id = a.task_id
            AND t.current_attempt_id = a.attempt_id
        WHERE t.state IN (2, 3, 9) AND a.worker_id IS NOT NULL
    ) AND (committed_cpu_millicores > 0 OR committed_mem_bytes > 0
           OR committed_gpu > 0 OR committed_tpu > 0)
""")
conn.commit()
print(f"Reset {cur.rowcount} workers")
```

### 7. Worker Attributes & Constraints

```python
# All attributes for a specific worker
rows = conn.execute("""
    SELECT key, value_type, str_value, int_value, float_value
    FROM worker_attributes WHERE worker_id = ?
""", (WORKER_ID,)).fetchall()

for r in rows:
    val = r["str_value"] or r["int_value"] or r["float_value"]
    print(f"  {r['key']} = {val}")
```

### 8. Workers by Zone/Scale Group

```python
# Workers per zone
rows = conn.execute("""
    SELECT wa.str_value as zone, COUNT(*) as cnt,
           SUM(w.healthy) as healthy, SUM(w.active) as active
    FROM workers w
    JOIN worker_attributes wa ON w.worker_id = wa.worker_id AND wa.key = 'zone'
    GROUP BY wa.str_value
""").fetchall()

for r in rows:
    print(f"  {r['zone']}: {r['cnt']} total, {r['healthy']} healthy, {r['active']} active")
```

### 9. Autoscaler State

```python
rows = conn.execute("""
    SELECT name, consecutive_failures, backoff_until_ms,
           last_scale_up_ms, quota_exceeded_until_ms, quota_reason
    FROM scaling_groups
""").fetchall()

import time
now_ms = int(time.time() * 1000)
for r in rows:
    backoff = max(0, (r["backoff_until_ms"] - now_ms) // 1000)
    quota = max(0, (r["quota_exceeded_until_ms"] - now_ms) // 1000)
    print(f"  {r['name']}: failures={r['consecutive_failures']} "
          f"backoff={backoff}s quota_block={quota}s reason={r['quota_reason']}")
```

### 10. Slice Lifecycle

```python
rows = conn.execute("""
    SELECT s.slice_id, s.scale_group, s.lifecycle, s.worker_ids, s.error_message
    FROM slices s
    ORDER BY s.scale_group, s.created_at_ms DESC
""").fetchall()

for r in rows:
    print(f"  {r['slice_id']} group={r['scale_group']} "
          f"lifecycle={r['lifecycle']} workers={r['worker_ids']} err={r['error_message']}")
```

### 11. Recent Transaction Log

```python
rows = conn.execute("""
    SELECT tl.id, tl.kind, tl.created_at_ms,
           ta.action, ta.entity_id, ta.details_json
    FROM txn_log tl
    JOIN txn_actions ta ON tl.id = ta.txn_id
    ORDER BY tl.id DESC
    LIMIT 30
""").fetchall()

for r in rows:
    print(f"  txn={r['id']} kind={r['kind']} action={r['action']} "
          f"entity={r['entity_id']}")
```

### 12. Task Attempt History for a Worker

Useful for understanding what ran on a worker and when:

```python
rows = conn.execute("""
    SELECT a.task_id, a.attempt_id, a.state, a.created_at_ms,
           a.started_at_ms, a.finished_at_ms, a.exit_code, a.error
    FROM task_attempts a
    WHERE a.worker_id = ?
    ORDER BY a.created_at_ms DESC
    LIMIT 20
""", (WORKER_ID,)).fetchall()

for r in rows:
    print(f"  {r['task_id']} attempt={r['attempt_id']} state={r['state']} "
          f"exit={r['exit_code']} err={r['error']}")
```

## Debugging Playbook

### Stuck Job Investigation

1. **Find the job** — Query `jobs` table by name pattern or user_id
2. **Check task states** — Are tasks PENDING (1) or stuck in ASSIGNED/BUILDING (9/2)?
3. **If PENDING**: Check worker availability in the required zone/region. Look for committed resource leaks (query 5). Check autoscaler state (query 9) and scaling group backoff/quota
4. **If ASSIGNED/BUILDING**: Check the worker health. Look at dispatch_queue for pending commands. Check task attempt history
5. **Check constraints**: Compare task's required attributes (from job's `request_proto`) against worker attributes. A constraint mismatch means no eligible workers

### Resource Leak Investigation

1. Run leak detection query (5) to find workers with committed resources but no active tasks
2. Check `txn_log` for recent decommit actions on those workers
3. If leak confirmed, reset with query (6) after user approval
4. **Root cause**: The leak occurs when `_decommit_worker_resources()` in `transitions.py` doesn't fire — typically when a task termination path skips the decommit step

### Scheduling Failure Investigation

1. Check pending tasks and their wait time (query 3)
2. Check worker resources in target region (query 4)
3. Check if the pending reason displayed in the UI is accurate — `service.py:654` may mask the real scheduler rejection with a generic autoscaler hint
4. Decode the task's constraints from `request_proto` (requires protobuf parsing) and compare against `worker_attributes`

## Known Issues

- **Committed resource leak**: `_decommit_worker_resources()` in `transitions.py` can fail to release resources when tasks terminate through certain paths. Symptom: workers show high committed memory/CPU with zero active tasks. Fix: reset query (6)
- **Diagnostic masking**: `service.py:654` unconditionally replaces the scheduler's detailed rejection reason with the autoscaler hint, even when the hint is the generic "Waiting for workers to become ready". The real constraint/resource failure reason gets hidden
- **Fleet view zone display**: `FleetTab.vue:82` reads `metadata.gceZone` (never populated) instead of `metadata.attributes["zone"]`

## Rules

- **NEVER modify the database without explicit user approval.** Always run read-only diagnostic queries first
- **NEVER restart the controller or Docker container** — this disrupts all running jobs
- Always verify VM name, zone, and container ID at session start
- Prefer read-only investigation; only write (UPDATE/DELETE) as a last resort with user consent
- When writing to the DB, always include a verification query after the change
