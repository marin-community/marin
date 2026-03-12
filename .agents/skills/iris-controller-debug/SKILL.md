---
name: iris-controller-debug
description: Debug Iris controller state using the live SQLite database. Use when investigating stuck jobs, resource leaks, scheduling failures, or worker issues.
---

# Skill: Iris Controller Debug

Debug Iris controller issues by querying the live SQLite database on the controller VM.

Read first: @lib/iris/AGENTS.md

## Access Pattern

The DB is inside a Docker container on a GCP VM. Pipe a Python script through SSH:

```bash
cat <<'PYEOF' | gcloud compute ssh <VM> --zone=<ZONE> --project=<PROJECT> \
  --command="cat > /tmp/query.py && docker cp /tmp/query.py <CONTAINER>:/tmp/ && docker exec <CONTAINER> python3 /tmp/query.py"
import sqlite3
conn = sqlite3.connect("/tmp/iris/controller-logs/controller.sqlite3")
conn.row_factory = sqlite3.Row
# ... queries ...
PYEOF
```

**Find the container ID** with `docker ps` on the VM — look for the iris controller container.

**Production defaults** (verify at session start):
- VM: `iris-controller-marin`, zone `us-central1-a`, project `hai-gcp-models`
- DB path: `/tmp/iris/controller-logs/controller.sqlite3`

## Schema

Core tables: `jobs`, `tasks`, `task_attempts`, `workers`, `worker_attributes`, `scaling_groups`, `slices`, `tracked_workers`, `dispatch_queue`, `endpoints`, `reservation_claims`, `logs`, `txn_log` / `txn_actions`.

The schema is self-describing — query `sqlite_master` or `.schema` if you need column details. Below are only the non-obvious parts.

### State Integers

Task and job states share the same mapping:

| State | Value | Terminal? |
|---|---|---|
| PENDING | 1 | No |
| BUILDING | 2 | No |
| RUNNING | 3 | No |
| SUCCEEDED | 4 | Yes |
| FAILED | 5 | Yes |
| KILLED | 6 | Yes |
| WORKER_FAILED | 7 | Yes |
| UNSCHEDULABLE | 8 | Yes |
| ASSIGNED | 9 | No |

**Active states** (task is holding resources): **2, 3, 9** — not just RUNNING. Forgetting ASSIGNED (9) will cause you to miss resource attribution and misdiagnose leaks.

### Workers: Committed vs Capacity

`workers` has `committed_cpu_millicores`, `committed_mem_bytes`, `committed_gpu`, `committed_tpu` — these are the resources currently reserved. Total capacity is in the `metadata_proto` blob (serialized `WorkerMetadata`). Available = capacity − committed.

### Worker Attributes

`worker_attributes` is a key-value table per worker with typed columns (`str_value`, `int_value`, `float_value`). Well-known keys: `zone`, `region`, `device_type`, `device_variant`, `device_count`, `preemptible`.

### request_proto

`jobs.request_proto` is a serialized protobuf (`LaunchJobRequest`). You need protobuf to decode task constraints from it — you cannot inspect constraints with plain SQL.

### Transaction Log

`txn_log` + `txn_actions` is an audit trail of state mutations (capped at ~1000 entries). Useful for answering "what happened to this task/worker recently?"

## Known Bugs & Traps

These are real issues in the codebase that will mislead you if you don't know about them:

1. **Committed resource leak**: `_decommit_worker_resources()` in `transitions.py` can fail to fire on certain task termination paths, leaving stale committed resources on workers. Symptom: workers show high committed CPU/memory/TPU with zero active tasks. Detect by joining `workers` against active tasks in `task_attempts`; fix by zeroing committed fields on affected workers (requires user approval).

2. **Diagnostic masking**: The scheduler returns a detailed rejection reason when a task can't be placed — but `service.py:654` unconditionally replaces it with the autoscaler hint, which is often the useless "Waiting for workers to become ready". The real failure reason (constraint mismatch, insufficient resources) is hidden. Don't trust the pending reason shown in the UI; query the scheduler state directly.

3. **Fleet view zone display**: `FleetTab.vue:82` reads `metadata.gceZone` (never populated) instead of `metadata.attributes["zone"]`. The dashboard shows blank zones even when workers have zone attributes.

## Rules

- **NEVER modify the database without explicit user approval.** Read-only queries first; writes only as a last resort with user consent. Always run a verification query after any write.
- **NEVER restart the controller or Docker container** — this kills all running jobs cluster-wide.
- Always verify VM name, zone, and container ID at the start of each session.
