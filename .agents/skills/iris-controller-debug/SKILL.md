---
name: iris-controller-debug
description: Debug Iris controller state using offline checkpoint snapshots and the process RPC. Use when investigating stuck jobs, resource leaks, scheduling failures, or worker issues.
---

# Skill: Iris Controller Debug

Debug Iris controller issues by triggering a fresh checkpoint, downloading it, and querying offline. Use the `/system/process` RPC (via `iris process`) for profiling; SSH is acceptable as a fallback when RPC doesn't cover your needs.

Read first: @lib/iris/AGENTS.md

## Access Pattern

**Always debug offline against a checkpoint copy — never run queries against the live controller DB.**
Trigger a fresh checkpoint on-demand and download it:

```bash
# Trigger a fresh checkpoint (uses the BeginCheckpoint RPC)
# The --config flag selects the cluster; adjust as needed.
uv run iris --config lib/iris/examples/marin.yaml cluster controller checkpoint
# Example output:
#   Checkpoint DB written: gs://marin-us-central2/iris/marin/state/controller-state/checkpoint-1773533644027.sqlite3
#   Jobs:    417
#   Tasks:   46790
#   Workers: 243

# Download the checkpoint (use the path from the output above)
gcloud storage cp gs://marin-us-central2/iris/marin/state/controller-state/checkpoint-<EPOCH_MS>.sqlite3 /tmp/controller.sqlite3

# Query offline
python3 -c "
import sqlite3
conn = sqlite3.connect('/tmp/controller.sqlite3')
conn.row_factory = sqlite3.Row
# ... queries ...
"
```

If the state has changed and you need another snapshot, trigger another checkpoint with `uv run iris --config lib/iris/examples/marin.yaml cluster controller checkpoint` and re-download. **Do not SSH into the controller VM to run scripts against the live database** — a slow query can stall the controller and break other users.

**Production defaults** (verify at session start):
- Checkpoint location: `gs://marin-us-central2/iris/marin/state/controller-state/latest.sqlite3`
- Timestamped checkpoints: `gs://marin-us-central2/iris/marin/state/controller-state/checkpoint-<epoch_ms>.sqlite3`

## Profiling

Prefer the `iris process` CLI which talks to the controller via the `/system/process` RPC. If the RPC endpoints don't cover what you need, SSH is acceptable as a fallback:

```bash
# Thread dump (instant snapshot of all threads)
uv run iris process profile threads

# CPU profile (writes speedscope JSON, 10s default)
uv run iris process profile cpu --duration 10 --output /tmp/profile.speedscope.json

# Memory profile (writes flamegraph HTML)
uv run iris process profile mem --duration 10 --output /tmp/profile.html

# Target a specific worker instead of the controller
uv run iris process profile threads --worker <WORKER_ID>

# Process status (host info, resource usage)
uv run iris process status
```

Controller logs can also be fetched via the CLI:

```bash
# Tail controller logs
uv run iris process logs --max-lines 200

# Filter for slow-path warnings
uv run iris process logs --substring "Slow "
```

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

4. **Heartbeat thread stall on gcloud subprocess**: The heartbeat loop calls `notify_worker_failed` → `scale_down` → `terminate` which runs a synchronous `gcloud compute tpus tpu-vm delete` (`gcp.py:591`). If the gcloud API hangs, **all task dispatch stops** because dispatches are delivered via heartbeats. Symptoms: `dispatch_queue` growing, tasks stuck in ASSIGNED (9), stale `last_heartbeat_ms` across all workers. The autoscaler thread has the same exposure independently. Check with `py-spy dump` — look for `subprocess.run` → `terminate` on the heartbeat or autoscaler thread. The stuck gcloud process can be killed to unblock (#3678).

## Rules

- **NEVER run scripts or queries against the live controller DB.** Always work offline against a downloaded checkpoint. A slow or locking query on the live DB can stall the controller and break other users.
- **Prefer `iris process profile` over SSH for profiling.** It uses the `/system/process` RPC and avoids direct access to the controller VM. SSH is acceptable as a fallback when the RPC endpoints don't cover what you need.
- **NEVER modify the database without explicit user approval.** Read-only queries on the local checkpoint copy only; writes only as a last resort with user consent on a fresh checkpoint.
- **NEVER restart the controller or Docker container** — this kills all running jobs cluster-wide.
