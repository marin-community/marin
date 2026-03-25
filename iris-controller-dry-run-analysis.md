# Iris Controller Dry-Run Mode: Codebase Analysis

## Controller Startup Flow

### Entry point: `lib/iris/src/iris/cluster/controller/main.py`

The controller daemon is a Click CLI:

```
cli -> serve command (main.py:34-239)
```

`serve` takes `--host`, `--port`, `--scheduler-interval`, `--config`, `--log-level`, `--checkpoint-path`, `--checkpoint-interval`.

Startup sequence:
1. Load cluster config from YAML (`load_config()`) — `main.py:81`
2. Resolve `remote_state_dir` and `local_state_dir` — `main.py:88-100`
3. Restore or create local SQLite DB from checkpoint — `main.py:106-121`
4. Create provider via `make_provider()` — `main.py:124`
5. Create autoscaler (if not K8sTaskProvider) — `main.py:128-165`
6. Create `ControllerConfig` dataclass — `main.py:182-193`
7. Instantiate `Controller(config, provider, autoscaler, db)` — `main.py:196`
8. Call `controller.start()` — `main.py:208`
9. Register SIGTERM/SIGINT handler that checkpoints + stops — `main.py:218-237`
10. Block on `stop_event.wait()` — `main.py:239`

### Controller.__init__: `controller.py:712-799`

Creates: `ControllerDB`, `LogStore`, `ControllerTransitions`, `Scheduler`, `BundleStore`, `ControllerServiceImpl` (RPC), `ControllerDashboard` (HTTP + dashboard UI).

### Controller.start(): `controller.py:832-871`

Spawns background threads depending on provider type:

**Worker provider path** (standard):
- `_run_scheduling_loop` thread — task assignment
- `_run_provider_loop` thread — heartbeats/sync with workers
- `_run_profile_loop` thread — periodic CPU profiling
- `_run_autoscaler_loop` thread (if autoscaler configured)
- uvicorn server thread (dashboard + RPC)

**K8sTaskProvider path** (direct):
- `_run_direct_provider_loop` thread — combined scheduling + sync
- uvicorn server thread

## Main Loop: Scheduling (`_run_scheduling_loop`)

Location: `controller.py:921-936`

Calls `_run_scheduling()` (`controller.py:1202-1315`) each cycle. This is the core scheduling function:

1. **Read reservation claims** from DB — `controller.py:1224`
2. **Cleanup stale claims** (dead workers, finished jobs) — `controller.py:1225`
3. **Claim workers for reservations** — `controller.py:1226` → `_claim_workers_for_reservations()` at `controller.py:1151`
4. **Read pending tasks** via `_schedulable_tasks()` — `controller.py:1232`
5. **Read healthy workers** — `controller.py:1233`
6. **Filter tasks**: check scheduling deadlines, reservation gates, per-job caps — `controller.py:1250-1273`
7. **Inject reservation taints** on workers — `controller.py:1281-1282`
8. **Create SchedulingContext** — `controller.py:1286-1291`
9. **Phase 1**: Preference pass (steer reservation tasks to claimed workers) — `controller.py:1295`
10. **Phase 2**: Normal scheduler `find_assignments()` — `controller.py:1298`
11. **Buffer assignments** → `_buffer_assignments()` → `transitions.queue_assignments()` — `controller.py:1303`
12. **Cache diagnostics** for unassigned jobs — `controller.py:1315`

### Where task assignment happens

- `_buffer_assignments()` at `controller.py:1359-1367`: calls `transitions.queue_assignments(command)` which writes ASSIGNED state to DB and enqueues dispatch batches
- `transitions.queue_assignments()` at `transitions.py:840`: the actual DB mutation

## Provider Sync / Heartbeat Loop (`_run_provider_loop`)

Location: `controller.py:972-987`

Calls `_sync_all_execution_units()` (`controller.py:1454-1553`):

1. **Reap stale workers** — `controller.py:1460`
2. **Drain dispatch batches** for all healthy workers — `transitions.drain_dispatch_all()` at `controller.py:1464`
3. **Provider.sync(batches)** — sends RPCs to workers (start tasks, heartbeat, collect status) — `controller.py:1470`
4. **Apply results**: `transitions.apply_heartbeats_batch()` for successes, `transitions.fail_heartbeat()` for failures — `controller.py:1475-1527`
5. **Kill tasks** on workers if needed — `controller.py:1529-1530`
6. **Notify autoscaler** of worker failures — `controller.py:1501-1523`

### Direct provider path (`_sync_direct_provider`)

Location: `controller.py:1005-1019`

For K8sTaskProvider:
1. `transitions.drain_for_direct_provider()` — gets tasks to run, running tasks, tasks to kill
2. `provider.sync(batch)` — applies to K8s
3. `transitions.apply_direct_provider_updates()` — applies results

## Autoscaler Loop (`_run_autoscaler_loop`)

Location: `controller.py:952-970`

Calls `_run_autoscaler_once()` (`controller.py:1555-1572`):
1. Build worker status map — `controller.py:1563`
2. `autoscaler.refresh(worker_status_map)` — probes cloud API for VM status
3. Compute demand entries — `controller.py:1566-1571`
4. `autoscaler.update(demand_entries)` — scales up/down VMs

Also runs periodic checkpoints if configured — `controller.py:966-970`.

## Checkpoint/Archive System

- `write_checkpoint()` at `checkpoint.py:76-130`: SQLite hot-backup → upload to `{remote_state_dir}/controller-state/{epoch_ms}/`
- `download_checkpoint_to_local()` at `checkpoint.py:170-218`: find latest checkpoint, download to local DB dir
- `begin_checkpoint()` at `controller.py:1587-1603`: sets `_checkpoint_in_progress=True`, acquires heartbeat lock, writes checkpoint, unsets flag
- Periodic checkpoint runs in autoscaler loop — `controller.py:966-970`
- Atexit checkpoint — `controller.py:913-919`

## Recommended Approach for --dry-run Flag

### What dry-run should do

Show what the controller *would* do without actually doing it. The controller should:
- ✅ Load config, restore DB, create all objects normally
- ✅ Run the scheduling loop to compute assignments
- ✅ Probe workers (heartbeats) to discover cluster state
- ❌ NOT dispatch task assignments to workers (suppress `transitions.queue_assignments()`)
- ❌ NOT start/stop VMs via autoscaler (`autoscaler.update()` should be read-only or skipped)
- ❌ NOT kill tasks on workers
- ❌ NOT write checkpoints (state hasn't changed)
- ❌ NOT modify reservation claims in DB
- ✅ Log what would happen (assignments, autoscaler decisions)

### Where to add the flag

1. **CLI**: Add `--dry-run` to the `serve` command in `main.py:34`.

2. **ControllerConfig**: Add `dry_run: bool = False` field at `controller.py:604`.

3. **Controller**: Gate side-effectful operations on `self._config.dry_run`:

   | Method | Location | What to gate |
   |--------|----------|-------------|
   | `_buffer_assignments()` | `controller.py:1359` | Skip `transitions.queue_assignments()`, log assignments instead |
   | `_claim_workers_for_reservations()` | `controller.py:1151` | Skip `transitions.replace_reservation_claims()`, log claims |
   | `_cleanup_stale_claims()` | `controller.py:1119` | Skip DB writes |
   | `kill_tasks_on_workers()` | `controller.py:1392` | Skip `buffer_kill`/`buffer_direct_kill` |
   | `_mark_task_unschedulable()` | `controller.py:1369` | Skip `transitions.mark_task_unschedulable()` |
   | `_sync_all_execution_units()` | `controller.py:1454` | Run heartbeats for probing but skip applying dispatch; or skip entirely |
   | `_sync_direct_provider()` | `controller.py:1005` | Skip `provider.sync()` |
   | `_run_autoscaler_once()` | `controller.py:1555` | Skip `autoscaler.update()` (scale decisions), keep `autoscaler.refresh()` for status |
   | `begin_checkpoint()` | `controller.py:1587` | Skip entirely |
   | `_maybe_prune()` | `controller.py:938` | Skip entirely |

4. **Logging in dry-run mode**: In `_run_scheduling()`, after computing `all_assignments`, log them:
   ```python
   if self._config.dry_run:
       for task_id, worker_id in all_assignments:
           logger.info("[DRY-RUN] Would assign %s -> %s", task_id, worker_id)
       return
   ```

### Alternative: read-only DB wrapper

Instead of scattering `if dry_run` checks, wrap `ControllerTransitions` with a subclass that logs mutations but doesn't execute them. This is cleaner but requires more upfront work since `ControllerTransitions` is not designed for composition.

### Simplest viable approach

The easiest path with minimal code changes:

1. Add `dry_run: bool` to `ControllerConfig`
2. In `_run_scheduling()` at line 1301: if dry_run, log assignments and return early instead of calling `_buffer_assignments()`
3. In `_claim_workers_for_reservations()`: if dry_run, skip the `transitions.replace_reservation_claims()` call
4. In `_run_autoscaler_once()`: skip `autoscaler.update()`
5. In `_sync_all_execution_units()`: skip the `provider.sync()` call (no RPCs to workers)
6. Skip checkpoints and pruning

This keeps the scheduling logic running normally (reads are fine) but suppresses all writes and RPCs. The dashboard + RPC server still runs so you can inspect state.

### Open questions

1. Should dry-run mode still accept new job submissions via RPC? Probably yes for testing, but the jobs would never be dispatched.
2. Should dry-run mode replay from a checkpoint without probing workers? This would be useful for offline analysis ("what would the controller do with this checkpoint?").
3. Should the autoscaler compute and *log* demand without acting? Or skip entirely?
