# Iris Budgets & Preemption: Design Document

## 1. Problem

Iris has no per-user fairness, priority bands, or budget-based scheduling. Today's
ordering is purely depth-first FIFO (`controller.py:294-298`):

```sql
ORDER BY priority_neg_depth ASC, priority_root_submitted_ms ASC, submitted_at_ms ASC, task_id ASC
```

A user submitting 100 jobs starves all other users. There is no mechanism to express
that production workloads should preempt batch work, or that idle cluster capacity
should be available opportunistically.

Specific gaps:
- **No priority bands** — all tasks compete equally. No way to mark a pipeline as
  production-critical vs. batch exploration.
- **No user fairness** — `_schedulable_tasks()` (`controller.py:287`) returns tasks in
  global FIFO order with no per-user cap or fair-share scoring.
- **No budget tracking** — no table or in-memory state tracks per-user resource consumption.
- **No preemption for priority** — `kill_tasks_on_workers()` (`controller.py:1389`) exists
  but is only used for reservation preemption and explicit kills, never for priority-based eviction.

## 2. Proposed Solution

### Overview

Three layers, each independently useful:

1. **Priority bands** (PRODUCTION > INTERACTIVE > BATCH) as a coarse sort key.
2. **Budget-weighted fairness** within each band — users who have consumed fewer
   resources get scheduled first.
3. **Preemption loop** — a post-scheduling pass that evicts lower-priority running
   tasks when higher-priority tasks cannot be placed.

### Why This Approach

- **Bands + fairness** mirrors SLURM's "QOS + fair-share" model, which is proven at
  scale. K8s has bands but no fairness; DRF has fairness but no bands.
- **Soft budgets** (deprioritize, don't block) satisfy the "opportunistic" requirement:
  if nobody else is using the cluster, even a heavy user gets everything.
- **Preemption as a separate loop** keeps scheduling logic clean — the scheduler stays
  pure-functional, and preemption is a controller-level concern.
- **Value function** collapses heterogeneous resources into a single "cost" number,
  avoiding the complexity of full DRF while still accounting for the 1000× cost
  difference between accelerators and CPUs.

## 3. Data Model Changes

### 3.1 Proto Changes (`lib/iris/src/iris/rpc/cluster.proto`)

```protobuf
// After ExistingJobPolicy enum (~line 558)
enum PriorityBand {
  PRIORITY_BAND_UNSPECIFIED = 0;  // defaults to INTERACTIVE
  PRIORITY_BAND_PRODUCTION = 1;   // admin-only, never preempted
  PRIORITY_BAND_INTERACTIVE = 2;  // normal work, default
  PRIORITY_BAND_BATCH = 3;        // opportunistic, preemptible
}

// Inside LaunchJobRequest, after existing_job_policy field:
  PriorityBand priority_band = 33;

// New RPC messages inside Controller message:
  message SetUserBudgetRequest {
    string user_id = 1;
    int64 budget_limit = 2;      // max budget value (0 = unlimited)
    PriorityBand max_band = 3;   // highest band this user can submit to (lower number = higher priority)
  }
  message SetUserBudgetResponse {}

  message GetUserBudgetRequest {
    string user_id = 1;
  }
  message GetUserBudgetResponse {
    string user_id = 1;
    int64 budget_limit = 2;
    int64 budget_spent = 3;      // current running resource value
    PriorityBand max_band = 4;
  }

  message ListUserBudgetsRequest {}
  message ListUserBudgetsResponse {
    repeated GetUserBudgetResponse users = 1;
  }
```

Add RPCs to the `ControllerService`:
```protobuf
rpc SetUserBudget(Controller.SetUserBudgetRequest) returns (Controller.SetUserBudgetResponse);
rpc GetUserBudget(Controller.GetUserBudgetRequest) returns (Controller.GetUserBudgetResponse);
rpc ListUserBudgets(Controller.ListUserBudgetsRequest) returns (Controller.ListUserBudgetsResponse);
```

### 3.2 DB Migration (`migrations/0021_budgets.py`)

```sql
-- New column on tasks for priority band sort key
ALTER TABLE tasks ADD COLUMN priority_band INTEGER NOT NULL DEFAULT 2;
-- 1=PRODUCTION, 2=INTERACTIVE, 3=BATCH (lower number = higher priority)

-- User budgets table
CREATE TABLE IF NOT EXISTS user_budgets (
    user_id TEXT PRIMARY KEY REFERENCES users(user_id),
    budget_limit INTEGER NOT NULL DEFAULT 0,   -- 0 = unlimited
    max_band INTEGER NOT NULL DEFAULT 2,       -- highest band allowed; lower number = higher priority (1=prod, 2=interactive, 3=batch)
    updated_at_ms INTEGER NOT NULL
);

-- Seed user_budgets for all existing users with defaults
INSERT OR IGNORE INTO user_budgets(user_id, budget_limit, max_band, updated_at_ms)
SELECT user_id, 0, 2, created_at_ms FROM users;

-- Update the pending-tasks index to include band as the primary sort key
DROP INDEX IF EXISTS idx_tasks_pending;
CREATE INDEX idx_tasks_pending ON tasks(
    state,
    priority_band ASC,           -- PRODUCTION(1) < INTERACTIVE(2) < BATCH(3)
    priority_neg_depth ASC,
    priority_root_submitted_ms ASC,
    submitted_at_ms ASC,
    priority_insertion ASC
);
```

### 3.3 User Budget Model

Stored in `user_budgets` table. Fields:
- `budget_limit`: max resource-value that can be simultaneously running. 0 = unlimited.
  Enforcement is *soft* — over-budget users are deprioritized, not blocked.
- `max_band`: highest priority band the user can submit to. Prevents non-admins from
  using PRODUCTION band.

**Budget spend** is not stored — it's computed live from running tasks:
```sql
SELECT j.user_id, SUM(value_of(t.resources)) as budget_spent
FROM tasks t JOIN jobs j ON t.job_id = j.job_id
WHERE t.state IN (ASSIGNED, BUILDING, RUNNING)
GROUP BY j.user_id
```

This avoids drift between stored spend and actual resource usage.

## 4. Value Function

### Formula

```python
def resource_value(resources: cluster_pb2.ResourceSpecProto) -> int:
    accel_count = get_gpu_count(resources.device) + get_tpu_count(resources.device)
    ram_gb = resources.memory_bytes // (1024 ** 3)
    cpu_cores = resources.cpu_millicores // 1000
    return 1000 * accel_count + ram_gb + 5 * cpu_cores
```

### Location

New file: `lib/iris/src/iris/cluster/controller/budget.py`. This module owns:
- `resource_value()` — the value function
- `compute_user_spend()` — queries DB for per-user running resource value
- `PriorityBand` — Python StrEnum mirroring the proto enum

```python
from enum import StrEnum

class PriorityBand(StrEnum):
    PRODUCTION = "production"
    INTERACTIVE = "interactive"
    BATCH = "batch"

# Map proto enum value → PriorityBand
BAND_FROM_PROTO: dict[int, PriorityBand] = {
    0: PriorityBand.INTERACTIVE,  # UNSPECIFIED defaults to INTERACTIVE
    1: PriorityBand.PRODUCTION,
    2: PriorityBand.INTERACTIVE,
    3: PriorityBand.BATCH,
}

# Band → sort key (lower = higher priority)
BAND_SORT_KEY: dict[PriorityBand, int] = {
    PriorityBand.PRODUCTION: 1,
    PriorityBand.INTERACTIVE: 2,
    PriorityBand.BATCH: 3,
}
```

### Budget Spend Tracking

Budget spend = sum of `resource_value()` for all tasks owned by a user in
ASSIGNED/BUILDING/RUNNING states. Computed at the start of each scheduling cycle
(not stored), so it's always consistent with actual task state.

## 5. Priority Scoring

### Sort Order for Pending Tasks

Replace current `_schedulable_tasks()` ordering (`controller.py:294-298`) with:

```
ORDER BY
    priority_band ASC,                  -- PRODUCTION(1) > INTERACTIVE(2) > BATCH(3)
    priority_neg_depth ASC,             -- deeper tasks first (livelock prevention)
    priority_root_submitted_ms ASC,     -- older root jobs first
    submitted_at_ms ASC,               -- tiebreak by task submission
    task_id ASC                        -- deterministic
```

This is handled entirely by the DB index — no application-level re-sorting needed.

### Per-User Fairness Within a Band

After reading pending tasks from DB (sorted by band + depth + time), the controller
applies a **round-robin interleave** within each band. The call site must loop over
bands explicitly to prevent cross-band reordering:

```python
def interleave_by_user(tasks: list[Task], user_spend: dict[str, int]) -> list[JobName]:
    """Round-robin tasks across users, ordered by ascending budget spend.

    Must be called once per band — mixing bands in a single call would allow
    a low-spend user's BATCH tasks to leapfrog another user's INTERACTIVE tasks.
    """
    by_user: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        by_user[task.user_id].append(task)

    # Sort users by spend ascending (least-spending users first)
    sorted_users = sorted(by_user.keys(), key=lambda u: user_spend.get(u, 0))

    result: list[JobName] = []
    round_idx = 0
    while True:
        added = False
        for user in sorted_users:
            user_tasks = by_user[user]
            if round_idx < len(user_tasks):
                result.append(user_tasks[round_idx].task_id)
                added = True
        if not added:
            break
        round_idx += 1
    return result
```

**Call site** (in `_run_scheduling()`, after computing `user_spend`):

```python
# Group pending tasks by band, then interleave within each band separately.
# This ensures band ordering is preserved — PRODUCTION tasks always come before
# INTERACTIVE, regardless of per-user spend.
tasks_by_band: dict[int, list[Task]] = defaultdict(list)
for task in pending_tasks:
    tasks_by_band[task.priority_band].append(task)

interleaved: list[JobName] = []
for band in sorted(tasks_by_band.keys()):  # 1=PRODUCTION, 2=INTERACTIVE, 3=BATCH
    interleaved.extend(interleave_by_user(tasks_by_band[band], user_spend))
```

A user with budget_spend=0 gets their tasks interleaved before a user with
budget_spend=8000 (i.e., running 8 accelerators) — but only within the same band.

### Per-User Scheduling Cap

Add `max_tasks_per_user_per_cycle` to `ControllerConfig` (default: 8). Within the
scheduling loop, track `tasks_scheduled_per_user` and skip users who hit the cap.
This bounds scheduling CPU time and ensures fairness even without full interleaving.

## 6. Scheduling Changes

### Modifications to `_run_scheduling()` (`controller.py:1199`)

Current flow:
1. Read reservation claims
2. Read pending tasks (FIFO order)
3. Filter (deadlines, reservation gates, per-job cap)
4. Inject reservation taints
5. Preference pass (reservation jobs → claimed workers)
6. Normal pass (`find_assignments()`)
7. Buffer assignments

New flow (changes in **bold**):
1. Read reservation claims
2. Read pending tasks (**new sort order includes band**)
3. **Compute per-user budget spend** via `compute_user_spend()`
4. Filter (deadlines, reservation gates, per-job cap, **per-user cap**)
5. **Interleave tasks by user within each band**
6. Inject reservation taints
7. Preference pass
8. Normal pass
9. Buffer assignments
10. **Preemption pass** (see §7)

### Code Sketch for Step 4 (Per-User Cap)

```python
# In _run_scheduling(), after building schedulable_task_ids:
user_spend = compute_user_spend(self._db)
tasks_per_user: dict[str, int] = defaultdict(int)
user_cap = self._config.max_tasks_per_user_per_cycle

filtered: list[JobName] = []
for task_id in schedulable_task_ids:
    user = task_id.user
    if user_cap > 0 and tasks_per_user[user] >= user_cap:
        continue
    tasks_per_user[user] += 1
    filtered.append(task_id)
```

### When Budget Spend Changes

Budget spend is *not* tracked incrementally. It's recomputed each scheduling cycle
(every 0.5s) by summing resource values of active tasks. This is a single SQL query
and scales well for the task counts Iris handles (thousands, not millions).

## 7. Preemption Loop

### Algorithm

Runs after the normal scheduling pass, on the same thread, within `_run_scheduling()`.

```python
def _run_preemption_pass(
    self,
    unscheduled_tasks: list[tuple[JobName, JobRequirements]],
    context: SchedulingContext,
) -> list[tuple[JobName, JobName]]:  # (preemptor_task_id, victim_task_id)
    """Find tasks to preempt for higher-priority unscheduled work.

    Rules:
    - PRODUCTION preempts INTERACTIVE and BATCH.
    - INTERACTIVE preempts BATCH only.
    - BATCH never preempts.
    - Within same band, no preemption (compete via scheduling order).
    """
    preemptions: list[tuple[JobName, JobName]] = []
    running = self._get_running_tasks_with_band_and_value()

    # Sort victims: lowest priority first (highest band number), then lowest value (cheapest to preempt)
    victims = sorted(running, key=lambda t: (-t.band_sort_key, t.resource_value))

    freed_capacity: dict[WorkerId, int] = defaultdict(int)

    for task_id, req in unscheduled_tasks:
        task_band = self._get_task_band(task_id)
        if task_band == PriorityBand.BATCH:
            continue  # batch never preempts

        for victim in victims:
            if victim.already_preempted:
                continue
            if BAND_SORT_KEY[victim.band] <= BAND_SORT_KEY[task_band]:
                continue  # can only preempt strictly lower bands

            # Check if victim's worker can fit the preemptor
            worker_id = victim.worker_id
            cap = context.capacities.get(worker_id)
            if cap is None:
                continue
            if cap.can_fit(req) is None:
                # Worker already has capacity — no preemption needed
                break
            # Would freeing this victim create enough capacity?
            if self._freeing_victim_fits(victim, req, cap):
                preemptions.append((task_id, victim.task_id))
                victim.already_preempted = True
                break

    return preemptions
```

### Integration Point

After `find_assignments()` returns, collect unscheduled tasks from `context.pending_tasks`
that were NOT assigned. For each, check if preemption is possible.

```python
# In _run_scheduling(), after line 1297:
assigned_ids = {task_id for task_id, _ in all_assignments}
unscheduled = [
    (tid, jobs[tid.parent])
    for tid in schedulable_task_ids
    if tid not in assigned_ids and tid.parent in jobs
]
preemptions = self._run_preemption_pass(unscheduled, context)
for preemptor_id, victim_id in preemptions:
    self._transitions.preempt_task(victim_id, reason=f"preempted by {preemptor_id}")
    self.kill_tasks_on_workers({victim_id})
```

### Coscheduled/Gang Job Preemption

Preempting one task of a coscheduled job is useless — all tasks must be evicted.
The preemption loop must:
1. Identify the victim as part of a coscheduled group.
2. Find ALL tasks in the group running on the same `group_by` attribute value.
3. Preempt all of them atomically.
4. Only proceed if freeing all of them creates enough capacity for the preemptor.

For v1, coscheduled jobs are **not preemptible** — they are skipped as victims.
This is safe because multi-host TPU jobs are typically production-priority anyway.

### Interaction with Existing Reservation Preemption

Reservation preemption (`_preference_pass`) runs BEFORE the budget preemption loop.
The budget preemption loop should skip tasks on workers claimed by reservations,
since those workers are already spoken for.

### Preempted Task Lifecycle

A preempted task transitions to `WORKER_FAILED`, consuming from its preemption retry
budget (`max_retries_preemption`, default 100). This reuses the existing retry path
at `transitions.py` — no new state needed.

## 8. Admin API

### RPCs (`service.py`)

Three new handlers on `ControllerServiceImpl`:

```python
def set_user_budget(self, request, ctx):
    """Set budget limit and max band for a user. Admin-only."""
    authorize(AuthzAction.MANAGE_BUDGETS)
    self._transitions.set_user_budget(
        user_id=request.user_id,
        budget_limit=request.budget_limit,
        max_band=request.max_band,
    )

def get_user_budget(self, request, ctx):
    """Get budget config and current spend for a user."""
    budget = self._db.get_user_budget(request.user_id)
    spend = compute_user_spend(self._db).get(request.user_id, 0)
    return GetUserBudgetResponse(
        user_id=request.user_id,
        budget_limit=budget.budget_limit,
        budget_spent=spend,
        max_band=budget.max_band,
    )

def list_user_budgets(self, request, ctx):
    """List all user budgets with current spend."""
    ...
```

### Auth

Add `MANAGE_BUDGETS` to `AuthzAction` in `lib/iris/src/iris/rpc/auth.py`:
```python
class AuthzAction(StrEnum):
    REGISTER_WORKER = "register_worker"
    MANAGE_OTHER_KEYS = "manage_other_keys"
    MANAGE_BUDGETS = "manage_budgets"  # new
```

Policy: admin-only (default for actions not in POLICY dict).

### CLI Commands

Add to CLI (`lib/iris/src/iris/cli/main.py`):
```
iris user budget set <user_id> --limit <value> --max-band <band>
iris user budget get <user_id>
iris user budget list
```

### Config File Defaults

Add to `ControllerConfig`:
```python
@dataclass
class UserBudgetDefaults:
    budget_limit: int = 0           # 0 = unlimited
    max_band: str = "interactive"   # default max band for new users

@dataclass
class ControllerConfig:
    ...
    max_tasks_per_user_per_cycle: int = 8
    user_budget_defaults: UserBudgetDefaults = field(default_factory=UserBudgetDefaults)
```

When `ensure_user()` creates a new user, it also creates a `user_budgets` row with
these defaults using `INSERT OR IGNORE` (idempotent if the row already exists from
migration or a prior call):

```python
# In submit_job(), alongside the existing user INSERT (~transitions.py:565):
cur.execute(
    "INSERT OR IGNORE INTO users(user_id, created_at_ms) VALUES (?, ?)",
    (job_id.user, effective_submission_ms),
)
cur.execute(
    "INSERT OR IGNORE INTO user_budgets(user_id, budget_limit, max_band, updated_at_ms) "
    "VALUES (?, ?, ?, ?)",
    (job_id.user, defaults.budget_limit, BAND_SORT_KEY[defaults.max_band], effective_submission_ms),
)
```

Admins can override at runtime via the RPC.

### Band Inheritance for Child Jobs

Child (nested) jobs inherit their parent's `priority_band` at submission time. When
`launch_job()` processes a request with a `parent_job_id`, it reads the parent's band
and propagates it to all children, ignoring any band specified in the child request:

```python
# In launch_job(), after resolving parent_job_id:
if parent_job_id is not None:
    parent_band = cur.execute(
        "SELECT priority_band FROM tasks WHERE job_id = ? LIMIT 1",
        (parent_job_id,),
    ).fetchone()
    if parent_band is not None:
        band_sort_key = parent_band["priority_band"]
    # else: parent has no tasks yet (shouldn't happen), fall back to request band
```

This ensures a PRODUCTION pipeline's subtasks remain PRODUCTION, and a BATCH
pipeline cannot escalate child jobs to a higher band.

### Band Validation at Submission

In `launch_job()` (`service.py:714`), after resolving the user:
```python
band = BAND_FROM_PROTO.get(request.priority_band, PriorityBand.INTERACTIVE)
if band == PriorityBand.PRODUCTION:
    # Only admins can submit production jobs
    authorize(AuthzAction.MANAGE_BUDGETS)
user_budget = self._db.get_user_budget(user_id)
if BAND_SORT_KEY[band] < user_budget.max_band:
    raise ConnectError(Code.PERMISSION_DENIED, f"User {user_id} cannot submit {band} jobs")
```

## 9. File-by-File Implementation Plan

### Work Unit A: Proto + Codegen (no dependencies)

| File | Change |
|---|---|
| `lib/iris/src/iris/rpc/cluster.proto` | Add `PriorityBand` enum, `priority_band` field on `LaunchJobRequest`, budget RPC messages and service methods |
| Run `uv run python lib/iris/scripts/generate_protos.py` | Regenerate Python stubs |

**Tests**: proto compiles, new fields accessible from Python.

### Work Unit B: Budget Module (no dependencies)

| File | Change |
|---|---|
| `lib/iris/src/iris/cluster/controller/budget.py` | New file: `resource_value()`, `PriorityBand` StrEnum, `BAND_FROM_PROTO`, `BAND_SORT_KEY`, `compute_user_spend()` |

**Tests**: `lib/iris/tests/test_budget.py` — unit tests for `resource_value()` with
various resource specs (CPU-only, GPU, TPU, mixed). Test `compute_user_spend()` with
a mock DB or in-memory SQLite.

### Work Unit C: DB Migration (depends on A for proto values)

| File | Change |
|---|---|
| `lib/iris/src/iris/cluster/controller/migrations/0021_budgets.py` | Add `priority_band` column to tasks, create `user_budgets` table, rebuild pending index |
| `lib/iris/src/iris/cluster/controller/db.py` | Add `get_user_budget()`, `set_user_budget()` methods on `TransactionCursor`/`ControllerDB`. Add `priority_band` to `TASKS` table descriptor. |

**Tests**: migration applies cleanly, round-trip budget CRUD, verify new index is used
by EXPLAIN QUERY PLAN.

### Work Unit D: Scheduling Changes (depends on B, C)

| File | Change |
|---|---|
| `lib/iris/src/iris/cluster/controller/controller.py:287-301` | Update `_schedulable_tasks()` ORDER BY to include `priority_band ASC` as first key |
| `lib/iris/src/iris/cluster/controller/controller.py:604-678` | Add `max_tasks_per_user_per_cycle` and `user_budget_defaults` to `ControllerConfig` |
| `lib/iris/src/iris/cluster/controller/controller.py:1199-1312` | Modify `_run_scheduling()`: compute user spend, apply per-user cap, interleave by user within band |
| `lib/iris/src/iris/cluster/controller/transitions.py:523-622` | In `submit_job()`, write `priority_band` to tasks table. Read user's `max_band`, validate band. |

**Task INSERT sketch** — add `priority_band` column to the task INSERT in `transitions.py:610-628`:

```python
cur.execute(
    "INSERT INTO tasks("
    "task_id, job_id, task_index, state, error, exit_code, submitted_at_ms, started_at_ms, "
    "finished_at_ms, max_retries_failure, max_retries_preemption, failure_count, preemption_count, "
    "resource_usage_proto, current_attempt_id, priority_neg_depth, priority_root_submitted_ms, "
    "priority_insertion, priority_band"
    ") VALUES (?, ?, ?, ?, NULL, NULL, ?, NULL, NULL, ?, ?, 0, 0, NULL, -1, ?, ?, ?, ?)",
    (
        task_id,
        job_id.to_wire(),
        idx,
        cluster_pb2.TASK_STATE_PENDING,
        effective_submission_ms,
        int(request.max_retries_failure),
        int(request.max_retries_preemption),
        -job_id.depth,
        root_submitted_ms,
        insertion_base + idx,
        band_sort_key,  # from BAND_SORT_KEY[resolved_band]
    ),
)
```

**Tests**: `lib/iris/tests/test_scheduling_fairness.py` — integration tests:
- Two users, one with 1 task, one with 100 → verify the single-task user gets scheduled first.
- Tasks in PRODUCTION band scheduled before INTERACTIVE.
- Per-user cap limits scheduling throughput per user.
- Deeper tasks still get depth boost within same band.

### Work Unit E: Preemption Loop (depends on B, D)

| File | Change |
|---|---|
| `lib/iris/src/iris/cluster/controller/controller.py` | Add `_run_preemption_pass()` method. Call it at end of `_run_scheduling()`. |
| `lib/iris/src/iris/cluster/controller/transitions.py` | Add `preempt_task()` method — transitions task to WORKER_FAILED with preemption reason. (May reuse existing `_on_worker_failed` path.) |
| `lib/iris/src/iris/cluster/controller/budget.py` | Add `get_running_tasks_with_band()` — query for running tasks with their band, worker, and resource value. |

**Tests**: `lib/iris/tests/test_preemption.py`:
- PRODUCTION task preempts BATCH task on same worker.
- INTERACTIVE task preempts BATCH but not PRODUCTION.
- BATCH task never triggers preemption.
- Preempted task retries via preemption budget.
- Coscheduled tasks are not preempted (v1).

### Work Unit F: Admin API + CLI (depends on A, C)

| File | Change |
|---|---|
| `lib/iris/src/iris/rpc/auth.py:86` | Add `MANAGE_BUDGETS` to `AuthzAction` |
| `lib/iris/src/iris/cluster/controller/service.py` | Add `set_user_budget()`, `get_user_budget()`, `list_user_budgets()` handlers |
| `lib/iris/src/iris/cli/main.py` | Add `iris user budget set/get/list` commands |

**Tests**: `lib/iris/tests/test_budget_api.py`:
- Admin can set/get/list budgets.
- Non-admin cannot set budgets.
- Non-admin cannot submit PRODUCTION jobs.
- Budget spend reflects running tasks.

### Dependency Graph

```
A (proto)  ──┐
             ├──→ C (migration) ──→ D (scheduling) ──→ E (preemption)
B (budget) ──┘                  ──→ F (admin API)
```

Parallelism: A and B can run in parallel. C depends on both. D and F can run in
parallel after C. E depends on D.

## 10. Testing Plan

### Unit Tests

| Test File | What |
|---|---|
| `lib/iris/tests/test_budget.py` | `resource_value()` with GPU/TPU/CPU-only specs, edge cases (0 resources) |
| `lib/iris/tests/test_priority_scoring.py` | Sort order verification: band > depth > root_time > submitted_time |
| `lib/iris/tests/test_preemption_selection.py` | Victim selection: lowest band first, then lowest value. Coscheduled skip. |

### Integration Tests

| Test File | What |
|---|---|
| `lib/iris/tests/test_scheduling_fairness.py` | Full scheduling cycle with multiple users, bands, and budget levels |
| `lib/iris/tests/test_preemption.py` | End-to-end: submit batch job, submit production job, verify preemption occurs |
| `lib/iris/tests/test_budget_api.py` | RPC round-trip for budget CRUD, auth enforcement |

### Existing Tests to Update

- `lib/iris/tests/test_scheduler.py` — update `_schedulable_tasks` mock to include
  `priority_band` column.
- `lib/iris/tests/test_controller.py` — verify scheduling cycle still works with new
  column and interleaving.
- Any test creating tasks directly in DB needs the `priority_band` column.

## 11. Risks and Open Questions

1. **Scheduling cycle latency**: `compute_user_spend()` adds one SQL query per cycle.
   With WAL mode and the small task counts Iris handles, this should be <1ms. Monitor.

2. **Interleave ordering vs. DB index**: The DB returns tasks sorted by band + depth +
   time. The interleave step re-sorts by user within each band. This means the final
   order differs from the DB index order — but the scheduler already handles
   head-of-line blocking (skips tasks that don't fit), so this is fine.

3. **Coscheduled preemption (deferred)**: Preempting gang jobs requires evicting all
   tasks atomically, which is complex. Deferred to v2. For now, coscheduled jobs are
   not preemptible.

4. **Graceful preemption (deferred)**: Today's kill is immediate. A SIGTERM grace period
   for checkpointing would require worker-side changes and a new task state
   (PREEMPTING). Deferred.

5. **Value function weights**: The `1000 * accel + RAM_GB + 5 * CPU` formula treats
   all accelerator types equally. An H100 and a v5litepod chip have very different
   costs. Refinement: multiply by a per-variant weight from config. Not blocking for v1.

6. **Budget limit enforcement**: The design uses soft limits (deprioritize, don't block).
   If hard limits are needed later, add a check in `launch_job()` that rejects
   submissions when `budget_spent >= budget_limit`.

7. **Race between spend computation and assignment**: Spend is computed at cycle start,
   but assignments happen at cycle end. Two cycles could both compute the same spend
   and both schedule tasks. This is acceptable — the cap is soft, and the 0.5s cycle
   interval makes double-counting unlikely.

8. **Migration on live cluster**: The ALTER TABLE adds a column with a default. SQLite
   handles this without rewriting the table (since 3.1.3). The index rebuild may take
   a moment on large task tables but is safe under WAL mode.
