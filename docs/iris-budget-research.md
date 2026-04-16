# Iris Budget & Scheduling Research

Research report on the Iris scheduler, user tracking, and approaches to user budgets,
priorities, and preemption.

## 1. Existing Iris Architecture

### 1.1 Scheduler

The scheduler is a pure-functional module with no state mutation or threading.

**Key files:**
- `lib/iris/src/iris/cluster/controller/scheduler.py` — `Scheduler` class, `SchedulingContext`, `JobRequirements`, `WorkerCapacity`
- `lib/iris/src/iris/cluster/controller/controller.py:1200` — `_run_scheduling_cycle()`, the main scheduling loop
- `lib/iris/src/iris/cluster/controller/transitions.py:390` — `ControllerTransitions`, all DB-mutating state transitions

**Scheduling loop** (`Controller._run_scheduling_cycle`):

1. Read reservation claims, clean stale claims, claim workers for reservations.
2. Read pending tasks (sorted by priority columns) and healthy workers.
3. Filter tasks: check scheduling deadlines, reservation gates, per-job caps.
4. Inject reservation taints (claimed workers get attributes, non-reservation jobs get NOT_EXISTS constraints).
5. Build `SchedulingContext` with posting-list index for O(1) constraint matching.
6. **Phase 1 — Preference pass**: reservation-job tasks try claimed workers first.
7. **Phase 2 — Normal pass**: `Scheduler.find_assignments()` — coscheduled jobs first (all-or-nothing), then first-fit for remaining tasks.
8. Commit assignments via `ControllerTransitions.queue_assignments()`.
9. Cache scheduling diagnostics for unassigned jobs.

**Scheduling interval**: configurable via `ControllerConfig.scheduler_interval`, default 0.5s.

**Task ordering** (the current "priority system"):
```sql
ORDER BY priority_neg_depth ASC,       -- deeper tasks first (negative depth)
         priority_root_submitted_ms ASC, -- older root jobs first (FIFO)
         submitted_at_ms ASC,            -- older tasks first
         task_id ASC                     -- deterministic tiebreak
```
See `controller.py:295` and the pending-tasks index at `migrations/0001_init.py:184`.

This gives depth-first scheduling (children before siblings) with FIFO ordering among
same-depth tasks based on the root job's submission time. **There is no per-user fairness,
no priority bands, and no budget-based ordering.**

### 1.2 User Tracking

Users are tracked via the `JobName` hierarchy. The first path component is the user:
`/alice/train/eval/0` → user is `alice`.

**DB schema** (`migrations/0001_init.py:22`):
```sql
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    created_at_ms INTEGER NOT NULL
);
```
A `role` column was added later (migration 0006): `user` or `admin`.

**User creation**: `TransactionCursor.ensure_user()` at `db.py:1211` — called during job
submission. Creates user if not exists.

**User aggregation**: `_live_user_stats()` at `service.py:526` aggregates job/task counts
per user for the dashboard's Users tab. This is read-only; it does not influence scheduling.

**Auth**: `auth.py` handles JWT tokens, API keys, GCP identity verification. Users
authenticate via `iris login` (GCP OAuth) or API keys. The `VerifiedIdentity` includes
`user_id` used to construct `JobName.root(user, name)`.

### 1.3 Task Lifecycle

```
PENDING → ASSIGNED → BUILDING → RUNNING → SUCCEEDED/FAILED/KILLED/WORKER_FAILED/UNSCHEDULABLE
```

See `lib/iris/docs/task-states.md` for the full state machine. Key points:

- Jobs expand into N tasks at submission (`replicas` field).
- Tasks are the unit of scheduling and execution.
- Job state is derived from task state counts — no independent job state machine.
- Two independent retry budgets: `failure_count` vs `max_retries_failure`, `preemption_count` vs `max_retries_preemption`.
- `WORKER_FAILED` uses preemption budget (default 100 retries).

### 1.4 Resource Model

Resources are specified per-job via `ResourceSpecProto` (`cluster.proto:397`):
- `cpu_millicores` (int32)
- `memory_bytes` (int64)
- `disk_bytes` (int64)
- `DeviceConfig` — either `GpuDevice(variant, count)` or `TpuDevice(variant, count)`

Accelerator tasks get a minimum of 32 CPU cores (`MIN_ACCELERATOR_CPU_MILLICORES = 32000`).

Workers track committed resources: `committed_cpu_millicores`, `committed_mem_bytes`,
`committed_gpu`, `committed_tpu`. Available = total − committed.

CPU demand is fungible (any group). GPU/TPU demand is non-fungible (must match device type/variant).

### 1.5 Database

**SQLite** — single file, WAL mode. `ControllerDB` at `db.py` provides:
- `snapshot()` — read-only queries
- `transaction()` — `BEGIN IMMEDIATE` for mutations
- Thread-safe via `RLock`

Key tables: `jobs`, `tasks`, `task_attempts`, `workers`, `worker_attributes`,
`reservation_claims`, `dispatch_queue`, `endpoints`, `users`.

### 1.6 Controller Config

`ControllerConfig` at `controller.py:604` — dataclass with defaults:
- `scheduler_interval`: 0.5s
- `heartbeat_interval`: 5s
- `max_tasks_per_job_per_cycle`: 4 (caps non-coscheduled tasks per job per cycle)
- `heartbeat_failure_threshold`: 10
- No budget/priority/fairness config exists today.

### 1.7 Existing Priority System

**None beyond FIFO with depth-boost.** The `priority_neg_depth` column ensures child
tasks are scheduled before their parent's siblings (depth-first). Within the same depth,
`priority_root_submitted_ms` gives FIFO by root job submission time.

There is no concept of priority bands, user quotas, or fair-share weights.

### 1.8 Job Hierarchy (Parent/Child)

Jobs form a tree: `jobs.parent_job_id` references the parent, `root_job_id` tracks the
top-level job. `depth` is stored on both jobs and tasks. Child jobs inherit the root
submission time for priority ordering.

The `preemption_policy` field (`cluster.proto:623`) controls child behavior when a task
is preempted:
- `TERMINATE_CHILDREN` (default for single-task jobs)
- `PRESERVE_CHILDREN` (default for multi-task jobs)

### 1.9 APIs

**Connect/gRPC** — `ControllerServiceImpl` at `service.py` handles all RPCs:
- `LaunchJob` / `TerminateJob` / `GetJobStatus` / `ListJobs`
- `ListUsers` — aggregates per-user stats
- Worker registration, heartbeat, task state updates
- Endpoint management, log streaming, profiling

**CLI** — `iris job run`, `iris job list`, `iris job kill`, `iris login`, etc.

### 1.10 Preemption Mechanics

**Today's capabilities:**
- **Kill a running task**: `Controller.kill_tasks_on_workers()` at `controller.py:1389` buffers kill requests delivered via heartbeat. The worker's task attempt is killed (container stopped).
- **Cascade children**: `_cascade_children()` at `transitions.py:325` recursively kills descendant jobs.
- **Worker failure**: `_on_worker_failed` transitions all tasks on a dead worker to `WORKER_FAILED`, which triggers retry via preemption budget.
- **Reservation preemption**: The scheduling loop already preempts holder tasks on claimed workers when peer tasks are pending (Phase 1 of `_run_scheduling_cycle`).

**What's missing for budget-based preemption:**
- No mechanism to select *which* running task to preempt based on priority/budget.
- No "graceful preemption" signal — kill is immediate.
- No preemption scoring (e.g., "evict the lowest-priority task that frees enough resources").

## 2. External Approaches

### 2.1 Kubernetes Priority & Preemption

| Aspect | K8s Approach |
|---|---|
| Priority | `PriorityClass` objects map names → int32 values (higher = more important) |
| Scheduling | kube-scheduler sorts pending pods by priority; higher-priority pods scheduled first |
| Preemption | If a pod can't be scheduled, scheduler tries evicting lower-priority pods to make room |
| Graceful termination | Evicted pods get their `terminationGracePeriodSeconds` |
| Non-preempting | `preemptionPolicy: Never` — gets queue priority but won't evict others |
| System classes | `system-node-critical` (2B+1K), `system-cluster-critical` (2B) |
| Nomination | Preempting pod gets `nominatedNodeName`; node is "reserved" until evictions complete |

**Relevance to Iris**: K8s priority classes map directly to the proposed 3-band system.
The `preemptionPolicy: Never` concept maps to "batch" band (gets scheduled when idle,
never evicts). K8s does NOT do fair-share or per-user budgets.

### 2.2 SLURM Fair-Share

| Aspect | SLURM Approach |
|---|---|
| Priority formula | `priority = Σ(weight_i × factor_i)` for age, fairshare, job-size, partition, QOS |
| Fair-share | Compares target share (configured) vs actual usage (measured). Under-users get higher priority. |
| Hierarchy | Tree of accounts → sub-accounts → users, each with shares |
| Algorithm | Fair Tree (default since 19.05): if account A has higher fairshare than sibling B, all children of A outrank all children of B |
| Budget model | Soft — not a hard cutoff. Over-users get deprioritized but not blocked. |
| Preemption | Separate from fairshare; uses partition-based preemption or QOS-based preemption |
| Decay | Half-life on historical usage so recent usage matters more |

**Relevance to Iris**: SLURM's multi-factor priority with fair-share decay is the
gold-standard for HPC. The soft-budget model (deprioritize, don't block) matches the
"no one else using cluster" requirement. The hierarchical account model may be overkill
for Iris's flat user list.

### 2.3 Dominant Resource Fairness (DRF)

DRF (Ghodsi et al., NSDI 2011) generalizes max-min fairness to multiple resource types.

**Algorithm**: For each user, compute the dominant share = max(share of any resource).
Allocate to the user with the smallest dominant share. Repeat.

**Properties**: sharing incentive, strategy-proofness, Pareto efficiency, envy-freeness.

**Example**: User A needs <1 CPU, 4 GB>. User B needs <3 CPUs, 1 GB>. On a cluster with
9 CPUs, 18 GB: A's dominant resource is memory (4/18 = 22%), B's is CPU (3/9 = 33%).
DRF equalizes dominant shares.

**Relevance to Iris**: DRF is relevant because Iris has heterogeneous resources (CPU vs
GPU vs TPU). However, in practice most Iris jobs are either CPU-only or accelerator-heavy,
so a simpler value function may suffice.

### 2.4 Comparison Table

| Feature | K8s | SLURM | DRF/Mesos | Iris Today |
|---|---|---|---|---|
| Priority bands | ✅ PriorityClass | ✅ Partitions + QOS | ❌ | ❌ |
| Per-user fairness | ❌ | ✅ Fair-share tree | ✅ Max-min fairness | ❌ |
| Multi-resource | ❌ (single-resource) | ✅ TRES weights | ✅ (core design) | Partial (constraint-based) |
| Preemption | ✅ (evict lower priority) | ✅ (partition/QOS-based) | ❌ (Mesos doesn't preempt) | Partial (kill only) |
| Budget model | Hard priority | Soft (deprioritize) | Soft (equalize shares) | None |
| Opportunistic | ❌ | ✅ (backfill scheduler) | ❌ | ❌ |
| Hierarchy | ❌ | ✅ (account tree) | ❌ | ✅ (job tree) |

## 3. Design Considerations

### 3.1 Value Function

**Proposed**: `value = 1000 * accel_count + RAM_GB + 5 * CPU_cores`

This is reasonable. The 1000× multiplier for accelerators captures the reality that a
single TPU chip costs ~100× a CPU core. Refinements to consider:

- Weight by accelerator type (H100 >> A100 >> v5litepod)
- Use actual cost ratios from GCP pricing if available
- For DRF-style fairness, the value function becomes the "dominant resource" — the
  resource where the user consumes the largest fraction of total cluster capacity.

For v1, the proposed formula is good enough. Exact weights matter less than having *any*
value function.

### 3.2 Three-Band System

Map to K8s-style priority classes:

| Band | Priority | Preemptible by | Behavior |
|---|---|---|---|
| **Production** | 1000 | Nothing | Never preempted. For critical pipelines. |
| **Interactive** | 500 | Production | Normal scheduling. Default for `iris job run`. |
| **Batch** | 100 | Production, Interactive | Scheduled when idle. Never preempts others. |

Implementation sketch:
- Add `priority_band` field to `LaunchJobRequest` (enum: PRODUCTION/INTERACTIVE/BATCH).
- Store in tasks table (or derive from job).
- Sort pending tasks by `(band DESC, fairshare_score, root_submitted_ms, depth)`.
- Preemption loop: for each unschedulable task in Production/Interactive band, find
  running tasks in lower bands that free enough resources.

### 3.3 Opportunistic Scheduling ("No One Else Using Cluster")

SLURM's backfill scheduler is the model here. When cluster is underutilized:
- Batch jobs can run on any idle resources.
- They must be preemptible when higher-band work arrives.

Implementation: batch tasks get the normal scheduling path but with a `preemptible=true`
flag. The preemption loop evicts them first. No special "idle detection" needed — if
higher-priority work arrives, it preempts batch; if not, batch runs.

### 3.4 User Fairness ("1 Task Beats 100 Tasks")

Two approaches:

**A. Per-user task cap in scheduling cycle** (simplest):
`max_tasks_per_job_per_cycle` already exists (default 4). Add `max_tasks_per_user_per_cycle`.
User A with 1 task and User B with 100 tasks each get at most N tasks scheduled per cycle.

**B. Fair-share scoring** (SLURM-like):
Track cumulative resource-seconds per user. Compute fair-share factor = target_share / actual_usage.
Users who have consumed less than their share get higher priority. Requires:
- A `user_resource_usage` table tracking cumulative resource-seconds.
- Periodic updates when tasks finish (or from heartbeat resource snapshots).
- A decay half-life so recent usage matters more than ancient history.

**Recommendation**: Start with (A) for immediate impact. Add (B) later for true fairness.

### 3.5 Nested Task Priority Boost (Livelock Prevention)

The existing `priority_neg_depth` column already boosts deeper tasks. This prevents the
scenario where a parent job spawns children that never run because the parent's siblings
are always scheduled first.

For the budget system, the key rule is: **a child task's effective priority inherits the
*maximum* of its own band and its root job's band.** A production job's children should
never be starved by interactive jobs. The `root_submitted_ms` column already ties children
to their root's submission time.

### 3.6 Preemption as a Secondary Loop

**Proposed design**: A separate preemption pass runs after the normal scheduling pass,
on the same thread, at the same interval.

```
for each unscheduled task T (highest band first):
    if T.band <= BATCH: skip (batch never preempts)
    candidates = running tasks with band < T.band, sorted by (band ASC, value ASC)
    for each candidate C:
        if killing C frees enough resources for T on C's worker:
            kill C, assign T to C's worker
            break
```

**Mechanical requirements** (all exist today):
- Kill a running task: `kill_tasks_on_workers()` → heartbeat delivers kill → container stopped.
- Decommit resources: `_decommit_worker_resources()` releases CPU/memory/GPU/TPU.
- Re-enqueue preempted task: `WORKER_FAILED` → retry via preemption budget.

**Missing pieces**:
- Selection logic: which task to preempt (lowest band, then lowest value, then oldest).
- Graceful preemption signal: today's kill is immediate. A grace period (like K8s
  `terminationGracePeriodSeconds`) would allow checkpointing.
- Preemption cooldown: avoid thrashing by not preempting the same job repeatedly.

## 4. Recommendations

### Phase 1: Priority Bands (Low effort, high impact)

1. Add `priority_band` enum to proto and `LaunchJobRequest`.
2. Store band on tasks table as a sortable column.
3. Update pending-task ordering: `(band DESC, neg_depth, root_submitted_ms, submitted_at_ms)`.
4. Default: INTERACTIVE. CLI flag: `iris job run --priority batch|interactive|production`.

This alone gives operators control over what runs first, with zero fairness complexity.

### Phase 2: Per-User Scheduling Cap

1. Add `max_tasks_per_user_per_cycle` to `ControllerConfig`.
2. In `_run_scheduling_cycle`, track tasks-per-user and skip when cap is hit.
3. This ensures User A with 1 task and User B with 100 tasks get roughly equal
   scheduling throughput within the same band.

### Phase 3: Preemption Loop

1. After `find_assignments()`, run a preemption pass for unscheduled high-band tasks.
2. Select victims from lower bands using the value function.
3. Kill victim, decommit resources, assign the preemptor.
4. Add a `preemption_grace_period` config for checkpointing support.

### Phase 4: Fair-Share Budgets

1. Track per-user cumulative resource-value-seconds.
2. Compute fair-share factor with exponential decay.
3. Use fair-share as a secondary sort key within each band.
4. Dashboard: show per-user budget usage.

## 5. Open Questions

1. **Who sets priority bands?** Submitter (self-service) vs admin-only? SLURM uses QOS
   assigned by admins; K8s lets anyone reference a PriorityClass.
2. **Should production band have a per-user cap?** Prevents one user from locking the
   entire cluster at production priority.
3. **Graceful preemption signal**: Should Iris send SIGTERM with a grace period before
   SIGKILL? This requires worker-side support for grace periods.
4. **Fair-share target shares**: Equal per user? Configurable? SLURM allows hierarchical
   shares. For a small team, equal shares may be fine.
5. **Resource value weights**: Should the value function be configurable, or is
   `1000*accel + RAM_GB + 5*CPU` sufficient?
6. **Coscheduled job preemption**: Preempting one task of a coscheduled (gang) job
   requires preempting ALL tasks. This makes preemption much harder for multi-host TPU jobs.
7. **Reservation interaction**: How do priority bands interact with the existing
   reservation system? Should reserved capacity be exempt from preemption?
