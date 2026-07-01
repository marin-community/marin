# Design: commit-ownership pivot (folded PR-5 + PR-6)

Part of the Iris multi-backend contract (`.agents/projects/iris_backend_contract/`). Turns the
controller into a DAG-fold + cascade coordinator over backends that author their own workers'
direct execution transitions. Lands two seams as one unit of work: the schedule/assign seam
(placement address + backend-owned validation) and the reconcile fold seam (hoist the job-DAG
cascade out of the per-backend kernel into the controller). Assumes P1/P3/P4 merged, P2 (#6792)
landing. Byte-behavior-preserving at N=1.

## Motivation

Today each backend's `reconcile()` runs the FULL reconcile kernel over a snapshot that
(deliberately) includes descendant jobs regardless of backend, and returns a `ControllerEffects`
mixing (a) its own workers' direct task transitions and (b) job-DAG recompute + cross-job
cascade kills. The controller commits every backend's effects in one transaction. This "works"
for multi-backend only by luck: two backends fold overlapping descendant subtrees in isolation
and rely on `merge_cascade_kill` idempotency + a single commit. It is not authoritative (no one
backend sees the union) and does not survive the per-backend-commit / separate-DB-file split
that remote requires. The pivot makes the DAG fold authoritative and controller-owned, and makes
each backend the author of only its own execution state — the shape that becomes remote-safe.

## The two seams

### Seam A — schedule/assign (was PR-5 + the validation fork, decision b)

Problem: at assign-commit the controller re-reads the `workers` table for the address
(`reads.bulk_get_worker_addresses`, `ops/task.py:84`) and rechecks liveness
(`health.all()`, `ops/task.py:80-93`) — a controller read of backend-owned state that becomes an
RPC for remote.

Design:
- **Author the address at schedule time.** Add `address: str` to `Assignment` (`ops/task.py:42`).
  The address must survive the scheduler: `reads.WorkerSnapshot` currently drops it
  (`reads.py:1335`, `scheduler.py:61`), so carry it through the scheduler input→output (or an
  opaque placement token that resolves to worker_id+address). The backend owns the worker table,
  so it authors the placement's address.
- **Validate in the backend, at commit, against the backend's post-reconcile liveness.** Because
  the intra-tick race is real (schedule → reconcile-marks-worker-bad → commit), validation MUST
  be post-reconcile. Realize it inside the backend's placement commit (Seam B makes the backend
  the placement author), validating the exact `worker_id + address + incarnation` it stamps,
  transactionally protected from concurrent worker removal (the `workers` FK, `schema.py:348`,
  still enforces in-process). The controller trusts the returned surviving placements — no
  controller-side liveness recheck, no worker-table read. In-process this is a local store call;
  for remote it stays inside the backend, so no commit-path RPC is introduced.
- **Incarnation token (codex Finding 3).** `worker_id + address` does NOT catch same-id/same-address
  reuse — registration upserts by `worker_id` (`writes.py:527`) with no generation column
  (`schema.py:426`), and recycled-IP worker reuse is a documented hazard in this system. Carry a
  worker incarnation marker in the placement (the worker's `registered_at_ms`, or a new generation
  counter bumped on (re)registration) and validate it unchanged at commit, so a placement authored
  against a worker that has since re-registered is dropped. Prefer reusing `registered_at_ms` if it
  is bumped on every (re)register; add a generation column only if it is not.

Result: the controller's last commit-time read of the `workers` table is deleted.

### Seam B — reconcile fold (was PR-6 core)

Split the reconcile kernel's two passes across the backend boundary:

- **Backend keeps the APPLY pass.** `backend.reconcile()` runs only: direct task/attempt
  transitions for its own workers + the intra-job peer cascade (`_cascade_to_peers`,
  `batches.py:115-123`; strictly intra-job via `find_coscheduled_siblings`, `peers.py:18-32`, and
  a job's tasks all share one backend via `stamp_backend`, so peer cascades never cross backends).
  The apply pass legitimately reads backend-owned worker state (e.g. `active_workers` gating,
  `batches.py:414`), which is why it stays backend-side.
- **The apply-pass result carries fold inputs, not just row deltas (codex Finding 1).** The
  recompute/cascade pass consumes `ReconcileState.touched` and `pending_child_cascades`
  (`batches.py:320,347,365`) — the latter records a deferred child cascade when a transition rolls
  a parent task back to PENDING. Merged `TaskRowDelta`s alone lose that. So the backend returns a
  `DirectTransitionResult`: the row deltas (tasks/attempts) + health + `touched_jobs` +
  `pending_child_cascades`. (Equivalently, it could return the raw `TaskUpdate`s and let the
  controller run the apply pass — but the apply pass needs backend-owned worker liveness, so it
  stays backend-side and exports these two structures instead.)
- **Controller owns the RECOMPUTE + CASCADE pass.** After collecting every backend's
  `DirectTransitionResult`, the controller threads the row deltas into one `Overlay` and, seeded by
  the union of `touched_jobs` + `pending_child_cascades`, runs `recompute → finalize →
  child-cascade` (`batches.py:339-357`, `_finalize_terminal_job` `103-112`, `_cascade_to_children`
  `82-100`) as a single job-DAG fold over the UNION. This is the authoritative cascade. It reads
  ONLY controller-owned DB state — job tree (`parent_job_id`), task liveness (tasks table), job
  config (policy/num_tasks) — and NEVER backend execution/worker state (verified: `recompute_state`
  `job.py:24`, finalize `batches.py:103`, child cascade `batches.py:82`). Precedent: `ops/job.py::
  cancel` (`284-317`) already performs exactly this controller-side, cross-backend, subtree kill in
  one transaction using the same primitives.
- **Cross-backend cascade kills need no RPC and no explicit push.** The controller writes
  descendant terminal state (`TaskRowDelta(KILLED)` + cascade `JobRowDelta`) directly; each owning
  backend's next reconcile diffs desired-vs-actual and tears down the execution (k8s `sync` deletes
  pods not in the desired set, `k8s/tasks.py:1531-1546`; worker-daemon plans a `stop` for
  terminal-but-worker-bound attempts). Optionally group descendant deltas by `tasks.backend_id`
  to push explicitly — clean at job granularity because `stamp_backend` pins a job's tasks to one
  backend.

### Commit model + ordering + atomicity (in-process)

One tick transaction, ordering + snapshot timing preserved:
1. schedule decisions authored (unchanged), from pre-schedule snapshots;
2. each backend runs its apply pass over its pre-schedule reconcile snapshot → returns its
   `DirectTransitionResult` (+ validated placements);
3. controller opens the single tick `transaction` (`controller.py:1115`) and, in order: applies
   schedule placements; runs the DAG fold over one `Overlay` built ONLY from the backends'
   apply-pass results (row deltas + touched_jobs + pending_child_cascades) — NOT reloaded from DB
   after `assign`; then flushes the combined effects via `commit_effects` — direct transitions +
   cascade kills + job-DAG deltas — in that one txn.
**Snapshot-timing pin (codex Finding 2).** The fold must operate on the reconcile-snapshot-derived
overlay, matching today's timing where reconcile is authored from a pre-schedule snapshot
(`controller.py:804`, `ops/worker.py:274`). It must NOT fold against a DB state that already
reflects this tick's `assign` writes, or recompute would see same-tick `ASSIGNED` rows today's
reconcile snapshot cannot — breaking N=1 parity. Shared-overlay (not reload) gives this for free.
The DAG fold still runs AFTER collecting all apply-pass results and BEFORE the single commit, so
reconcile effects never become visible before schedule/autoscale (autoscale reads task status,
`rpc/backend.py:405`), and no reader observes a partial fold (`GetJobStatus`, `service.py:1465`).
**Atomicity is NOT relaxed in-process.** The per-backend-commit / separate-file split — and the
reader-consistency question it raises — is deferred to PR-9 (remote), where separate DB files make
it unavoidable and a tick/version marker (or explicit stale-read acceptance) is designed then.

### DAG summary shape

The direct-vs-DAG split of `ControllerEffects` (`effects.py:118-143`):
- Direct (backend `DirectTransitionResult`): `tasks`, `attempts`, `endpoint_deletions`, `health`,
  PLUS the fold-seed metadata `touched_jobs` + `pending_child_cascades` (Finding 1).
- DAG fold (controller): `jobs` deltas — `JobRowDelta` semantics (started/finished/error +
  `is_cascade_kill` guard, `effects.py:63`) + touched/cascaded job ids + log events. **No budget
  deltas** — spend is derived from active tasks (`budget.py:43`), not committed here.

The conceptual split is who AUTHORS the DAG fold: the controller runs recompute/finalize/cascade
over the union, not the backend. The backend authors only the apply-pass direct transitions and
exports the fold-seed metadata.

## Endpoints (kept shared through this work)

`commit_effects` deletes endpoints by task (`commit.py:223`). Keep threading the shared
`EndpointsProjection` facade the backend already holds; endpoint state still crosses via the shared
handle until the endpoint per-backend split (PR-8). Do not attempt endpoint relocation here.

## Out of scope (explicit)
- Autoscaler provisioning-persist single-writer → its own small PR-6.5 (after this ordering change).
- Published `BackendStatus` read-side → PR-7.
- Endpoints per-backend + router → PR-8.
- Remote transport + separate-DB-file + reader-atomicity marker → PR-9.
- **Subtree backend-pinning** (child job inherits root's backend) is NOT required by this design —
  the controller-side fold handles cross-backend cascades correctly regardless. It is a possible
  future simplification/locality feature (and matches the original "root job → same backend"
  intent) but is a separate routing decision, called out below.

## Staging — ONE PR, two commits (codex Q1)
Land together (address plumbing and validation/fold ownership touch the same commit path), but keep
the reviewer-visible split:
- **Commit 6a:** introduce the controller-side DAG fold — move recompute/finalize/child-cascade out
  of the per-backend kernel into a controller pass over the union of backends' `DirectTransitionResult`s
  (row deltas + touched_jobs + pending_child_cascades); backends return direct transitions only. Land
  Seam A's `Assignment.address` plumbing here (so the controller/backends stop reading the worker
  table for the address). Shared-overlay, single tick txn, reconcile-snapshot timing.
- **Commit 6b:** move placement validation into the backend (decision b, with the incarnation token)
  and delete the controller's liveness recheck + `bulk_get_worker_addresses`.

## Test plan
- N=1 parity across the existing reconcile/timeout/kick/preemption/cancel suites (the fold moved,
  behavior identical for a single backend).
- Cross-backend cascade: parent job on backend A goes SUCCEEDED/preempted-terminate-children →
  descendant job's tasks on backend B are KILLED by the controller fold and torn down by B's next
  reconcile. Assert the kill is authored once, in one txn, and B's execution stops.
- Assign race: schedule places on worker W; reconcile (same tick) reaps W → placement dropped at
  commit by the backend's validation, no `workers`-FK error, correct address stamped otherwise.
- Reader-consistency: `GetJobStatus` never observes task counts inconsistent with job state within
  a tick (guaranteed by the single in-process txn).

## Design decisions (resolved by codex review)
1. **6a/6b:** one PR, two commits (above).
2. **Overlay threading:** shared-overlay, single-commit — do NOT reload after `assign` (that makes
   direct transitions visible before the cascade and gives up in-process atomicity). Carry the
   fold-seed metadata (touched_jobs + pending_child_cascades) so the fold is faithful.
3. **Placement validation:** add a worker incarnation token (Seam A) — `worker_id + address` alone
   does not catch same-id reuse, and the code does not prove reuse is impossible.
4. **Subtree backend-pinning:** do NOT add it here. Per-job routing + the controller fold is the more
   general model and is correct without pinning; pinning can be a later routing policy, not a
   correctness dependency. (Noted: the code does not currently enforce the "root job → same backend"
   intent from the project goal — that is a separate future routing decision.)
