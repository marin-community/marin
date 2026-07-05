# Iris: evaluate `max_task_failures` from derived counts; drop the reconcile in-flight `failure_count`

**Status:** proposed (weaver #395, issue #6963)
**Author:** agent session `evaluate-max-failures-separately`
**Scope:** `lib/iris` reconcile kernel — delete the last denormalized in-flight retry counter; evaluate the job-wide failure budget from derived attempt counts.

## 1. Problem

After #6946 / PR #6956, a task's failure and preemption counts are a pure function
of its `task_attempts` rows (`controller/attempt_counts.py`): there are no
denormalized `tasks.failure_count` / `tasks.preemption_count` columns and no proto
scalars. The reconcile kernel is already symmetric for **preemption** —
`resolve_task_failure_state` reads the committed-derived per-task count for the
local retry decision and carries nothing across the batch.

**Failure is the one exception.** `TaskRowDelta.failure_count`
(`reconcile/effects.py:54`) is a *prospective* per-task counter, stamped on every
terminal-failure task update and folded through `Overlay.merge_task`
(`overlay.py:241`), solely so `Overlay.job_basis` (`overlay.py:120`) can sum a
mid-batch `total_failures`. That sum is consumed at exactly one place — the budget
branch of `recompute_state`:

```python
elif basis.total_failures > max_task_failures:   # reconcile/job.py:37
    new_state = job_pb2.JOB_STATE_FAILED
```

It exists only because the budget is evaluated *mid-batch*, before the batch's
attempt rows commit — a naive committed-derived read would miss the in-flight
failures. Every other branch of `recompute_state` reads `task_state_counts` (the
overlay-aware task-state histogram), which needs no counter scratch.

## 2. The linchpin fact

**The snapshot loader already loads the committed-derived job-wide failure count.**

`_load_all_tasks_for_jobs` derives each task's `failure_count` from its attempts
(`loader.py:190,198` → `reads.attempt_counts_for_tasks`), and
`_bulk_load_job_state_basis` sums those per job into `JobStateBasis.total_failures`
(`loader.py:151-165`). So `snapshot.job_state_basis[job].total_failures` **is** the
committed-derived job-wide failure count as of snapshot load.

`Overlay.job_basis` then *discards* that loaded value and recomputes
`total_failures` from the scratch (committed per-task base `??` `delta.failure_count`).
The scratch's only job is to add *this batch's not-yet-committed failures* on top of
the committed base the loader already holds.

This reframes the change: we do not need a new projection read or a post-commit
sweep to obtain the committed-derived count — the loader already computes it. We
only need to decide how this batch's in-flight failures are folded in, and delete
the scratch.

## 3. What gets deleted

| Site | Today | After |
|---|---|---|
| `effects.py:54` | `TaskRowDelta.failure_count` field | removed |
| `overlay.py:241` | `merge_task` folds `failure_count` (last-non-null) | removed |
| `task.py:148-150,194` | `merge_task_termination(failure_count=)` param + stamp | removed |
| `task.py:545` | `timeout_one(... failure_count=row.failure_count + 1)` | removed |
| `task.py:507` | `apply_one_transition` stamps `failure_count=` onto the delta | removed (keep the local `+1` used by the retry gate) |
| `overlay.py:120-122` | `job_basis` sums `delta.failure_count ?? row.failure_count` | replaced by §4 |

**Newly-dead, delete too:** `ActiveTaskRow.failure_count` (`task_state.py:84`,
populated `reads.py:1239`, copied through `overlay.py:187`) is read by exactly one
site — `timeout_one`'s `row.failure_count + 1` (`task.py:545`). Once that `+1` goes,
the field is dead and should be removed (the codebase deletes dead fields). Its
sibling `ActiveTaskRow.preemption_count` stays (read by the preempt / worker-fail
paths).

**Kept, deliberately:**

- `TaskHistogramRow.failure_count` and the loader derivation (`loader.py:155,198`) —
  it is the committed-derived per-task base the budget still sums.
- The local `failure_count = task.failure_count + 1` in `apply_one_transition`
  (`task.py:408,452`) that gates the **per-task** retry decision
  (`failure_count <= task.max_retries_failure`). This reads the committed-derived
  count from the snapshot `TaskDetailRow` (`reads.py:1168`, derived) and applies a
  local `+1` for the current attempt — exactly mirroring preemption's
  `preemption_count + 1 <= max_preemptions`. It is not carried across the batch; it
  is only no longer *stamped* onto the effects delta.

## 4. Design options

### Option A — overlay-aware atomic derivation (recommended)

In `job_basis`, keep `total_failures` but derive it from the **loaded committed base
plus this batch's real attempt writes**, with no scratch field. `job_basis` stops
touching `row.failure_count` entirely — it reads the pre-summed loader value and adds
this batch's FAILED attempt deltas:

```python
total_failures = basis.total_failures          # committed base, summed by the loader
for (task_id, _), d in self._effects.attempts.items():
    if task_id.parent == job_id and d.state == job_pb2.TASK_STATE_FAILED:
        total_failures += 1
# (the existing all_tasks_by_job loop still builds task_state_counts + first_error,
#  but no longer reads row.failure_count)
```

The FAILED attempt deltas are the real writes already in `overlay._effects.attempts`
(state `TASK_STATE_FAILED`) — the same rows `commit.py` flushes to `task_attempts`.
So the total equals *"the committed-derived count projected forward by this batch's
own attempt writes"* — i.e. exactly what the next tick's loader would compute once
these attempts commit. (`TaskHistogramRow.failure_count` stays, but only the loader
reads it now, to compute `basis.total_failures`; the overlay no longer does.)

Properties:

- **Atomic and behavior-preserving.** The job fails on the *same* tick as the
  crossing failure, identical to today. `_recompute_and_finalize` →
  `_finalize_terminal_job` runs unchanged.
- **No new machinery** — no extra transaction, no post-commit sweep, no
  affected-job plumbing, no new verb.
- **No coverage gap** (see Option C for the gap this avoids).

Correctness of the equivalence (why existing tests pass unchanged): a task's
`current_attempt_id` is immutable within a batch and `merge_task_termination`'s
terminal-attempt guard drops any second terminal write to the same attempt, so a
task contributes **at most one** FAILED attempt delta per batch. Therefore
`count(FAILED attempt deltas)` equals today's "+1 per task that failed this batch",
and `snapshot_basis.total_failures` equals today's committed per-task base sum.
Option A computes the identical mid-batch total the scratch does today — with no
persisted or prospective counter.

Residual asymmetry vs preemption: `job_basis` still contains a failure-specific line
(count FAILED attempt deltas). But that is inherent — failure *has* a job-wide
budget and preemption does not — and it reuses the single failure predicate
(`state == TASK_STATE_FAILED`) over attempt writes that already exist for other
reasons. No scratch is carried; failure now carries no *more* than preemption.

**Precondition — the `timeout_one` edge (codex P2, must-fix).** There is one path
where today's unconditional scratch `+1` and Option A's "count the FAILED attempt
delta" can diverge: `timeout_one` passes `failure_count=row.failure_count + 1`
*unconditionally* (`task.py:545`), but `merge_task_termination` **skips the attempt
write when the current attempt is already terminal** (`task.py:174-185`). So if a
TIMEOUT decision ever targets a task whose current attempt is already terminal in the
overlay, the old code charges `+1` while Option A charges `0` (no FAILED delta is
written).

In production this edge is unreachable: `_cascade_timeouts` selects rows via
`active_row_from_snapshot` (active tasks only, `batches.py:254`), execution-timeout
scans only pick BUILDING/RUNNING rows (`reads.py:1675`), and RPC kicks reject
non-active tasks (`service.py:1988`) — so the current attempt is always non-terminal
and the FAILED delta is always written. Moreover, Option A's behavior on the edge is
arguably the *correct* one: if the current attempt already recorded a terminal
outcome, a timeout should not fabricate an extra application failure. Post-refactor,
`timeout_one` carries no `failure_count` at all — its charge is *entirely* the FAILED
attempt delta it writes, which is exactly the derived semantics.

Action before implementation: make the precondition explicit — either assert in
`finalize_tasks`/`timeout_one` that a TIMEOUT target's current attempt is non-terminal,
or add a test proving the edge is unreachable (§7c). Do not leave the equivalence
resting on an unstated invariant in a safety-critical kernel.

### Option B — post-commit sweep (the issue's recommendation)

Drop the budget branch from `recompute_state` entirely; also drop
`JobStateBasis.total_failures`, `TaskHistogramRow.failure_count`, and the loader
derivation. After the batch commits, sweep the jobs that had a terminal-failure
task update this batch, read the committed-derived per-job count via
`reads.attempt_counts_for_jobs(cur, jobs)`, and fail any non-terminal job over
budget.

- **Purest symmetry:** failure and preemption both "read committed-derived, carry
  nothing"; the budget leaves the recompute kernel.
- **Costs:** the job-fail cascade moves to a *separate transaction* (a real ordering
  change from today's atomic "fail tasks + fail job in one batch"); it needs
  affected-job-set plumbing, a fail-over-budget path that re-runs
  `_finalize_terminal_job` over a reloaded snapshot (or defers to the next tick's
  recompute), and its own tests — all in the safety-critical kernel.
- Note: `attempt_counts.py`'s own contract says reconcile derives counts via SQL in
  lockstep with its snapshot and **does not** consult `AttemptCountsProjection`
  (which is for the dashboard / list-jobs reads). So the sweep would read via
  `reads.attempt_counts_for_jobs(cur)`, not the projection the issue names.

### Option C — committed-only pass-through (rejected)

The minimal deletion: `job_basis.total_failures = snapshot_basis.total_failures`
(just stop overriding the loaded value); do not fold in this batch's failures.

- Simplest diff, but introduces a **one-tick delay** (the crossing batch's failures
  are not in the loaded base until the next tick) **and a coverage gap**: a job that
  crosses the budget on its last failure of a round and then goes idle-PENDING (e.g.
  no capacity to reschedule) is not "touched" again, so `recompute_state` never
  re-runs for it and it lingers RUNNING over budget until it is next scheduled.
  Rejected in favor of A, which has neither problem.

## 5. Recommendation

**Option A.** It deletes exactly the scratch the issue targets (the `TaskRowDelta`
field, its `merge_task` fold, the `merge_task_termination` param/stamp, the
`timeout_one` stamp, the `apply_one_transition` stamp, and the scratch-based
`job_basis` sum), while preserving atomic same-tick failure. That makes it the
lowest-risk change to a safety-critical kernel: it is behavior-preserving, so the
existing budget suite passes unchanged and the golden replay fixtures are expected to
stay byte-identical — the deleted counter was never persisted and the commit timing
does not move (verify by running the goldens post-implementation, §7).

The issue recommends Option B, but explicitly hedges — *"Recommend the post-commit
sweep unless the atomicity window proves to matter."* We argue the atomicity is a
virtue worth keeping: failing a crash-looping gang one tick earlier and in a single
transaction is strictly better for a crash-loop guard, and Option A removes the
scratch without moving the cascade out of the atomic batch or adding kernel surface.

**Open question for review:** is the architectural purity of Option B — budget fully
out of the recompute kernel, `JobStateBasis.total_failures` and
`TaskHistogramRow.failure_count` deleted, one uniform post-pass — worth the added
machinery and the batch-atomicity/ordering change? If the team wants the budget out
of `recompute_state` on principle, B is the target; otherwise A achieves the
deletion goal with less risk.

## 6. Correctness — the coscheduled-gang crash-loop

The subtle case the issue flags: the derived job-wide count must equal the old
running counter for a **coscheduled gang** that crash-loops across rounds.

Per crashed gang round, exactly one attempt is charged:

- The task that genuinely crashed gets a `TASK_STATE_FAILED` attempt → charges 1.
- Its siblings are moved by the peer cascade to `TASK_STATE_COSCHED_FAILED` — both
  `terminate_coscheduled_siblings` and `requeue_coscheduled_siblings`
  (`peers.py`) stamp the attempt `COSCHED_FAILED`, which is excluded from **both**
  the failure predicate (`== FAILED`) and the preemption predicate
  (`PREEMPTION_ATTEMPT_STATES`) → charges 0.

So a K-round crash-loop accrues exactly K FAILED attempts across the job, one per
round, regardless of which task crashes each round or whether any single task ever
exhausts its own per-task retry budget. Under Option A, `total_failures` =
`snapshot_basis.total_failures` (sum of prior rounds' committed FAILED attempts) +
this round's single FAILED delta = the correct cumulative K, and the budget branch
fires when K exceeds `max_task_failures`. This is where an off-by-one between "charge
per round" and "count FAILED attempts" would surface, so it gets a dedicated test
(§7b).

## 7. Tests

**Must pass unchanged (Option A is behavior-preserving).**

- **Unit — the budget decision** (drive `recompute_state` directly, in
  `test_transitions.py`): `test_recompute_fails_job_on_cumulative_failures_while_active`
  (`:4403`, ids `one-failure-retried`, `failures-spread-across-tasks` — its docstring
  already says it "Models a coscheduled gang mid-crash-loop"),
  `test_recompute_keeps_job_running_within_budget` (`:4431`), and the all-terminal
  contrast `test_recompute_fails_job_when_all_tasks_terminal_with_a_failure` (`:4373`).
  These use the `_recompute_snapshot` harness (`:4315`), which sets
  `JobStateBasis.total_failures = sum(failure_counts)` directly and applies no attempt
  deltas — so under Option A `job_basis` returns `basis.total_failures + 0`, identical
  to today. **Implementation note:** the `_basis` helper (`:4256`) hard-codes
  `total_failures=0`; keep `job_basis` reading `basis.total_failures` (not re-summing
  `row.failure_count`) so these harnesses stay authoritative, and audit that any test
  building a basis by hand sets `total_failures` consistently with its histogram.
- **E2E / harness budget tests** (`test_transitions.py`):
  `test_max_task_failures_tolerance` (`:850`, the canonical tolerate-then-trip),
  `test_failure_domain_kills_remaining_tasks` (`:641`),
  `test_batch_success_and_failure_is_order_independent` (`:672`),
  `test_preemption_does_not_count_toward_max_task_failures` (`:882`, negative control),
  `test_job_failure_threshold_applies` (`:4193`),
  `test_max_failures_kills_dispatch_tasks` (`:4158`, direct provider),
  `test_task_failure_with_retry_requeues` (`:245`).
- **Derivation** (`test_attempt_counts.py`): `test_counts_from_attempts` (`:71`) and
  `test_sql_exprs_match_pure` (`:76`) over the 11-case `_CASES` table — unaffected;
  the failure/preemption predicates do not change.

**Golden replay fixtures (`replay/golden/*.json`) — a load-bearing regression check.**
The dump reads only persisted SQLite tables; `TaskRowDelta.failure_count` is
never-flushed scratch and `JobStateBasis.total_failures` is unpersisted, so deleting
them leaves every golden byte-identical **provided no persisted value and no
`Timestamp.now()` call count changes**. Because Option A keeps the job-FAILED
transition on the same tick, the 15 existing goldens are *expected* to stay
byte-identical — strong evidence Option A is behavior-preserving. This is **not** a
design-time proof (codex P8): the implementation must not add a persisted write, a
log line, or an extra `Timestamp.now()` call, so **run the goldens after
implementation** and treat any diff as a behavior change to explain, not to blindly
regenerate. (Under Option B or C the job-FAILED tick moves; the goldens embed
absolute frozen-clock timestamps, so a one-tick shift cascades through the whole dump
and forces regeneration via `pytest --update-goldens`. That the goldens *must* move
under B/C but *should not* under A is itself a signal in A's favor.)

**New coverage — the gap.** The multi-round coscheduled crash-loop that trips the
cumulative budget exists today **only** as the synthetic-histogram unit test at
`test_transitions.py:4403`; nothing drives multiple failure rounds through the
reconcile kernel to the budget. Add:

- **(b1)** a new replay scenario `scenario_coscheduled_crash_loop_fails_on_budget`
  (+ golden) in `replay/scenarios.py`: a coscheduled gang, one task crashing per
  round with siblings bounced `COSCHED_FAILED`, no single task exhausting its
  per-task retry budget, looped until the cumulative FAILED count crosses
  `max_task_failures` → job `JOB_STATE_FAILED`. This is the first budget-trip golden
  and pins the derived count end-to-end. (Under Option A its golden is generated once
  and then frozen; the off-by-one in §6 would show up as a different failing tick.)
- **(b2)** optionally an E2E assertion alongside `test_max_task_failures_tolerance`
  that drives the gang rounds through the kernel and asserts the trip, for a readable
  companion to the golden.
- **(c) The `timeout_one` precondition** (codex P2). Assert or test that a TIMEOUT
  decision whose target's current attempt is already terminal charges nothing under
  Option A (and cannot arise in production). Either a unit test on
  `finalize_tasks([TIMEOUT])` over an already-terminalized attempt, or an assertion in
  the timeout path plus a test that the normal active-task timeout still charges
  exactly one (via its written FAILED attempt).

## 8. Definition of done

- `TaskRowDelta.failure_count` and all in-flight failure-count plumbing removed; the
  failure path carries nothing extra through a reconcile batch (symmetric with
  preemption — both write only real attempt deltas).
- The `max_task_failures` budget is evaluated from the loaded committed-derived base
  plus this batch's real FAILED attempt writes; `job_basis` no longer consults a
  prospective failure counter.
- Existing `max_task_failures` budget tests pass unchanged; golden replay fixtures
  unchanged; new gang crash-loop test added.

## 9. Risk

Localized to the one budget branch and the scratch plumbing. Option A is
behavior-preserving, so the blast radius is the derivation swap in `job_basis` and
the mechanical deletions; it is guarded by the existing budget suite (unchanged),
the golden fixtures (unchanged), and the new gang coverage.

Refs: #6946, PR #6956, `reconcile/job.py:37`, `reconcile/effects.py:54`
(`TaskRowDelta.failure_count`), `reconcile/overlay.py:120` (`job_basis`),
`reconcile/loader.py:151-165` (`_bulk_load_job_state_basis`),
`controller/attempt_counts.py`, `reads.attempt_counts_for_jobs`.

## 10. Codex peer review (2026-07-05)

Ran `codex exec` (read-only) against this doc and the real code with instructions to
verify or refute each claim and hunt for a reason Option B is *required*. Verdict:

> **Option A is sound and correctly recommended over Option B for the intended
> production paths. I found no correctness reason Option B is required.**

Confirmed by review, with file:line evidence: the linchpin (loader pre-sums
`total_failures`, overlay discards it); that only `state == TASK_STATE_FAILED`
attempts are charged (worker/preempt/cosched siblings write WORKER_FAILED / PREEMPTED
/ COSCHED_FAILED); the per-round gang charging (crashing task FAILED, siblings
COSCHED_FAILED); Option A's same-batch atomicity vs B/C's one-tick shift; Option C's
idle-PENDING gap (reject C); and **no missed runtime consumer** of the scratch or
`JobStateBasis.total_failures` — dashboard / RPC / federation all read attempt-derived
counts via projection/SQL (`service.py:643`, `federation_store.py:203`), not the
scratch.

Two items folded back into the doc:

1. **Must-fix — `timeout_one` precondition (P2).** `timeout_one` charges `+1`
   unconditionally, but `merge_task_termination` skips the attempt write on an
   already-terminal current attempt — so old-scratch and Option A diverge on that
   edge. Unreachable in production (timeouts select active rows only) and Option A's
   behavior is arguably the more-correct one, but the invariant must be made explicit
   by an assertion or a test (§4 Option A precondition, §7c). This is the one
   must-address item before implementation.
2. **Soften the goldens claim (P8).** Byte-identity is *expected* but not
   design-time-provable; run the replay goldens post-implementation rather than
   asserting identity from review alone (§7 updated).

The full review transcript is in the session scratchpad
(`codex_review_out.md`).
