# Sub-doc: Rollout & Compatibility Matrix

Companion to `spec.md` §7. Draft 2.

## Changes from Draft 1

- **4 phases not 6.** Phase A (pure-compute) is independent of everything and ships first. Phase B is the Reconcile RPC keyed by composite `(task_id, attempt_id)` — the real wire-level win, doesn't need UIDs. Phase C adds UIDs. Phase D is cleanup (delete legacy, swap PK).
- **Phase C decouples "code routes by UID" from "DDL swaps PK."** The reviewer's concern: Phase 5 in v1 combined a dual-pathed code change with an irreversible DDL swap, making rollback wishful. v2 splits them: Phase C ships dual-routing code; Phase D ships the PK swap only after C is stable for a release.

## Compatibility matrix

C = controller, W = worker. Phase numbers refer to `spec.md` §7.

| Scenario | C Phase | W Phase | Wire | Notes |
|---|---|---|---|---|
| **Pre-rollout** | 0 | 0 | StartTasks / PollTasks / UpdateTaskStatus / GetTaskAttemptInfo | Current state on `main` |
| **Phase A deployed** | A | 0 | unchanged | Controller uses `reconcile_worker` internally; wire identical |
| **Phase B deployed, old W** | B | 0 | unchanged | C dispatches via compat shim to legacy RPCs |
| **Phase B deployed, new W** | B | B | Reconcile RPC | Controlled by `IRIS_RECONCILE_RPC_ENABLED` flag |
| **Phase C deployed (additive UID)** | C | C | Reconcile with `attempt_uid` in addition to legacy fields | Routing still by composite under feature flag `IRIS_UID_PRIMARY_KEY=false` |
| **Phase C UID-routing on** | C | C | same wire | Flag flipped; same wire shape, different code path |
| **Phase D deployed** | D | D | Reconcile only, UID-keyed | Legacy RPCs removed; PK swapped |

## Per-phase rollback story

### Phase A — Pure-compute split

Roll back: revert the PR. No schema or wire change. Worst case: revert one PR, deploy, done.

### Phase B — Reconcile RPC keyed by composite

Roll back: flip `IRIS_RECONCILE_RPC_ENABLED=false` (controller-side flag). Controller dispatches via compat shim immediately. Workers on Phase B continue to support Reconcile but stop receiving it; they fall back to legacy handlers.

If the bug is in the compat shim itself: revert the deploy. Controller goes back to today's `_reconcile_worker_batch` calling old wire. Worker code on Phase B is dual-pathed (Reconcile + legacy handlers); both work.

### Phase C — Additive UIDs

Roll back: flip `IRIS_UID_PRIMARY_KEY=false`. Routing reverts to composite keys. The UID columns stay in the DB (harmless). New writes still populate them (no harm).

If we want to remove the UID columns entirely: separate cleanup migration. We won't normally do this; the columns are cheap and the secondary indexes are fast.

### Phase D — Cleanup (PK swap + legacy RPC removal)

Roll back: ugly. The PK swap migration (`0028_uid_primary_keys.py`) is technically reversible (see `sub/migration.md` for the symmetric rollback), but if code is already deployed that assumes UID-primary routing, application-level rollback requires reverting deploys. The honest answer:

> **Phase D rollback is forward-fix, not revert.**

Gate: Phase D ships only after Phase C has been on production for ≥1 release cycle with no UID-routing bugs. Forward-fix means: if a Phase-D bug surfaces, write a new fix PR rather than trying to flip a switch. This is the operational tradeoff of irreversible cleanup migrations.

If pressure to ship Phase D is high but stability isn't proven, **split it further**: ship the legacy-RPC removal first (one release cycle), then the PK swap (next release). Two micro-phases instead of one.

## Feature flags

Three flags, controller-side:

1. **`IRIS_RECONCILE_RPC_ENABLED`** (Phase B+): when true, controller dispatches Reconcile RPC to all workers (fleet-wide flip). Default false in Phase B (canary), true in Phase B steady state.

2. **`IRIS_UID_PRIMARY_KEY`** (Phase C+): when false, code routes by composite `(task_id, attempt_id)` even though UID columns exist. When true, routing is by UID. Default false in Phase C (canary), true in Phase C steady state.

3. **`IRIS_RECONCILE_INTERVAL_SECONDS`** (all phases): the controller's tick interval. Default 1 second — matches today's `poll_interval` (`controller.py:988`). Operators can lower during incident response, down to ~200 ms (below that, batched-apply transactions begin to overlap). *No wake event* — this is the only timing primitive.

No worker-side flags. Workers support both wires (Reconcile + legacy); the controller-side flag drives the fleet-wide wire choice.

## Capability negotiation

No negotiation; controlled by `IRIS_RECONCILE_RPC_ENABLED`.

## Failure-mode playbook

| Symptom | Likely cause | Action |
|---|---|---|
| Controller restart → all workers reset → tasks killed | `_heartbeat_deadline` fires during restart | Out-of-scope here; tune `heartbeat_timeout` higher or work on fail-open (`spec.md` §5.10) |
| Worker reports MISSING for many uids | Spec cache lost (worker restart) or controller dispatched without a prior ASSIGNED tick | Attempts fail forward as `worker_lost_spec`; scheduler reissues. If pattern is broad, check `controller.py` for a dispatch bug omitting spec on ASSIGNED |
| UID collision | `secrets.token_hex(8)` (2^64 keys) | Realistic at fleet scale only after billions of attempts. INSERT failure surfaces as ConnectError; regenerate on collision |
| Phase B rollout: some workers killed wrong tasks | Compat shim bug | Roll back via `IRIS_RECONCILE_RPC_ENABLED=false`; controller emits legacy RPCs; tasks reconverge on next tick |
| Phase C: UID-keyed lookup misses, composite hits | Race in resolution helper | Use composite fallback path; investigate; the dual-routing makes this self-healing |
| Phase C: flapping worker saturates worker-initiated Reconcile | Misbehaving worker or controller load | Token-bucket in `ControllerService.reconcile` handler rejects with `RESOURCE_EXHAUSTED`; worker backs off |
| Operator wants to force-reconcile one worker | Operations | Add `iris admin reconcile-worker <id>` CLI: invokes `reconcile_worker(inputs)` for that worker and dispatches immediately, out-of-band of the timer loop |
| Operator wants to see what controller would do without actually reconciling | Debugging | Add `iris admin reconcile-dry-run <worker_id>` CLI: runs `reconcile_worker(inputs)`, prints output, doesn't send |

## Canary plans

### Phase B canary

1. Ship Phase B binary to controller with `IRIS_RECONCILE_RPC_ENABLED=false`.
2. Pick one worker (`worker-canary-1`) — set per-worker override `IRIS_RECONCILE_RPC_FOR_WORKERS=worker-canary-1`. Watch for 24 h.
   - Task throughput on canary worker: should match neighbors
   - Reconcile RPC success rate: ~100%
   - Spec cache hit rate (steady state): ~100% after first round
   - Killed-task audit log: zero unexpected kills
3. Extend to 10% of fleet. Watch for 48 h.
4. Flip global flag to true. All workers receive Reconcile RPC dispatches from the controller.

### Phase C canary

1. Ship Phase C with `IRIS_UID_PRIMARY_KEY=false`. UIDs populate but don't route.
2. Verify UID columns populated correctly via SQL audit. No code change in critical path yet.
3. Set per-worker override `IRIS_UID_PRIMARY_KEY_FOR_WORKERS=worker-canary-1`. Watch for 24 h.
   - Container adoption on worker restart: must succeed
   - Log key resolution: must return correct attempts
4. Extend to 10%. Watch 48 h.
5. Global flip.

### Phase D canary

Phase D is the irreversible one. **No partial rollout**. Pre-canary checklist:

- Phase C has been on prod for ≥1 release cycle with no UID-routing bugs (audit `worker.audit.log` for fallback-path usage).
- All workers in production have shipped binaries ≥ Phase C for ≥1 release cycle. Verified via the `workers.client_revision_date` audit.
- Migration `0028_uid_primary_keys.py` tested on prod snapshot. Backfill performance acceptable.

If all gates pass: deploy controller during a maintenance window. Run migration. Restart controller. Watch.

## Operator escape hatches

Add to `lib/iris/src/iris/cli/`:

- `iris admin reconcile-worker <worker_id>` — invokes `reconcile_worker(inputs)` for one worker and dispatches the RPC immediately, out-of-band of the timer loop. Logs an audit event.
- `iris admin reconcile-dry-run <worker_id>` — runs the pure function, prints output as JSON, doesn't send RPC.
- `iris admin spec-cache-status <worker_id>` — RPC to the worker that returns the uids currently in its `SpecCache`. Useful for diagnosing "did the worker actually receive the spec on the ASSIGNED dispatch?"
- `iris admin attempt-trace <attempt_uid>` — full audit trail for one attempt: DB rows, Reconcile RPCs sent/received, state transitions. The single most valuable admin tool.

These pay for themselves the first time a Phase B+ issue surfaces in production.

## What v1 got wrong that this fixes

- **v1 had 6 phases, 8 PRs**: v2 has 4 phases, ~6 PRs. The win comes from decoupling UID introduction from protocol introduction.
- **v1 Phase 5 was un-rolling-back**: v2 Phase C splits routing-flip from DDL-swap. Code rollback is real for Phase C; only Phase D is "forward-fix only" and is gated harder.
- **v1 hand-waved the canary plan**: v2 has explicit canary plans per phase with measurable gates.
