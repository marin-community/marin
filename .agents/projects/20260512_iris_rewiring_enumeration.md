# Iris SA Core Rewiring — Enumeration Summary

**Date:** 2026-05-12
**Compiled from:** two sub-agent enumeration passes (store-method call sites + direct-SQL call sites).

This is the input to the 12-task migration plan (`M1-M12`) registered in the harness.

## 1. Store-method call sites (208 total)

| File | Count | Notes |
|---|---|---|
| `lib/iris/src/iris/cluster/controller/transitions.py` | 165 | Largest target. Jobs: 24, tasks: 29, workers: 21, attempts: 7, endpoints: 4, reservations: 2, etc. |
| `lib/iris/src/iris/cluster/controller/service.py` | 23 | RPC handlers |
| `lib/iris/src/iris/cluster/controller/controller.py` | 19 | Scheduler loop |
| `lib/iris/src/iris/cluster/controller/actor_proxy.py` | 1 | `endpoints.resolve` |

## 2. Store constructors / imports (18 sites)

- `stores.py:2110-2115` — `ControllerStore.__init__` builds 5 Store instances
- `controller.py:307, 1128` — direct instantiation of `TaskAttemptStore` and `ControllerStore`
- 7 test conftests + 9 `benchmark_controller.py` calls

## 3. Legacy types still imported (104 refs)

- `TransactionCursor`: 82 — transitions.py (37), stores.py (37), db.py (5), test_db.py (3)
- `QuerySnapshot`: 22 — db.py (8), service.py (5), transitions.py (3), stores.py (3), budget.py (2), schema.py (1)
- `Row` (db.py): a few in tests

## 4. Direct SQL call sites (60, all `?` placeholders)

| File | Count | Style |
|---|---|---|
| `service.py` | 24 | `q.raw()`, `q.fetchall()`, `q.execute_sql()`, `q.fetchone()` |
| `db.py` | 14 | `cur.execute()` + budget reads via `q.raw()` |
| `projections/endpoints.py` | 7 | `cur.execute()` + `q.fetchall()` |
| `controller.py` | 4 | `q.raw()`, `q.execute_sql()`, `q.fetchone()` |
| `auth.py` | 2 | `cur.execute()` |
| `query.py` | 1 | `q.execute_sql()` (user-provided SQL) |
| `projections/worker_attrs.py` | 1 | `q.raw()` |

Every one uses `?` placeholders. Must convert to SA Core expression language (or `:name` binds where SA Core builders don't fit).

## 5. Legacy `Projection` / `*_PROJECTION` references (15+ files)

- `JOB_DETAIL_PROJECTION` (6 files), `TASK_DETAIL_PROJECTION` (9 files), `JOB_SCHEDULING_PROJECTION` (3 files), plus `WORKER_ROW_PROJECTION`, `WORKER_DETAIL_PROJECTION`, `ATTEMPT_PROJECTION`, `ENDPOINT_PROJECTION`, `TASK_ROW_PROJECTION`, `API_KEY_PROJECTION`, `USER_BUDGET_PROJECTION`, `JOB_RESERVATION_PROJECTION`, `JOB_ROW_PROJECTION`.

All deleted in M8 once readers stop using them.

## 6. Decoder helpers in legacy `schema.py`

`decode_worker_id`, `decode_timestamp_ms`, `_decode_bool_int`, `_nullable`, `_decode_json_dict`, `_decode_json_list`, `proto_decoder`, `_identity`, `JOB_CONFIG_JOIN`.

External import: `lib/iris/src/iris/cluster/autoscaler/recovery.py:19` uses two of them. Must be rewired in M5/M8.

## 7. Row dataclasses (KEEP for now)

`JobRow`, `JobDetailRow`, `JobReservationRow`, `JobSchedulingRow`, `TaskRow`, `TaskDetailRow`, `WorkerRow`, `WorkerDetailRow`, `EndpointRow`, `AttemptRow`, `ApiKeyRow`, `UserBudgetRow`.

Some heavily used as named return types (`TaskDetailRow` 29 refs, `WorkerDetailRow` 15, `EndpointRow` 45). Plan: keep these in `schema.py` (or move to `controller/rows.py`); reads/* return SA `Row` mappings by default, but where a named return type is genuinely useful at an RPC boundary, the helper constructs the dataclass from the SA Row.

## 8. Vestigial parity tests (deleted in M10)

3073 LOC of `test_reads_*.py` + `test_writes_*.py` files that asserted "legacy `JobStore.get_detail` returns the same value as `reads.jobs.get_detail`." Once the legacy code is gone these tests are nonsense. Delete:

- `test_reads_budgets.py`, `test_reads_dashboard.py`, `test_reads_jobs.py`, `test_reads_reservations.py`, `test_reads_scheduler.py`, `test_reads_task_attempts.py`, `test_reads_tasks.py`, `test_reads_workers.py`
- `test_writes_jobs.py`, `test_writes_reservations.py`, `test_writes_task_attempts.py`, `test_writes_tasks.py`, `test_writes_workers.py`

KEEP `test_writes_to_check.py` (it's the `@writes_to` invariant test, not a parity test).

## 9. Migration task list (M1-M12)

| # | Task | Touches |
|---|---|---|
| M1 | Rewrite `reads/*` to SA Core `select(table.c.col)` | reads/*.py |
| M2 | Rewrite `writes/*` to SA Core `insert/update/delete` | writes/*.py |
| M3 | Rewrite `projections/*` SQL to SA Core | projections/*.py |
| M4 | Canonicalize `Tx`; delete legacy `db.py` machinery | db.py, db_v2.py |
| M5 | Rewire `controller/service/auth/budget/query/actor_proxy` call sites | 6 files |
| M6 | Rewire `transitions.py` call sites (165) | transitions.py |
| M7 | Delete `stores.py` + Store classes | stores.py |
| M8 | Delete legacy `schema.py` machinery | schema.py |
| M9 | Rename `schema_v2` → `schema`, `db_v2` → `db` | broad import sweep |
| M10 | Delete parity tests (3073 LOC) | tests/cluster/controller/test_*.py |
| M11 | Fix integration tests | test_transitions/test_direct_controller/etc |
| M12 | 100% SA audit + AGENTS.md + PR | grep sweep + doc |

## 10. Success criteria

- `grep -r "_v2"` across `lib/iris/` returns 0 hits (except historical design docs in `.agents/projects/`).
- `grep -r "TransactionCursor\|QuerySnapshot"` returns 0.
- `grep -r "JobStore\|TaskStore\|WorkerStore\|ReservationStore\|ControllerStore\|TaskAttemptStore"` returns 0.
- `grep -r "ProtoCache\|adhoc_projection\|class Projection"` in `schema.py` returns 0.
- All non-parity tests pass under `uv run pytest lib/iris/tests/`.
- `./infra/pre-commit.py --all-files` clean.
