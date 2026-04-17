# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for TaskStore.query() + TaskFilter.

Exercises the filter combinations actually produced by transitions.py after the
store-layer consolidation: task_ids, job_ids, worker_id / worker_is_null,
states, limit, and the with_job / with_job_config join variants.
"""

from __future__ import annotations

import pytest
from iris.cluster.controller.schema import ACTIVE_TASK_STATES, EXECUTING_TASK_STATES, TaskDetailRow
from iris.cluster.controller.store import TaskFilter, TaskProjection
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2

from .conftest import (
    ControllerTestHarness,
    make_job_request,
    submit_job,
)

# --- Dataclass validation ---------------------------------------------------


def test_task_filter_rejects_worker_id_plus_is_null() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        TaskFilter(worker_id=WorkerId("w1"), worker_is_null=True)


# --- Empty-list short-circuits ---------------------------------------------


def test_query_empty_task_ids_short_circuits(state) -> None:
    """Empty task_ids tuple should return [] without executing SQL."""
    with state._stores.transact() as ctx:
        assert ctx.tasks.query(ctx.cur, TaskFilter(task_ids=())) == []


def test_query_empty_job_ids_short_circuits(state) -> None:
    with state._stores.transact() as ctx:
        assert ctx.tasks.query(ctx.cur, TaskFilter(job_ids=())) == []


# --- No-join path returns TaskDetailRow ------------------------------------


def test_query_no_filter_returns_task_detail_rows(state) -> None:
    tasks = submit_job(state, "j", make_job_request("j", replicas=3))
    assert len(tasks) == 3

    with state._stores.transact() as ctx:
        rows = ctx.tasks.query(ctx.cur, TaskFilter())

    assert len(rows) == 3
    assert all(isinstance(r, TaskDetailRow) for r in rows)
    assert {r.task_id for r in rows} == {t.task_id for t in tasks}


def test_query_by_task_ids_returns_subset(state) -> None:
    tasks = submit_job(state, "j", make_job_request("j", replicas=3))
    wanted = (tasks[0].task_id.to_wire(), tasks[2].task_id.to_wire())

    with state._stores.transact() as ctx:
        rows = ctx.tasks.query(ctx.cur, TaskFilter(task_ids=wanted))

    assert {r.task_id.to_wire() for r in rows} == set(wanted)


def test_query_by_job_ids(state) -> None:
    ja = submit_job(state, "ja", make_job_request("ja", replicas=2))
    jb = submit_job(state, "jb", make_job_request("jb", replicas=1))
    _ = jb  # silence unused warning — we filter to ja only

    with state._stores.transact() as ctx:
        rows = ctx.tasks.query(
            ctx.cur,
            TaskFilter(job_ids=(ja[0].job_id.to_wire(),)),
        )

    assert {r.task_id for r in rows} == {t.task_id for t in ja}


def test_query_limit_truncates(state) -> None:
    submit_job(state, "j", make_job_request("j", replicas=5))

    with state._stores.transact() as ctx:
        rows = ctx.tasks.query(ctx.cur, TaskFilter(limit=2))

    assert len(rows) == 2


# --- States filter ---------------------------------------------------------


def test_query_states_filter(state) -> None:
    """Filter tasks to only the PENDING state — all tasks start PENDING."""
    submit_job(state, "j", make_job_request("j", replicas=2))

    with state._stores.transact() as ctx:
        pending = ctx.tasks.query(
            ctx.cur,
            TaskFilter(states=frozenset({job_pb2.TASK_STATE_PENDING})),
        )
        active = ctx.tasks.query(
            ctx.cur,
            TaskFilter(states=ACTIVE_TASK_STATES),
        )

    assert len(pending) == 2
    assert active == []


# --- worker_id / worker_is_null --------------------------------------------


def test_query_worker_id_and_worker_is_null(state) -> None:
    """After dispatching one task, that task has current_worker_id set; the
    remainder are still NULL."""
    harness = ControllerTestHarness(state)
    wid = harness.add_worker("w1", cpu=10)
    tasks = harness.submit("j", replicas=3)
    harness.dispatch(tasks[0], wid)

    with state._stores.transact() as ctx:
        on_worker = ctx.tasks.query(ctx.cur, TaskFilter(worker_id=wid))
        unassigned = ctx.tasks.query(ctx.cur, TaskFilter(worker_is_null=True))

    assert {r.task_id for r in on_worker} == {tasks[0].task_id}
    assert {r.task_id for r in unassigned} == {tasks[1].task_id, tasks[2].task_id}


def test_query_worker_and_state_combination(state) -> None:
    """AND of worker_id + EXECUTING_TASK_STATES matches the migrated
    get_active_with_resources / cancel_tasks_for_timeout call path."""
    harness = ControllerTestHarness(state)
    wid = harness.add_worker("w1", cpu=10)
    tasks = harness.submit("j", replicas=2)
    harness.dispatch(tasks[0], wid)
    # tasks[0] is now RUNNING after dispatch; tasks[1] is still PENDING.

    with state._stores.transact() as ctx:
        rows = ctx.tasks.query(
            ctx.cur,
            TaskFilter(worker_id=wid, states=EXECUTING_TASK_STATES),
        )

    assert {r.task_id for r in rows} == {tasks[0].task_id}


# --- Joined variants ----------------------------------------------------------


def test_query_with_job_returns_typed_rows(state) -> None:
    """query WITH_JOB returns TaskDetailRow with is_reservation_holder/num_tasks populated."""
    submit_job(state, "j", make_job_request("j", replicas=1))

    with state._stores.transact() as ctx:
        rows = ctx.tasks.query(ctx.cur, TaskFilter(), projection=TaskProjection.WITH_JOB)

    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, TaskDetailRow)
    assert isinstance(row.is_reservation_holder, bool)
    assert isinstance(row.num_tasks, int)
    assert row.current_worker_id is None  # task not yet dispatched
    # Resource fields are not populated at this projection level.
    assert row.resources is None


def test_query_with_job_config_exposes_resource_columns(state) -> None:
    """query WITH_JOB_CONFIG returns TaskDetailRow with resource columns populated."""
    submit_job(state, "j", make_job_request("j", cpu=2, memory_bytes=2 * 1024**3))

    with state._stores.transact() as ctx:
        rows = ctx.tasks.query(ctx.cur, TaskFilter(), projection=TaskProjection.WITH_JOB_CONFIG)

    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, TaskDetailRow)
    assert row.resources is not None
    assert row.resources.cpu_millicores == 2000
    assert row.resources.memory_bytes == 2 * 1024**3
    assert isinstance(row.resources.disk_bytes, int)
    assert row.timeout_ms is None  # not set in make_job_request default


# --- Chunking -------------------------------------------------------------


def test_query_chunking_across_id_in_cap(state) -> None:
    """Supplying more task_ids than the SQLite host-param chunk size still
    returns the full matching set — chunk results are concatenated."""
    # submit 3 real tasks; pad task_ids with 950 bogus wires to force two chunks.
    # The chunker splits at 900 per batch (safely under SQLite's 999 cap).
    tasks = submit_job(state, "j", make_job_request("j", replicas=3))
    real = [t.task_id.to_wire() for t in tasks]
    padding = tuple(f"/bogus/task-{i}" for i in range(950))
    all_ids = tuple(real) + padding

    with state._stores.transact() as ctx:
        rows = ctx.tasks.query(ctx.cur, TaskFilter(task_ids=all_ids))

    assert {r.task_id.to_wire() for r in rows} == set(real)


def test_query_chunking_respects_limit(state) -> None:
    """When ``limit`` is set, the chunked loop stops once the limit is reached."""
    tasks = submit_job(state, "j", make_job_request("j", replicas=5))
    ids = tuple(t.task_id.to_wire() for t in tasks) + tuple(f"/bogus/{i}" for i in range(1000))

    with state._stores.transact() as ctx:
        rows = ctx.tasks.query(ctx.cur, TaskFilter(task_ids=ids, limit=2))

    assert len(rows) == 2


# --- Ordering -------------------------------------------------------------


def test_query_orders_by_task_id_ascending(state) -> None:
    """ORDER BY t.task_id ASC is stable across call sites that diff results."""
    tasks = submit_job(state, "j", make_job_request("j", replicas=4))
    expected_order = sorted(t.task_id.to_wire() for t in tasks)

    with state._stores.transact() as ctx:
        rows = ctx.tasks.query(ctx.cur, TaskFilter())

    assert [r.task_id.to_wire() for r in rows] == expected_order


# --- Parity with JobName-typed row field -----------------------------------


def test_query_rows_decode_job_name_fields(state) -> None:
    """Non-joined rows go through TASK_DETAIL_PROJECTION.decode → typed
    fields like ``job_id`` come back as JobName, not str."""
    submit_job(state, "j", make_job_request("j", replicas=1))

    with state._stores.transact() as ctx:
        [row] = ctx.tasks.query(ctx.cur, TaskFilter())

    assert isinstance(row.job_id, JobName)
    assert isinstance(row.task_id, JobName)
