# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for ``reads/dashboard.py`` (Stage 10 of the SA Core migration).

Each test exercises one of the legacy dashboard composite reads
(``service._query_jobs``, ``service._task_summaries_for_jobs``,
``service._parent_ids_with_children``) against its SA Core port in
:mod:`iris.cluster.controller.reads.dashboard` and asserts the two paths
return equal results against the same DB state. For ``list_jobs`` we
cover the representative dashboard query shapes called out in the Stage
10 plan: default page (DATE/DESC), sort by name, scope=CHILDREN, the
failures aggregate join, and a name-filter.
"""

from __future__ import annotations

from iris.cluster.controller import db_v2
from iris.cluster.controller.reads import dashboard as reads_dashboard
from iris.cluster.controller.service import (
    USER_JOB_STATES,
    _parent_ids_with_children,
    _query_jobs,
    _task_summaries_for_jobs,
)
from iris.cluster.types import JobName
from iris.rpc import controller_pb2

from .conftest import make_job_request, submit_job


def _submit(state, name: str, **kwargs):
    return submit_job(state, name, make_job_request(name, **kwargs))


def _query(**fields) -> controller_pb2.Controller.JobQuery:
    q = controller_pb2.Controller.JobQuery()
    for k, v in fields.items():
        setattr(q, k, v)
    return q


# ---------------------------------------------------------------------------
# list_jobs parity
# ---------------------------------------------------------------------------


def _run_list_jobs(state, query, state_ids):
    with state._db.read_snapshot() as legacy_tx:
        legacy_rows, legacy_total = _query_jobs(legacy_tx, query, state_ids)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa_rows, sa_total = reads_dashboard.list_jobs(sa_tx, query, state_ids)
    return (legacy_rows, legacy_total), (sa_rows, sa_total)


def test_list_jobs_default_page_parity(state):
    _submit(state, "alpha")
    _submit(state, "beta")
    _submit(state, "gamma")

    query = _query(limit=20)  # default sort DATE/DESC
    (legacy_rows, legacy_total), (sa_rows, sa_total) = _run_list_jobs(state, query, USER_JOB_STATES)
    assert legacy_rows == sa_rows
    assert legacy_total == sa_total == 3


def test_list_jobs_sort_by_name_parity(state):
    _submit(state, "gamma")
    _submit(state, "alpha")
    _submit(state, "beta")

    query = _query(
        limit=20,
        sort_field=controller_pb2.Controller.JOB_SORT_FIELD_NAME,
        sort_direction=controller_pb2.Controller.SORT_DIRECTION_ASC,
    )
    (legacy_rows, legacy_total), (sa_rows, sa_total) = _run_list_jobs(state, query, USER_JOB_STATES)
    assert legacy_rows == sa_rows
    assert [r.name for r in sa_rows] == sorted(r.name for r in sa_rows)
    assert legacy_total == sa_total == 3


def test_list_jobs_scope_children_parity(state):
    _submit(state, "parent")
    _submit(state, "/test-user/parent/c1")
    _submit(state, "/test-user/parent/c2")
    _submit(state, "other")  # no parent — should not appear

    parent_wire = JobName.from_string("/test-user/parent").to_wire()
    query = _query(
        limit=20,
        scope=controller_pb2.Controller.JOB_QUERY_SCOPE_CHILDREN,
        parent_job_id=parent_wire,
    )
    (legacy_rows, legacy_total), (sa_rows, sa_total) = _run_list_jobs(state, query, USER_JOB_STATES)
    assert legacy_rows == sa_rows
    assert legacy_total == sa_total == 2


def test_list_jobs_sort_by_failures_parity(state):
    _submit(state, "j1", replicas=2)
    _submit(state, "j2", replicas=2)
    # Bump failure_count for j2's tasks so it sorts above j1 when descending.
    j2_wire = JobName.from_string("/test-user/j2").to_wire()
    state._db.execute("UPDATE tasks SET failure_count = 5 WHERE job_id = ?", (j2_wire,))

    query = _query(
        limit=20,
        sort_field=controller_pb2.Controller.JOB_SORT_FIELD_FAILURES,
        sort_direction=controller_pb2.Controller.SORT_DIRECTION_DESC,
    )
    (legacy_rows, legacy_total), (sa_rows, sa_total) = _run_list_jobs(state, query, USER_JOB_STATES)
    assert legacy_rows == sa_rows
    assert legacy_total == sa_total == 2
    assert sa_rows[0].name == "/test-user/j2"


def test_list_jobs_name_filter_parity(state):
    _submit(state, "production-train")
    _submit(state, "production-eval")
    _submit(state, "scratch")

    query = _query(limit=20, name_filter="production")
    (legacy_rows, legacy_total), (sa_rows, sa_total) = _run_list_jobs(state, query, USER_JOB_STATES)
    assert legacy_rows == sa_rows
    assert legacy_total == sa_total == 2


# ---------------------------------------------------------------------------
# task_summaries_for_jobs parity
# ---------------------------------------------------------------------------


def test_task_summaries_for_jobs_parity(state):
    _submit(state, "a", replicas=2)
    _submit(state, "b", replicas=3)
    ids = {JobName.from_string("/test-user/a"), JobName.from_string("/test-user/b")}

    with state._db.read_snapshot() as legacy_tx:
        legacy = _task_summaries_for_jobs(legacy_tx, ids)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_dashboard.task_summaries_for_jobs(sa_tx, ids)
    assert legacy == sa
    assert sum(s.task_count for s in sa.values()) == 5


def test_task_summaries_for_jobs_empty_parity(state):
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_dashboard.task_summaries_for_jobs(sa_tx, [])
    assert sa == {}


# ---------------------------------------------------------------------------
# parent_ids_with_children parity
# ---------------------------------------------------------------------------


def test_parent_ids_with_children_parity(state):
    _submit(state, "p1")
    _submit(state, "p2")
    _submit(state, "/test-user/p1/c1")

    p1 = JobName.from_string("/test-user/p1")
    p2 = JobName.from_string("/test-user/p2")
    candidates = [p1, p2]

    with state._db.read_snapshot() as legacy_tx:
        legacy = _parent_ids_with_children(legacy_tx, candidates)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_dashboard.parent_ids_with_children(sa_tx, candidates)
    assert legacy == sa == {p1}


def test_parent_ids_with_children_empty_parity(state):
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_dashboard.parent_ids_with_children(sa_tx, [])
    assert sa == set()
