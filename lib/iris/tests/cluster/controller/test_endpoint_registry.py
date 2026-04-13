# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for EndpointRegistry — the in-memory cache over the ``endpoints`` table."""

from __future__ import annotations

import threading

import pytest

from iris.cluster.controller.db import EndpointQuery
from iris.cluster.controller.endpoint_registry import EndpointRegistry
from iris.cluster.controller.schema import ENDPOINT_PROJECTION, EndpointRow
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from rigging.timing import Timestamp

from .conftest import make_job_request, submit_job


# --- Parity helper: the legacy SQL builder, preserved solely for parity tests.
# Deleted from production; kept here so a parity test demonstrates the registry
# returns an identical row set for representative queries.
def _endpoint_query_sql_legacy(query: EndpointQuery) -> tuple[str, list[object]]:
    from_clause = f"SELECT {ENDPOINT_PROJECTION.select_clause()} FROM endpoints e"
    conditions: list[str] = []
    params: list[object] = []

    if query.task_ids:
        from_clause += " JOIN endpoints et ON e.endpoint_id = et.endpoint_id"
        placeholders = ",".join("?" for _ in query.task_ids)
        conditions.append(f"et.task_id IN ({placeholders})")
        params.extend(tid.to_wire() for tid in query.task_ids)

    if query.endpoint_ids:
        placeholders = ",".join("?" for _ in query.endpoint_ids)
        conditions.append(f"e.endpoint_id IN ({placeholders})")
        params.extend(query.endpoint_ids)

    if query.name_prefix:
        conditions.append("e.name LIKE ?")
        params.append(f"{query.name_prefix}%")

    if query.exact_name:
        conditions.append("e.name = ?")
        params.append(query.exact_name)

    sql = from_clause
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    if query.limit is not None:
        sql += " LIMIT ?"
        params.append(query.limit)
    return sql, params


def _make_row(endpoint_id: str, name: str, task_id: JobName, *, address: str = "h:1") -> EndpointRow:
    return EndpointRow(
        endpoint_id=endpoint_id,
        name=name,
        address=address,
        task_id=task_id,
        metadata={},
        registered_at=Timestamp.now(),
    )


# --- Load / add / remove ----------------------------------------------------


def test_registry_loads_existing_rows_on_startup(state):
    """On construction, the registry should contain every row in the ``endpoints`` table."""
    tasks = submit_job(state, "j", make_job_request("j"))
    with state._db.transaction() as cur:
        assert state._db.endpoints.add(cur, _make_row("e1", "svc", tasks[0].task_id))

    fresh = EndpointRegistry(state._db)
    rows = fresh.query()
    assert [r.endpoint_id for r in rows] == ["e1"]


def test_add_updates_memory_after_commit(state):
    tasks = submit_job(state, "j", make_job_request("j"))
    t = tasks[0].task_id

    with state._db.transaction() as cur:
        assert state._db.endpoints.add(cur, _make_row("e1", "alpha", t))
        # Not yet committed; memory should not reflect the insert.
        assert state._db.endpoints.get("e1") is None

    assert state._db.endpoints.get("e1") is not None
    assert [r.endpoint_id for r in state._db.endpoints.query()] == ["e1"]


def test_rollback_leaves_memory_untouched(state):
    tasks = submit_job(state, "j", make_job_request("j"))
    t = tasks[0].task_id

    class BoomError(RuntimeError):
        pass

    with pytest.raises(BoomError):
        with state._db.transaction() as cur:
            state._db.endpoints.add(cur, _make_row("e1", "alpha", t))
            raise BoomError

    # DB rolled back → memory must NOT see the insert.
    assert state._db.endpoints.get("e1") is None
    assert state._db.endpoints.query() == []


def test_add_rejects_terminal_task(state):
    """Writing an endpoint for a terminal task should return False and not mutate memory."""
    tasks = submit_job(state, "j", make_job_request("j"))
    task_id = tasks[0].task_id
    # Drive the task to SUCCEEDED to mark it terminal.
    state._db.execute(
        "UPDATE tasks SET state = ? WHERE task_id = ?",
        (job_pb2.TASK_STATE_SUCCEEDED, task_id.to_wire()),
    )

    with state._db.transaction() as cur:
        assert state._db.endpoints.add(cur, _make_row("e1", "alpha", task_id)) is False

    assert state._db.endpoints.get("e1") is None


def test_remove_drops_endpoint_by_id(state):
    tasks = submit_job(state, "j", make_job_request("j"))
    t = tasks[0].task_id
    with state._db.transaction() as cur:
        state._db.endpoints.add(cur, _make_row("e1", "alpha", t))
        state._db.endpoints.add(cur, _make_row("e2", "beta", t))

    with state._db.transaction() as cur:
        removed = state._db.endpoints.remove(cur, "e1")
    assert removed is not None and removed.endpoint_id == "e1"
    assert {r.endpoint_id for r in state._db.endpoints.query()} == {"e2"}


def test_remove_by_task_drops_all_task_endpoints(state):
    tasks = submit_job(state, "j", make_job_request("j", replicas=2))
    t1, t2 = tasks[0].task_id, tasks[1].task_id

    with state._db.transaction() as cur:
        state._db.endpoints.add(cur, _make_row("e1", "alpha", t1))
        state._db.endpoints.add(cur, _make_row("e2", "beta", t1))
        state._db.endpoints.add(cur, _make_row("e3", "gamma", t2))

    with state._db.transaction() as cur:
        removed = state._db.endpoints.remove_by_task(cur, t1)

    assert set(removed) == {"e1", "e2"}
    assert {r.endpoint_id for r in state._db.endpoints.query()} == {"e3"}


def test_remove_by_job_ids_drops_subtree(state):
    tasks_a = submit_job(state, "a", make_job_request("a"))
    tasks_b = submit_job(state, "b", make_job_request("b"))
    ja = tasks_a[0].task_id.require_task()[0]
    t1 = tasks_a[0].task_id
    t2 = tasks_b[0].task_id

    with state._db.transaction() as cur:
        state._db.endpoints.add(cur, _make_row("e1", "alpha", t1))
        state._db.endpoints.add(cur, _make_row("e2", "beta", t2))

    with state._db.transaction() as cur:
        removed = state._db.endpoints.remove_by_job_ids(cur, [ja])

    assert removed == ["e1"]
    assert [r.endpoint_id for r in state._db.endpoints.query()] == ["e2"]


# --- Query semantics --------------------------------------------------------


@pytest.fixture
def populated(state):
    """A registry populated with a small fixture set spanning names, tasks, prefixes."""
    tasks_j = submit_job(state, "j", make_job_request("j", replicas=2))
    tasks_other = submit_job(state, "other", make_job_request("other"))
    t0 = tasks_j[0].task_id
    t1 = tasks_j[1].task_id
    t2 = tasks_other[0].task_id

    rows = [
        _make_row("e1", "alpha/svc", t0),
        _make_row("e2", "alpha/worker", t0),
        _make_row("e3", "beta/svc", t1),
        _make_row("e4", "gamma/svc", t2),
    ]
    with state._db.transaction() as cur:
        for r in rows:
            state._db.endpoints.add(cur, r)
    return state, rows, (t0, t1, t2)


def test_query_by_exact_name(populated):
    state, _, _ = populated
    ids = {r.endpoint_id for r in state._db.endpoints.query(EndpointQuery(exact_name="alpha/svc"))}
    assert ids == {"e1"}


def test_query_by_prefix(populated):
    state, _, _ = populated
    ids = {r.endpoint_id for r in state._db.endpoints.query(EndpointQuery(name_prefix="alpha/"))}
    assert ids == {"e1", "e2"}


def test_query_by_task_ids(populated):
    state, _, (t0, _, t2) = populated
    ids = {r.endpoint_id for r in state._db.endpoints.query(EndpointQuery(task_ids=(t0, t2)))}
    assert ids == {"e1", "e2", "e4"}


def test_query_by_endpoint_ids(populated):
    state, _, _ = populated
    ids = {r.endpoint_id for r in state._db.endpoints.query(EndpointQuery(endpoint_ids=("e2", "e3")))}
    assert ids == {"e2", "e3"}


def test_query_limit(populated):
    state, _, _ = populated
    rows = state._db.endpoints.query(EndpointQuery(limit=2))
    assert len(rows) == 2


def test_query_empty_matches_all(populated):
    state, rows, _ = populated
    assert {r.endpoint_id for r in state._db.endpoints.query()} == {r.endpoint_id for r in rows}


def test_resolve_returns_address_for_exact_name(populated):
    state, _, _ = populated
    row = state._db.endpoints.resolve("alpha/svc")
    assert row is not None
    assert row.endpoint_id == "e1"
    assert state._db.endpoints.resolve("nope") is None


# --- Parity with the legacy SQL builder -------------------------------------


@pytest.mark.parametrize(
    "build_query",
    [
        lambda t0, t1, t2: EndpointQuery(),
        lambda t0, t1, t2: EndpointQuery(exact_name="alpha/svc"),
        lambda t0, t1, t2: EndpointQuery(name_prefix="alpha"),
        lambda t0, t1, t2: EndpointQuery(task_ids=(t0,)),
        lambda t0, t1, t2: EndpointQuery(task_ids=(t0, t2)),
        lambda t0, t1, t2: EndpointQuery(endpoint_ids=("e1", "e3")),
        lambda t0, t1, t2: EndpointQuery(name_prefix="alpha", limit=1),
    ],
)
def test_registry_parity_with_legacy_sql(populated, build_query):
    state, _, (t0, t1, t2) = populated
    query = build_query(t0, t1, t2)

    sql, params = _endpoint_query_sql_legacy(query)
    with state._db.read_snapshot() as q:
        expected_ids = sorted(r.endpoint_id for r in ENDPOINT_PROJECTION.decode(q.fetchall(sql, tuple(params))))
    actual_ids = sorted(r.endpoint_id for r in state._db.endpoints.query(query))

    # For LIMIT queries, both sides just need to be a valid subset of matching rows.
    if query.limit is not None:
        assert len(actual_ids) == len(expected_ids)
        return
    assert actual_ids == expected_ids


# --- Concurrency ------------------------------------------------------------


def test_concurrent_readers_never_see_torn_snapshot(state):
    """Interleave add/remove with concurrent queries; every snapshot must be internally consistent."""
    tasks = submit_job(state, "stress", make_job_request("stress", replicas=4))
    task_ids = [t.task_id for t in tasks]

    stop = threading.Event()
    errors: list[str] = []

    def writer():
        try:
            i = 0
            while not stop.is_set():
                t = task_ids[i % len(task_ids)]
                eid = f"e{i % len(task_ids)}"
                name = f"svc-{i % len(task_ids)}"
                with state._db.transaction() as cur:
                    state._db.endpoints.add(cur, _make_row(eid, name, t))
                with state._db.transaction() as cur:
                    state._db.endpoints.remove(cur, eid)
                i += 1
        except Exception as exc:
            errors.append(f"writer: {exc!r}")

    def reader():
        try:
            while not stop.is_set():
                snapshot = state._db.endpoints.query()
                # Verify the snapshot itself is internally consistent: every
                # endpoint_id in the result set is unique (no duplicates from
                # a torn index).
                ids = [r.endpoint_id for r in snapshot]
                assert len(ids) == len(set(ids)), f"duplicate ids in snapshot: {ids}"
                # Exercise the secondary-index query paths concurrently with
                # mutations. We cannot assert that a row from query() is still
                # present in a subsequent get() — the writer may remove it
                # between the two calls (TOCTOU).
                for row in snapshot:
                    state._db.endpoints.get(row.endpoint_id)
                for i in range(len(task_ids)):
                    state._db.endpoints.query(EndpointQuery(name_prefix=f"svc-{i}"))
                    state._db.endpoints.query(EndpointQuery(exact_name=f"svc-{i}"))
                    state._db.endpoints.query(EndpointQuery(task_ids=(task_ids[i],)))
        except Exception as exc:
            errors.append(f"reader: {exc!r}")

    barrier = threading.Barrier(4)

    def runner(fn):
        barrier.wait()
        fn()

    threads = [
        threading.Thread(target=runner, args=(writer,)),
        threading.Thread(target=runner, args=(reader,)),
        threading.Thread(target=runner, args=(reader,)),
        threading.Thread(target=runner, args=(reader,)),
    ]
    for th in threads:
        th.start()

    # Short bounded run, polling a monotonic deadline instead of time.sleep.
    deadline = Timestamp.now().epoch_ms() + 500
    while Timestamp.now().epoch_ms() < deadline:
        pass
    stop.set()
    for th in threads:
        th.join(timeout=5)
    assert not errors, errors
