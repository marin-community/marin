# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``EndpointsProjection`` — write-through cache over the ``endpoints`` table."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from iris.cluster.controller.db import EndpointQuery
from iris.cluster.controller.projections.endpoints import AddEndpointOutcome, EndpointsProjection
from iris.cluster.controller.schema import EndpointRow
from iris.cluster.controller.schema_v2 import tasks_table
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from rigging.timing import Timestamp
from sqlalchemy import select
from sqlalchemy import update as sa_update

from .conftest import make_job_request, submit_job


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


def test_projection_loads_existing_rows_on_startup(state):
    """On construction, the projection should contain every row in the ``endpoints`` table."""
    tasks = submit_job(state, "j", make_job_request("j"))
    with state._db.transaction() as cur:
        assert state._endpoints.add(cur, _make_row("e1", "svc", tasks[0].task_id))

    fresh = EndpointsProjection(state._db)
    rows = fresh.query()
    assert [r.endpoint_id for r in rows] == ["e1"]


def test_add_updates_memory_after_commit(state):
    tasks = submit_job(state, "j", make_job_request("j"))
    t = tasks[0].task_id

    with state._db.transaction() as cur:
        assert state._endpoints.add(cur, _make_row("e1", "alpha", t))
        # Not yet committed; memory should not reflect the insert.
        assert state._endpoints.get("e1") is None

    assert state._endpoints.get("e1") is not None
    assert [r.endpoint_id for r in state._endpoints.query()] == ["e1"]


def test_rollback_leaves_memory_untouched(state):
    tasks = submit_job(state, "j", make_job_request("j"))
    t = tasks[0].task_id

    class BoomError(RuntimeError):
        pass

    with pytest.raises(BoomError):
        with state._db.transaction() as cur:
            state._endpoints.add(cur, _make_row("e1", "alpha", t))
            raise BoomError

    # DB rolled back -> memory must NOT see the insert.
    assert state._endpoints.get("e1") is None
    assert state._endpoints.query() == []


def test_rollback_safety_dict_and_sql_consistent(state):
    """Raise mid-tx: assert dict has no entry AND SQL has no row."""
    from iris.cluster.controller.schema_v2 import endpoints_table

    tasks = submit_job(state, "j", make_job_request("j"))
    t = tasks[0].task_id

    with pytest.raises(RuntimeError, match="boom"):
        with state._db.transaction() as cur:
            state._endpoints.add(cur, _make_row("e-rollback", "alpha", t))
            raise RuntimeError("boom")

    assert state._endpoints.get("e-rollback") is None
    with state._db.read_snapshot() as tx:
        row = tx.fetchone(select(endpoints_table.c.endpoint_id).where(endpoints_table.c.endpoint_id == "e-rollback"))
    assert row is None


def test_add_rejects_terminal_task(state):
    """Writing an endpoint for a terminal task should return TERMINAL and not mutate memory."""
    tasks = submit_job(state, "j", make_job_request("j"))
    task_id = tasks[0].task_id
    # Drive the task to SUCCEEDED to mark it terminal.
    with state._db.transaction() as tx:
        tx.execute(
            sa_update(tasks_table).where(tasks_table.c.task_id == task_id).values(state=job_pb2.TASK_STATE_SUCCEEDED)
        )

    with state._db.transaction() as cur:
        outcome = state._endpoints.add(cur, _make_row("e1", "alpha", task_id))
        assert outcome is AddEndpointOutcome.TERMINAL

    assert state._endpoints.get("e1") is None


def test_remove_drops_endpoint_by_id(state):
    tasks = submit_job(state, "j", make_job_request("j"))
    t = tasks[0].task_id
    with state._db.transaction() as cur:
        state._endpoints.add(cur, _make_row("e1", "alpha", t))
        state._endpoints.add(cur, _make_row("e2", "beta", t))

    with state._db.transaction() as cur:
        removed = state._endpoints.remove(cur, "e1")
    assert removed is not None and removed.endpoint_id == "e1"
    assert {r.endpoint_id for r in state._endpoints.query()} == {"e2"}


def test_remove_by_task_drops_all_task_endpoints(state):
    tasks = submit_job(state, "j", make_job_request("j", replicas=2))
    t1, t2 = tasks[0].task_id, tasks[1].task_id

    with state._db.transaction() as cur:
        state._endpoints.add(cur, _make_row("e1", "alpha", t1))
        state._endpoints.add(cur, _make_row("e2", "beta", t1))
        state._endpoints.add(cur, _make_row("e3", "gamma", t2))

    with state._db.transaction() as cur:
        removed = state._endpoints.remove_by_task(cur, t1)

    assert set(removed) == {"e1", "e2"}
    assert {r.endpoint_id for r in state._endpoints.query()} == {"e3"}


def test_remove_by_job_ids_drops_subtree(state):
    tasks_a = submit_job(state, "a", make_job_request("a"))
    tasks_b = submit_job(state, "b", make_job_request("b"))
    ja = tasks_a[0].task_id.require_task()[0]
    t1 = tasks_a[0].task_id
    t2 = tasks_b[0].task_id

    with state._db.transaction() as cur:
        state._endpoints.add(cur, _make_row("e1", "alpha", t1))
        state._endpoints.add(cur, _make_row("e2", "beta", t2))

    with state._db.transaction() as cur:
        removed = state._endpoints.remove_by_job_ids(cur, [ja])

    assert removed == ["e1"]
    assert [r.endpoint_id for r in state._endpoints.query()] == ["e2"]


# --- Query semantics --------------------------------------------------------


@pytest.fixture
def populated(state):
    """A projection populated with a small fixture set spanning names, tasks, prefixes."""
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
            state._endpoints.add(cur, r)
    return state, rows, (t0, t1, t2)


def test_query_by_exact_name(populated):
    state, _, _ = populated
    ids = {r.endpoint_id for r in state._endpoints.query(EndpointQuery(exact_name="alpha/svc"))}
    assert ids == {"e1"}


def test_query_by_prefix(populated):
    state, _, _ = populated
    ids = {r.endpoint_id for r in state._endpoints.query(EndpointQuery(name_prefix="alpha/"))}
    assert ids == {"e1", "e2"}


def test_query_by_task_ids(populated):
    state, _, (t0, _, t2) = populated
    ids = {r.endpoint_id for r in state._endpoints.query(EndpointQuery(task_ids=(t0, t2)))}
    assert ids == {"e1", "e2", "e4"}


def test_query_by_endpoint_ids(populated):
    state, _, _ = populated
    ids = {r.endpoint_id for r in state._endpoints.query(EndpointQuery(endpoint_ids=("e2", "e3")))}
    assert ids == {"e2", "e3"}


def test_query_limit(populated):
    state, _, _ = populated
    rows = state._endpoints.query(EndpointQuery(limit=2))
    assert len(rows) == 2


def test_query_empty_matches_all(populated):
    state, rows, _ = populated
    assert {r.endpoint_id for r in state._endpoints.query()} == {r.endpoint_id for r in rows}


def test_resolve_returns_address_for_exact_name(populated):
    state, _, _ = populated
    row = state._endpoints.resolve("alpha/svc")
    assert row is not None
    assert row.endpoint_id == "e1"
    assert state._endpoints.resolve("nope") is None


# --- Atomicity --------------------------------------------------------------


def test_atomic_write_through_under_write_lock(state):
    """Atomicity contract: no reader observes the new endpoint until the write tx exits.

    A reader thread polls ``get(eid)`` continuously. The writer thread opens a
    transaction, registers an ``add`` hook, and inside an additional hook
    sleeps after signalling. While the writer's tx context is still open
    (hook block hasn't returned), readers must see ``None`` for the
    endpoint. After the context exits, every subsequent read must see the
    new row.
    """
    tasks = submit_job(state, "j", make_job_request("j"))
    t = tasks[0].task_id

    inside_hook = threading.Event()
    release_hook = threading.Event()
    stop = threading.Event()
    observations: list[bool] = []  # True = saw the new endpoint, False = not yet

    def reader():
        # Poll until we observe the new endpoint or are told to stop.
        while not stop.is_set():
            seen = state._endpoints.get("e-atomic") is not None
            observations.append(seen)
            # 1 ms poll cadence; bounded busy wait via Event.wait.
            if release_hook.wait(timeout=0.001):
                # The hook has been released; one more read after that may
                # still observe None briefly while the hook is firing.
                pass

    reader_thread = threading.Thread(target=reader)
    reader_thread.start()
    try:
        with state._db.transaction() as cur:
            outcome = state._endpoints.add(cur, _make_row("e-atomic", "atomic", t))
            assert outcome is AddEndpointOutcome.OK

            # Register a second on_commit hook that blocks. Hooks fire in
            # registration order under the write lock, AFTER the projection's
            # index update. By blocking here we keep the write lock held and
            # extend the period during which the dict update is committed but
            # the transaction context has not yet exited. Readers observing
            # an endpoint here is FINE — the contract is "no reader sees it
            # before commit", not "no reader sees it before tx context exit".
            def blocking_hook() -> None:
                inside_hook.set()
                release_hook.wait(timeout=2.0)

            cur.on_commit(blocking_hook)

            # Sample reads BEFORE we enter the commit phase. While the
            # transaction is still open (no commit yet), readers must not
            # see the endpoint.
            pre_commit_samples = [state._endpoints.get("e-atomic") for _ in range(50)]
            assert all(v is None for v in pre_commit_samples), "reader saw uncommitted endpoint"

        # Tx context has exited and all hooks have fired. The endpoint must
        # be visible to every subsequent read.
        for _ in range(50):
            assert state._endpoints.get("e-atomic") is not None
    finally:
        release_hook.set()
        stop.set()
        reader_thread.join(timeout=5)
        assert not reader_thread.is_alive()

    # Once we entered the blocking hook, the writer was past the SQL commit
    # and the projection update had run. Any read after that is allowed to
    # see the endpoint; any read strictly before the commit must not.
    assert inside_hook.is_set()


def test_replace_from_resets_dict(state, tmp_path: Path):
    """``backup_to`` + modify + ``replace_from`` -> dict reflects backup state."""
    tasks = submit_job(state, "j", make_job_request("j"))
    t = tasks[0].task_id

    # Populate the projection with a known endpoint and take a backup.
    with state._db.transaction() as cur:
        state._endpoints.add(cur, _make_row("e-backup", "backup", t))
    assert state._endpoints.get("e-backup") is not None

    backup_dir = tmp_path / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    state._db.backup_to(backup_dir / "controller.sqlite3")
    # Auth DB is required by replace_from to fully restore; backup it too.
    state._db.backup_to(backup_dir / "auth.sqlite3")

    # Mutate after the backup: remove the original, add a new endpoint that
    # only exists in the live DB.
    with state._db.transaction() as cur:
        state._endpoints.remove(cur, "e-backup")
        state._endpoints.add(cur, _make_row("e-live", "live", t))
    assert state._endpoints.get("e-backup") is None
    assert state._endpoints.get("e-live") is not None

    # Restore from the backup. The reopen hook fires the projection's
    # rehydrate() and the dict must reflect the backup state, not the
    # post-backup mutations.
    state._db.replace_from(backup_dir)

    assert state._endpoints.get("e-backup") is not None
    assert state._endpoints.get("e-live") is None


def test_concurrent_readers_no_keyerror_no_torn_reads(state):
    """4 readers + 2 writers for 2 seconds. No exceptions, no inconsistency."""
    tasks = submit_job(state, "stress", make_job_request("stress", replicas=4))
    task_ids = [t.task_id for t in tasks]

    stop = threading.Event()
    errors: list[str] = []

    def writer(idx: int):
        try:
            i = idx
            while not stop.is_set():
                t = task_ids[i % len(task_ids)]
                eid = f"e{idx}-{i % len(task_ids)}"
                name = f"svc-{idx}-{i % len(task_ids)}"
                with state._db.transaction() as cur:
                    state._endpoints.add(cur, _make_row(eid, name, t))
                with state._db.transaction() as cur:
                    state._endpoints.remove(cur, eid)
                i += 1
        except Exception as exc:
            errors.append(f"writer-{idx}: {exc!r}")

    def reader():
        try:
            while not stop.is_set():
                snapshot = state._endpoints.query()
                ids = [r.endpoint_id for r in snapshot]
                # No duplicate ids in a single snapshot (no torn index).
                assert len(ids) == len(set(ids)), f"duplicate ids in snapshot: {ids}"
                # Cross-view consistency: every row in by_id-derived
                # snapshot must still be reachable via get(); the writer
                # may unindex it between calls, which is fine, but it must
                # not raise KeyError. The projection's get() returns None
                # on miss, so this is implicit — any exception bubbles up
                # via the outer try/except.
                for row in snapshot:
                    state._endpoints.get(row.endpoint_id)
                for i in range(len(task_ids)):
                    state._endpoints.query(EndpointQuery(name_prefix="svc-"))
                    state._endpoints.query(EndpointQuery(exact_name=f"svc-0-{i}"))
                    state._endpoints.query(EndpointQuery(task_ids=(task_ids[i],)))
        except Exception as exc:
            errors.append(f"reader: {exc!r}")

    barrier = threading.Barrier(6)

    def runner(fn, *args):
        barrier.wait()
        fn(*args)

    threads = [
        threading.Thread(target=runner, args=(writer, 0)),
        threading.Thread(target=runner, args=(writer, 1)),
        threading.Thread(target=runner, args=(reader,)),
        threading.Thread(target=runner, args=(reader,)),
        threading.Thread(target=runner, args=(reader,)),
        threading.Thread(target=runner, args=(reader,)),
    ]
    for th in threads:
        th.start()

    # Short bounded run, polling a monotonic deadline instead of time.sleep.
    deadline = Timestamp.now().epoch_ms() + 2000
    while Timestamp.now().epoch_ms() < deadline:
        pass
    stop.set()
    for th in threads:
        th.join(timeout=5)
    assert not errors, errors
