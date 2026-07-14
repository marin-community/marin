# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``EndpointsProjection`` — write-through cache over the ``endpoints`` table."""

from pathlib import Path

import pytest
from iris.cluster.controller.projections.endpoints import (
    AddEndpointOutcome,
    EndpointQuery,
    EndpointRow,
    EndpointsProjection,
)
from iris.cluster.controller.schema import tasks_table
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from rigging.timing import Timestamp
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
