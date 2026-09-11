# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``WorkerAttrsProjection`` — write-through cache over ``worker_attributes``."""

import threading
from pathlib import Path

import pytest
from iris.cluster.constraints import AttributeValue
from iris.cluster.controller import db, writes
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.schema import worker_attributes_table, workers_table
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import WorkerId
from sqlalchemy import insert, select


def _insert_worker(state, worker_id: str, *, scale_group: str = "") -> WorkerId:
    """SA Core worker insertion used to drive the FK-cascade scenario."""
    with state._db.transaction() as cur:
        cur.execute(
            insert(workers_table).values(
                worker_id=WorkerId(worker_id), address=f"{worker_id}.example:8080", scale_group=scale_group
            )
        )
    return WorkerId(worker_id)


def _insert_worker_attribute(state, worker_id: WorkerId, key: str, value: str) -> None:
    """Set a single string attribute via the projection's ``set`` method."""
    with state._db.transaction() as cur:
        existing = state._worker_attrs.get(worker_id)
        merged = {**existing, key: AttributeValue(value)}
        state._worker_attrs.set(cur, worker_id, merged)


def test_set_and_get_returns_cached_attributes(state):
    worker_id = _insert_worker(state, "w-set")
    _insert_worker_attribute(state, worker_id, "region", "us-east1")

    assert state._worker_attrs.get(worker_id) == {"region": AttributeValue("us-east1")}


def test_get_missing_worker_returns_empty_dict(state):
    assert state._worker_attrs.get(WorkerId("never-registered")) == {}


def test_all_returns_copy(state):
    worker_id = _insert_worker(state, "w-all")
    _insert_worker_attribute(state, worker_id, "zone", "us-east1-b")

    snapshot = state._worker_attrs.all()
    assert snapshot == {worker_id: {"zone": AttributeValue("us-east1-b")}}
    # Mutating the snapshot must not leak back into the cache.
    snapshot[worker_id]["zone"] = AttributeValue("mutated")
    assert state._worker_attrs.get(worker_id) == {"zone": AttributeValue("us-east1-b")}


def test_invalidate_drops_cache_entry_after_commit(state):
    worker_id = _insert_worker(state, "w-remove")
    _insert_worker_attribute(state, worker_id, "region", "us-east1")
    assert state._worker_attrs.get(worker_id) != {}

    with state._db.transaction() as cur:
        state._worker_attrs.invalidate_for_worker(cur, worker_id)
        # Hook fires post-commit; mid-tx the dict still has the entry.
        assert state._worker_attrs.get(worker_id) != {}

    assert state._worker_attrs.get(worker_id) == {}


def test_rehydrate_reflects_disk_state(state):
    worker_id = _insert_worker(state, "w-rehydrate")
    # Insert SQL row only, bypassing the projection — the cache is intentionally stale.
    with state._db.transaction() as cur:
        cur.execute(
            insert(worker_attributes_table).values(
                worker_id=worker_id,
                key="zone",
                value_type="str",
                str_value="us-east1-b",
                int_value=None,
                float_value=None,
            )
        )

    fresh = WorkerAttrsProjection(state._db)
    assert fresh.get(worker_id) == {"zone": AttributeValue("us-east1-b")}


def test_rehydrate_loads_all_workers(state):
    """Rehydration loads every worker's attributes unconditionally."""
    worker_a = _insert_worker(state, "w-a", scale_group="sg-a")
    worker_b = _insert_worker(state, "w-b", scale_group="sg-b")
    _insert_worker_attribute(state, worker_a, "zone", "us-east1-a")
    _insert_worker_attribute(state, worker_b, "zone", "us-east1-b")

    fresh = WorkerAttrsProjection(state._db)
    assert fresh.get(worker_a) == {"zone": AttributeValue("us-east1-a")}
    assert fresh.get(worker_b) == {"zone": AttributeValue("us-east1-b")}


def test_atomic_write_through_no_visibility_before_commit(state):
    """A reader inside the writer's transaction must not see the new attrs.

    Tightens the contract: until the surrounding ``with state._db.transaction()``
    block exits and the SQL is committed, no observer can see the new dict
    entry. After exit, the entry is unconditionally visible.
    """
    worker_id = _insert_worker(state, "w-atomic")

    with state._db.transaction() as cur:
        state._worker_attrs.set(cur, worker_id, {"region": AttributeValue("eu-west1")})
        # Hooks have not fired yet; cache is still empty for this worker.
        assert state._worker_attrs.get(worker_id) == {}

    assert state._worker_attrs.get(worker_id) == {"region": AttributeValue("eu-west1")}


def test_rollback_leaves_cache_untouched(state):
    worker_id = _insert_worker(state, "w-rollback")

    class BoomError(RuntimeError):
        pass

    with pytest.raises(BoomError):
        with state._db.transaction() as cur:
            state._worker_attrs.set(cur, worker_id, {"region": AttributeValue("ap-south1")})
            raise BoomError

    assert state._worker_attrs.get(worker_id) == {}
    with state._db.read_snapshot() as q:
        row = q.execute(
            select(worker_attributes_table.c.key).where(worker_attributes_table.c.worker_id == worker_id)
        ).first()
    assert row is None


def test_replace_from_resets_cache(state, tmp_path: Path):
    """A backup → restore round-trip rehydrates the projection from disk."""
    worker_id = _insert_worker(state, "w-backup")
    _insert_worker_attribute(state, worker_id, "region", "us-east1")

    backup_dir = tmp_path / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    state._db.backup_to(backup_dir / "controller.sqlite3")
    state._db.backup_to(backup_dir / "auth.sqlite3")

    # Mutate post-backup: add a second worker that exists only in the live DB.
    worker_live = _insert_worker(state, "w-live")
    _insert_worker_attribute(state, worker_live, "zone", "us-east1-c")
    assert state._worker_attrs.get(worker_live) != {}

    state._db.replace_from(backup_dir)

    assert state._worker_attrs.get(worker_id) == {"region": AttributeValue("us-east1")}
    assert state._worker_attrs.get(worker_live) == {}


def test_cascade_delete_invalidates_projection(state):
    """Deleting a worker FK-cascades into worker_attributes; the cache must follow.

    ``writes.remove_worker`` calls ``WorkerAttrsProjection.invalidate_for_worker``
    via ``tx.caches`` so the in-memory dict drops the entry atomically with the
    SQL commit.
    """
    worker_id = _insert_worker(state, "w-cascade")
    _insert_worker_attribute(state, worker_id, "region", "us-east1")
    assert state._worker_attrs.get(worker_id) == {"region": AttributeValue("us-east1")}

    health = WorkerHealthTracker()
    health.register(worker_id, now_ms=1000)
    with db.write_transaction(state._db.sa_write_engine, threading.RLock(), state._db.caches) as tx:
        writes.remove_worker(
            tx,
            worker_id,
            health=health,
        )

    with state._db.read_snapshot() as q:
        row = q.execute(
            select(worker_attributes_table.c.key).where(worker_attributes_table.c.worker_id == worker_id)
        ).first()
    assert row is None

    assert state._worker_attrs.get(worker_id) == {}
