# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``WorkerAttrsProjection`` — write-through cache over ``worker_attributes``."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from iris.cluster.constraints import AttributeValue
from iris.cluster.controller import db
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.controller.writes import workers as writes_workers
from iris.cluster.types import WorkerId


def _insert_worker(state, worker_id: str) -> WorkerId:
    """Direct-SQL worker insertion used to drive the FK-cascade scenario."""
    with state._db.transaction() as cur:
        cur.execute(
            "INSERT INTO workers (worker_id, address) VALUES (?, ?)",
            (worker_id, f"{worker_id}.example:8080"),
        )
    return WorkerId(worker_id)


def _insert_worker_attribute(state, worker_id: WorkerId, key: str, value: str) -> None:
    """Direct-SQL string-typed attribute insertion. The projection update is
    registered via ``set`` so the in-memory cache matches the on-disk row."""
    with state._db.transaction() as cur:
        cur.execute(
            "INSERT INTO worker_attributes(worker_id, key, value_type, str_value, int_value, float_value) "
            "VALUES (?, ?, ?, ?, NULL, NULL)",
            (str(worker_id), key, "str", value),
        )
        state._store.worker_attrs.set(cur, worker_id, {key: AttributeValue(value)})


def test_set_and_get_returns_cached_attributes(state):
    worker_id = _insert_worker(state, "w-set")
    _insert_worker_attribute(state, worker_id, "region", "us-east1")

    assert state._store.worker_attrs.get(worker_id) == {"region": AttributeValue("us-east1")}


def test_get_missing_worker_returns_empty_dict(state):
    assert state._store.worker_attrs.get(WorkerId("never-registered")) == {}


def test_all_returns_copy(state):
    worker_id = _insert_worker(state, "w-all")
    _insert_worker_attribute(state, worker_id, "zone", "us-east1-b")

    snapshot = state._store.worker_attrs.all()
    assert snapshot == {worker_id: {"zone": AttributeValue("us-east1-b")}}
    # Mutating the snapshot must not leak back into the cache.
    snapshot[worker_id]["zone"] = AttributeValue("mutated")
    assert state._store.worker_attrs.get(worker_id) == {"zone": AttributeValue("us-east1-b")}


def test_remove_drops_cache_entry_after_commit(state):
    worker_id = _insert_worker(state, "w-remove")
    _insert_worker_attribute(state, worker_id, "region", "us-east1")
    assert state._store.worker_attrs.get(worker_id) != {}

    with state._db.transaction() as cur:
        state._store.worker_attrs.remove(cur, worker_id)
        # Hook fires post-commit; mid-tx the dict still has the entry.
        assert state._store.worker_attrs.get(worker_id) != {}

    assert state._store.worker_attrs.get(worker_id) == {}


def test_rehydrate_reflects_disk_state(state):
    worker_id = _insert_worker(state, "w-rehydrate")
    # Insert SQL row only, no projection update — the cache is intentionally stale.
    with state._db.transaction() as cur:
        cur.execute(
            "INSERT INTO worker_attributes(worker_id, key, value_type, str_value, int_value, float_value) "
            "VALUES (?, ?, ?, ?, NULL, NULL)",
            (str(worker_id), "zone", "str", "us-east1-b"),
        )

    fresh = WorkerAttrsProjection(state._db)
    assert fresh.get(worker_id) == {"zone": AttributeValue("us-east1-b")}


def test_atomic_write_through_no_visibility_before_commit(state):
    """A reader inside the writer's transaction must not see the new attrs.

    Tightens the contract: until the surrounding ``with state._db.transaction()``
    block exits and the SQL is committed, no observer can see the new dict
    entry. After exit, the entry is unconditionally visible.
    """
    worker_id = _insert_worker(state, "w-atomic")

    with state._db.transaction() as cur:
        cur.execute(
            "INSERT INTO worker_attributes(worker_id, key, value_type, str_value, int_value, float_value) "
            "VALUES (?, ?, ?, ?, NULL, NULL)",
            (str(worker_id), "region", "str", "eu-west1"),
        )
        state._store.worker_attrs.set(cur, worker_id, {"region": AttributeValue("eu-west1")})
        # Hooks have not fired yet; cache is still empty for this worker.
        assert state._store.worker_attrs.get(worker_id) == {}

    assert state._store.worker_attrs.get(worker_id) == {"region": AttributeValue("eu-west1")}


def test_rollback_leaves_cache_untouched(state):
    worker_id = _insert_worker(state, "w-rollback")

    class BoomError(RuntimeError):
        pass

    with pytest.raises(BoomError):
        with state._db.transaction() as cur:
            cur.execute(
                "INSERT INTO worker_attributes(worker_id, key, value_type, str_value, int_value, float_value) "
                "VALUES (?, ?, ?, ?, NULL, NULL)",
                (str(worker_id), "region", "str", "ap-south1"),
            )
            state._store.worker_attrs.set(cur, worker_id, {"region": AttributeValue("ap-south1")})
            raise BoomError

    assert state._store.worker_attrs.get(worker_id) == {}
    with state._db.read_snapshot() as q:
        row = q.fetchone(
            "SELECT key FROM worker_attributes WHERE worker_id = ?",
            (str(worker_id),),
        )
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
    assert state._store.worker_attrs.get(worker_live) != {}

    state._db.replace_from(backup_dir)

    assert state._store.worker_attrs.get(worker_id) == {"region": AttributeValue("us-east1")}
    assert state._store.worker_attrs.get(worker_live) == {}


def test_cascade_delete_invalidates_projection(state):
    """Deleting a worker FK-cascades into worker_attributes; the cache must follow.

    Stage 12 routes the cascading delete through ``writes/workers.remove_worker``,
    which calls ``WorkerAttrsProjection.invalidate_for_worker`` inline so the
    in-memory dict drops the entry atomically with the SQL commit.
    """
    worker_id = _insert_worker(state, "w-cascade")
    _insert_worker_attribute(state, worker_id, "region", "us-east1")
    assert state._store.worker_attrs.get(worker_id) == {"region": AttributeValue("us-east1")}

    health = WorkerHealthTracker()
    health.register(worker_id, now_ms=1000)
    with db.write_transaction(state._db.sa_write_engine, threading.RLock()) as tx:
        writes_workers.remove_worker(
            tx,
            worker_id,
            health=health,
            worker_attrs=state._store.worker_attrs,
        )

    with state._db.read_snapshot() as q:
        row = q.fetchone(
            "SELECT key FROM worker_attributes WHERE worker_id = ?",
            (str(worker_id),),
        )
    assert row is None

    assert state._store.worker_attrs.get(worker_id) == {}
