# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for controller startup state selection and rollback."""

import sqlite3
from pathlib import Path

import pytest
from iris.cluster.controller.main import prepare_controller_state
from iris.cluster.controller.persistence.checkpoint import write_checkpoint
from iris.cluster.controller.persistence.database import ControllerDB
from iris.cluster.controller.persistence.writes import meta_value_set
from iris.cluster.controller.rollout import (
    RolloutPhase,
    RolloutRecord,
    read_rollout_record,
    write_rollout_record,
)
from rigging.filesystem import prefix_join

_MARKER_KEY = "test_rollout_marker"


def _remote_state(tmp_path: Path) -> str:
    path = tmp_path / "remote"
    path.mkdir(exist_ok=True)
    return f"file://{path}"


def _seed_database(db_dir: Path, marker: int) -> None:
    db = ControllerDB(db_dir=db_dir)
    with db.transaction() as tx:
        meta_value_set(tx, _MARKER_KEY, marker)
    db.close()


def _database_marker(db_dir: Path) -> int:
    connection = sqlite3.connect(db_dir / ControllerDB.DB_FILENAME)
    try:
        row = connection.execute("SELECT value FROM meta WHERE key = ?", (_MARKER_KEY,)).fetchone()
    finally:
        connection.close()
    assert row is not None
    return int(row[0])


def _checkpoint(db_dir: Path, remote_state_dir: str, marker: int) -> str:
    _seed_database(db_dir, marker)
    db = ControllerDB(db_dir=db_dir)
    try:
        path, _ = write_checkpoint(db, remote_state_dir)
    finally:
        db.close()
    return path


def test_prepare_controller_state_fresh_removes_existing_database(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    _seed_database(db_dir, 1)

    prepare_controller_state(db_dir, _remote_state(tmp_path), fresh=True, checkpoint_path=None)

    assert db_dir.exists()
    assert not (db_dir / ControllerDB.DB_FILENAME).exists()
    assert not (db_dir / ControllerDB.AUTH_DB_FILENAME).exists()


def test_prepare_controller_state_applies_requested_rollback_once(tmp_path: Path) -> None:
    remote_state_dir = _remote_state(tmp_path)
    checkpoint = _checkpoint(tmp_path / "checkpoint-source", remote_state_dir, marker=1)
    db_dir = tmp_path / "db"
    _seed_database(db_dir, 2)
    write_rollout_record(
        remote_state_dir,
        RolloutRecord(
            phase=RolloutPhase.ROLLBACK_REQUESTED,
            image="controller:old",
            previous_image="controller:new",
            rollback_checkpoint=checkpoint,
        ),
    )

    prepare_controller_state(db_dir, remote_state_dir, fresh=False, checkpoint_path=None)

    assert _database_marker(db_dir) == 1
    record = read_rollout_record(remote_state_dir)
    assert record is not None
    assert record.phase is RolloutPhase.ROLLED_BACK
    assert record.previous_image is None
    assert record.rollback_checkpoint is None


def test_prepare_controller_state_rollback_without_checkpoint_reuses_local_database(tmp_path: Path) -> None:
    remote_state_dir = _remote_state(tmp_path)
    db_dir = tmp_path / "db"
    _seed_database(db_dir, 7)
    requested = RolloutRecord(
        phase=RolloutPhase.ROLLBACK_REQUESTED,
        image="controller:old",
        rollback_checkpoint=None,
    )
    write_rollout_record(remote_state_dir, requested)

    prepare_controller_state(db_dir, remote_state_dir, fresh=False, checkpoint_path=None)

    assert _database_marker(db_dir) == 7
    assert read_rollout_record(remote_state_dir) == requested


def test_prepare_controller_state_reuses_local_database_at_latest_checkpoint(tmp_path: Path) -> None:
    remote_state_dir = _remote_state(tmp_path)
    db_dir = tmp_path / "db"
    _checkpoint(db_dir, remote_state_dir, marker=11)

    prepare_controller_state(db_dir, remote_state_dir, fresh=False, checkpoint_path=None)

    assert _database_marker(db_dir) == 11


def test_prepare_controller_state_restores_latest_when_local_ancestry_differs(tmp_path: Path) -> None:
    remote_state_dir = _remote_state(tmp_path)
    _checkpoint(tmp_path / "checkpoint-source", remote_state_dir, marker=20)
    db_dir = tmp_path / "db"
    _seed_database(db_dir, 10)

    prepare_controller_state(db_dir, remote_state_dir, fresh=False, checkpoint_path=None)

    assert _database_marker(db_dir) == 20


def test_prepare_controller_state_restores_requested_checkpoint_when_local_absent(tmp_path: Path) -> None:
    remote_state_dir = _remote_state(tmp_path)
    checkpoint = _checkpoint(tmp_path / "checkpoint-source", remote_state_dir, marker=30)
    db_dir = tmp_path / "db"

    prepare_controller_state(db_dir, remote_state_dir, fresh=False, checkpoint_path=checkpoint)

    assert _database_marker(db_dir) == 30


def test_prepare_controller_state_rejects_missing_checkpoint(tmp_path: Path) -> None:
    remote_state_dir = _remote_state(tmp_path)
    checkpoint_root = prefix_join(remote_state_dir, "controller-state")
    missing = prefix_join(checkpoint_root, "999")

    with pytest.raises(ValueError, match="Checkpoint not found"):
        prepare_controller_state(tmp_path / "db", remote_state_dir, fresh=False, checkpoint_path=missing)


def test_prepare_controller_state_rejects_checkpoint_without_numeric_epoch(tmp_path: Path) -> None:
    remote_state_dir = _remote_state(tmp_path)
    checkpoint_root = prefix_join(remote_state_dir, "controller-state")
    malformed = prefix_join(checkpoint_root, "not-an-epoch")

    with pytest.raises(ValueError, match="numeric epoch"):
        prepare_controller_state(tmp_path / "db", remote_state_dir, fresh=False, checkpoint_path=malformed)


def test_prepare_controller_state_quarantines_corrupt_database_before_restore(tmp_path: Path) -> None:
    remote_state_dir = _remote_state(tmp_path)
    _checkpoint(tmp_path / "checkpoint-source", remote_state_dir, marker=40)
    db_dir = tmp_path / "db"
    _seed_database(db_dir, 10)
    corrupt_bytes = b"not a sqlite database"
    (db_dir / ControllerDB.DB_FILENAME).write_bytes(corrupt_bytes)

    prepare_controller_state(db_dir, remote_state_dir, fresh=False, checkpoint_path=None)

    assert _database_marker(db_dir) == 40
    quarantined = list(tmp_path.glob("db.corrupt-*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / ControllerDB.DB_FILENAME).read_bytes() == corrupt_bytes
