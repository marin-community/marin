# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for task profile DB storage and the controller profile loop."""

from pathlib import Path

import pytest

from iris.cluster.controller.db import (
    ControllerDB,
    get_task_profiles,
    insert_task_profile,
)
from iris.rpc import job_pb2
from rigging.timing import Timestamp


@pytest.fixture
def db(tmp_path: Path) -> ControllerDB:
    return ControllerDB(db_dir=tmp_path)


def _ensure_task(db: ControllerDB, task_id: str) -> None:
    """Insert a minimal user + job + task so FK constraints on task_profiles are satisfied."""
    job_id = task_id.rsplit("/", 1)[0] if "/" in task_id else f"_job_for_{task_id}"
    user_id = "test-user"
    db.execute(
        "INSERT OR IGNORE INTO users (user_id, created_at_ms, role) VALUES (?, 0, 'user')",
        (user_id,),
    )
    db.execute(
        "INSERT OR IGNORE INTO jobs "
        "(job_id, user_id, state, depth, root_job_id, submitted_at_ms, "
        "root_submitted_at_ms, num_tasks, is_reservation_holder) "
        "VALUES (?, ?, ?, 1, ?, 0, 0, 1, 0)",
        (job_id, user_id, job_pb2.JOB_STATE_RUNNING, job_id),
    )
    db.execute(
        "INSERT OR IGNORE INTO job_config (job_id, name) VALUES (?, ?)",
        (job_id, job_id),
    )
    db.execute(
        "INSERT OR IGNORE INTO tasks "
        "(task_id, job_id, state, task_index, submitted_at_ms, "
        "max_retries_failure, max_retries_preemption, failure_count, preemption_count, "
        "current_attempt_id, priority_neg_depth, priority_root_submitted_ms, priority_insertion) "
        "VALUES (?, ?, ?, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)",
        (task_id, job_id, job_pb2.TASK_STATE_RUNNING),
    )


def test_insert_and_retrieve_profile(db: ControllerDB) -> None:
    now = Timestamp.now()
    _ensure_task(db, "task-1")
    insert_task_profile(db, "task-1", b"profile-data-here", now)

    profiles = get_task_profiles(db, "task-1")
    assert len(profiles) == 1
    assert profiles[0][0] == b"profile-data-here"
    assert profiles[0][1].epoch_ms() == now.epoch_ms()
    assert profiles[0][2] == "cpu"


def test_profiles_ordered_newest_first(db: ControllerDB) -> None:
    _ensure_task(db, "task-1")
    for i in range(3):
        insert_task_profile(db, "task-1", f"profile-{i}".encode(), Timestamp.from_ms(1000 + i))

    profiles = get_task_profiles(db, "task-1")
    assert len(profiles) == 3
    assert profiles[0][0] == b"profile-2"
    assert profiles[2][0] == b"profile-0"


def test_cap_at_ten_profiles(db: ControllerDB) -> None:
    """The DB trigger should evict oldest profiles when count exceeds 10."""
    _ensure_task(db, "task-1")
    for i in range(15):
        insert_task_profile(db, "task-1", f"profile-{i}".encode(), Timestamp.from_ms(1000 + i))

    profiles = get_task_profiles(db, "task-1")
    assert len(profiles) == 10
    # Newest 10 should be kept (profiles 5..14)
    data_values = [p[0] for p in profiles]
    assert data_values[0] == b"profile-14"
    assert data_values[-1] == b"profile-5"


def test_cap_is_per_task(db: ControllerDB) -> None:
    """Profiles for different tasks are capped independently."""
    _ensure_task(db, "task-a")
    _ensure_task(db, "task-b")
    for i in range(12):
        insert_task_profile(db, "task-a", f"a-{i}".encode(), Timestamp.from_ms(1000 + i))
        insert_task_profile(db, "task-b", f"b-{i}".encode(), Timestamp.from_ms(1000 + i))

    assert len(get_task_profiles(db, "task-a")) == 10
    assert len(get_task_profiles(db, "task-b")) == 10


def test_empty_profiles(db: ControllerDB) -> None:
    profiles = get_task_profiles(db, "nonexistent")
    assert profiles == []


def test_cap_is_per_task_and_kind(db: ControllerDB) -> None:
    """Cap trigger retains 10 per (task_id, profile_kind)."""
    _ensure_task(db, "task-1")
    for i in range(12):
        insert_task_profile(db, "task-1", f"cpu-{i}".encode(), Timestamp.from_ms(1000 + i), profile_kind="cpu")
        insert_task_profile(db, "task-1", f"mem-{i}".encode(), Timestamp.from_ms(1000 + i), profile_kind="memory")

    cpu_profiles = get_task_profiles(db, "task-1", profile_kind="cpu")
    mem_profiles = get_task_profiles(db, "task-1", profile_kind="memory")
    assert len(cpu_profiles) == 10
    assert len(mem_profiles) == 10
    # Total should be 20 (10 cpu + 10 memory)
    all_profiles = get_task_profiles(db, "task-1")
    assert len(all_profiles) == 20
