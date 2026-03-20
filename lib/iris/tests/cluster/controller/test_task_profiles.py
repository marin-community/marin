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
from iris.time_utils import Timestamp


@pytest.fixture
def db(tmp_path: Path) -> ControllerDB:
    return ControllerDB(tmp_path / "test.db", auth_db_path=tmp_path / "auth.sqlite3")


def test_insert_and_retrieve_profile(db: ControllerDB) -> None:
    now = Timestamp.now()
    insert_task_profile(db, "task-1", b"profile-data-here", now)

    profiles = get_task_profiles(db, "task-1")
    assert len(profiles) == 1
    assert profiles[0][0] == b"profile-data-here"
    assert profiles[0][1].epoch_ms() == now.epoch_ms()


def test_profiles_ordered_newest_first(db: ControllerDB) -> None:
    for i in range(3):
        insert_task_profile(db, "task-1", f"profile-{i}".encode(), Timestamp.from_ms(1000 + i))

    profiles = get_task_profiles(db, "task-1")
    assert len(profiles) == 3
    assert profiles[0][0] == b"profile-2"
    assert profiles[2][0] == b"profile-0"


def test_cap_at_ten_profiles(db: ControllerDB) -> None:
    """The DB trigger should evict oldest profiles when count exceeds 10."""
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
    for i in range(12):
        insert_task_profile(db, "task-a", f"a-{i}".encode(), Timestamp.from_ms(1000 + i))
        insert_task_profile(db, "task-b", f"b-{i}".encode(), Timestamp.from_ms(1000 + i))

    assert len(get_task_profiles(db, "task-a")) == 10
    assert len(get_task_profiles(db, "task-b")) == 10


def test_empty_profiles(db: ControllerDB) -> None:
    profiles = get_task_profiles(db, "nonexistent")
    assert profiles == []
