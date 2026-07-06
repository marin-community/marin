# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the rollout record — the deploy/rollback pointer a controller restart writes."""

from iris.cluster.controller.rollout import (
    ROLLOUT_RECORD_FILENAME,
    RolloutPhase,
    RolloutRecord,
    read_rollout_record,
    write_rollout_record,
)


def test_write_then_read_round_trips(tmp_path):
    state_dir = str(tmp_path)
    record = RolloutRecord(
        phase=RolloutPhase.COMMITTED,
        image="ghcr.io/marin-community/iris-controller:new",
        previous_image="ghcr.io/marin-community/iris-controller:old",
        rollback_checkpoint="gs://b/state/controller-state/1783357684695",
        updated_at_ms=1783357700000,
    )

    write_rollout_record(state_dir, record)

    assert read_rollout_record(state_dir) == record
    assert (tmp_path / ROLLOUT_RECORD_FILENAME).exists()


def test_read_absent_returns_none(tmp_path):
    assert read_rollout_record(str(tmp_path)) is None


def test_read_tolerates_missing_optional_fields(tmp_path):
    # A first deploy has no image to roll back to and no pre-deploy checkpoint.
    (tmp_path / ROLLOUT_RECORD_FILENAME).write_text('{"phase": "committed", "image": "img:first"}')

    record = read_rollout_record(str(tmp_path))

    assert record == RolloutRecord(phase=RolloutPhase.COMMITTED, image="img:first")
    assert record.previous_image is None
    assert record.rollback_checkpoint is None
    assert record.updated_at_ms == 0


def test_read_malformed_json_returns_none(tmp_path):
    (tmp_path / ROLLOUT_RECORD_FILENAME).write_text("{not json")

    assert read_rollout_record(str(tmp_path)) is None


def test_read_unknown_phase_returns_none(tmp_path):
    (tmp_path / ROLLOUT_RECORD_FILENAME).write_text('{"phase": "banana", "image": "img:x"}')

    assert read_rollout_record(str(tmp_path)) is None


def test_read_ignores_extra_fields(tmp_path):
    (tmp_path / ROLLOUT_RECORD_FILENAME).write_text('{"phase": "pending", "image": "img:x", "legacy_field": 42}')

    assert read_rollout_record(str(tmp_path)) == RolloutRecord(phase=RolloutPhase.PENDING, image="img:x")


def test_write_overwrites_existing_record(tmp_path):
    state_dir = str(tmp_path)
    write_rollout_record(state_dir, RolloutRecord(phase=RolloutPhase.PENDING, image="img:a"))

    write_rollout_record(
        state_dir,
        RolloutRecord(
            phase=RolloutPhase.COMMITTED, image="img:b", previous_image="img:a", rollback_checkpoint="gs://b/cs/1"
        ),
    )

    got = read_rollout_record(state_dir)
    assert got.phase is RolloutPhase.COMMITTED
    assert got.image == "img:b"
    assert got.previous_image == "img:a"
