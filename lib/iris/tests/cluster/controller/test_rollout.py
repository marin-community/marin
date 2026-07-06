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


def test_read_malformed_json_returns_none(tmp_path):
    # A corrupt record must not crash the controller on boot — it starts from the
    # local DB instead.
    (tmp_path / ROLLOUT_RECORD_FILENAME).write_text("{not json")

    assert read_rollout_record(str(tmp_path)) is None
