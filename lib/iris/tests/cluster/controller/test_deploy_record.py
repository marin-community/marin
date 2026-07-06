# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the deploy record — the rollback pointer a controller restart writes."""

import json

from iris.cluster.controller.deploy_record import (
    DEPLOY_RECORD_FILENAME,
    DeployRecord,
    read_deploy_record,
    write_deploy_record,
)


def test_write_then_read_round_trips(tmp_path):
    state_dir = str(tmp_path)
    record = DeployRecord(
        current_image="ghcr.io/marin-community/iris-controller:new",
        previous_image="ghcr.io/marin-community/iris-controller:old",
        pre_deploy_checkpoint="gs://b/state/controller-state/1783357684695",
        recorded_at_ms=1783357700000,
    )

    write_deploy_record(state_dir, record)

    assert read_deploy_record(state_dir) == record
    assert (tmp_path / DEPLOY_RECORD_FILENAME).exists()


def test_read_absent_returns_none(tmp_path):
    assert read_deploy_record(str(tmp_path)) is None


def test_read_tolerates_missing_optional_fields(tmp_path):
    # A first deploy has no image to roll back to and no pre-deploy checkpoint.
    (tmp_path / DEPLOY_RECORD_FILENAME).write_text(json.dumps({"current_image": "img:first"}))

    record = read_deploy_record(str(tmp_path))

    assert record == DeployRecord(
        current_image="img:first", previous_image=None, pre_deploy_checkpoint=None, recorded_at_ms=0
    )


def test_read_malformed_json_returns_none(tmp_path):
    (tmp_path / DEPLOY_RECORD_FILENAME).write_text("{not json")

    assert read_deploy_record(str(tmp_path)) is None


def test_write_overwrites_existing_record(tmp_path):
    state_dir = str(tmp_path)
    write_deploy_record(state_dir, DeployRecord("img:a", None, None, 1))

    write_deploy_record(state_dir, DeployRecord("img:b", "img:a", "gs://b/cs/1", 2))

    assert read_deploy_record(state_dir) == DeployRecord("img:b", "img:a", "gs://b/cs/1", 2)
