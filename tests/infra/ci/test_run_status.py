# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from infra.ci.run_status import run_status


def test_run_status_records_running_and_success(tmp_path):
    path = tmp_path / "status" / "run.json"

    with run_status(str(path), marin_prefix="gs://marin-us-west4/tmp/ferry"):
        assert json.loads(path.read_text()) == {
            "marin_prefix": "gs://marin-us-west4/tmp/ferry",
            "status": "running",
        }

    assert json.loads(path.read_text()) == {
        "marin_prefix": "gs://marin-us-west4/tmp/ferry",
        "status": "succeeded",
    }


def test_run_status_records_failure_and_reraises(tmp_path):
    path = tmp_path / "run.json"

    with pytest.raises(RuntimeError, match="ferry failed"):
        with run_status(str(path), marin_prefix="gs://marin-us-east5/tmp/ferry"):
            raise RuntimeError("ferry failed")

    assert json.loads(path.read_text()) == {
        "marin_prefix": "gs://marin-us-east5/tmp/ferry",
        "status": "failed",
    }
