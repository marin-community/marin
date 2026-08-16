# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys

import pytest
from iris.cluster.client.job_info import JobInfo, set_job_info
from iris.cluster.types import JobName, tpu_device
from iris.run_status import run_command_with_status, run_status
from rigging.timing import Timestamp


@pytest.fixture(autouse=True)
def _reset_job_info():
    set_job_info(None)
    yield
    set_job_info(None)


def test_run_status_persists_placement_and_terminal_state(tmp_path, monkeypatch):
    status_path = tmp_path / "nested" / "run.json"
    timestamps = iter((Timestamp.from_ms(1_000), Timestamp.from_ms(2_000), Timestamp.from_ms(3_000)))
    monkeypatch.setattr("iris.run_status.Timestamp.now", lambda: next(timestamps))
    set_job_info(
        JobInfo(
            task_id=JobName.from_wire("/alice/nightly/0"),
            attempt_id=2,
            worker_id="worker-1",
            worker_region="us-west4",
            worker_device=tpu_device("v5litepod-8"),
        )
    )

    with run_status(
        str(status_path),
        output_prefix="gs://marin-us-west4/tmp/nightly",
        requested_tpu_types=("v5litepod-8", "v6e-8"),
    ):
        running = json.loads(status_path.read_text())
        assert running["status"] == "running"
        assert running["finished_at"] is None

    succeeded = json.loads(status_path.read_text())
    assert succeeded["status"] == "succeeded"
    assert succeeded["marin_prefix"] == "gs://marin-us-west4/tmp/nightly"
    assert succeeded["requested_tpu_types"] == ["v5litepod-8", "v6e-8"]
    assert succeeded["worker_region"] == "us-west4"
    assert succeeded["worker_device"]["tpu"]["variant"] == "v5litepod-8"
    assert succeeded["job_id"] == "/alice/nightly"
    assert succeeded["attempt_id"] == 2
    assert succeeded["exit_code"] == 0
    assert succeeded["started_at"] == "1970-01-01T00:00:01+00:00"
    assert succeeded["updated_at"] == "1970-01-01T00:00:03+00:00"
    assert succeeded["finished_at"] == "1970-01-01T00:00:03+00:00"


def test_run_command_with_status_preserves_nonzero_exit(tmp_path):
    status_path = tmp_path / "run.json"

    with pytest.raises(SystemExit) as exc_info:
        run_command_with_status(
            [sys.executable, "-c", "raise SystemExit(7)"],
            str(status_path),
            "gs://marin-us-east5/tmp/nightly",
            ("v6e-4",),
        )

    assert exc_info.value.code == 7
    failed = json.loads(status_path.read_text())
    assert failed["status"] == "failed"
    assert failed["exit_code"] == 7
    assert failed["error_type"] == "SystemExit"
