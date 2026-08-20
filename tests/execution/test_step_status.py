# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

import marin.execution.step_status as step_status
import pytest
from iris.client.job_info import JobInfo, set_job_info
from iris.resources.names import JobName
from marin.execution.step_status import STATUS_RUNNING, StatusFile, should_run, worker_id


@pytest.fixture(autouse=True)
def _reset_job_info():
    set_job_info(None)
    yield
    set_job_info(None)


def test_should_run_repeats_active_iris_lock_owner(tmp_path: Path, caplog, monkeypatch):
    output_path = str(tmp_path / "active-lock")
    iris_task_id = "/larry/executor/0:2"
    set_job_info(JobInfo(task_id=JobName.from_wire("/larry/executor/0"), attempt_id=2))

    owner = StatusFile(output_path, worker_id())
    waiter = StatusFile(output_path, "waiting-worker")
    assert owner.try_acquire_lock()
    owner.write_status(STATUS_RUNNING)

    sleep_calls = 0

    def release_owner_after_second_log(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 2:
            owner.release_lock()
        elif sleep_calls > 2:
            raise AssertionError("Lock wait did not stop after the owner released the lock")

    monkeypatch.setattr(step_status, "_LOCK_WAIT_LOG_INTERVAL", 0)
    monkeypatch.setattr(step_status, "sleep", release_owner_after_second_log)

    with caplog.at_level(logging.INFO, logger="marin.execution.step_status"):
        assert should_run(waiter, "active-step")

    waiter.release_lock()
    owner_logs = [
        record
        for record in caplog.records
        if record.name == "marin.execution.step_status" and iris_task_id in record.getMessage()
    ]
    assert len(owner_logs) == 2
    assert all(record.levelno == logging.INFO for record in owner_logs)
    assert all("RUNNING" in record.getMessage() and "active lock" in record.getMessage() for record in owner_logs)
