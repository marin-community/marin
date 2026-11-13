# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import errno
import logging
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import marin.execution.step_status as step_status
import pytest
from iris.cluster.client.job_info import JobInfo, set_job_info
from iris.cluster.types import JobName
from marin.execution.step_status import STATUS_RUNNING, STATUS_SUCCESS, StatusFile, should_run, worker_id
from s3fs import S3FileSystem


@pytest.fixture(autouse=True)
def _reset_job_info():
    set_job_info(None)
    yield
    set_job_info(None)


def test_status_reads_latest_object_after_concurrent_overwrite(tmp_path: Path, monkeypatch):
    status = StatusFile(str(tmp_path / "shared-download"), "waiting-worker")
    status.path = "s3://test-bucket/shared-download/.executor_status"
    status.fs = S3FileSystem(anon=True, skip_instance_cache=True)

    async def s3_request(method, *args, **kwargs):
        if method == "head_object":
            # The writer finishes after the metadata lookup. Its old ETag is
            # invalid by the time the reader fetches the object's contents.
            return {"ContentLength": len(STATUS_RUNNING), "ETag": '"running"'}
        if method == "get_object":
            if kwargs.get("IfMatch") == '"running"':
                raise OSError(errno.EINVAL, "At least one of the pre-conditions you specified did not hold")
            return {"Body": Mock(read=AsyncMock(return_value=STATUS_SUCCESS.encode()))}
        raise AssertionError(f"Unexpected S3 operation: {method}")

    monkeypatch.setattr(status.fs, "_call_s3", s3_request)

    assert status.status == STATUS_SUCCESS


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
