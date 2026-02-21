# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for fray v2 Client protocol, LocalClient, and wait_all."""

import threading
import time

import pytest

from fray.v2 import (
    Entrypoint,
    JobFailed,
    JobRequest,
    JobStatus,
    LocalClient,
    wait_all,
)
from fray.v2.client import JobAlreadyExists


@pytest.fixture
def client():
    c = LocalClient(max_threads=4)
    yield c
    c.shutdown(wait=True)


def _noop():
    pass


def _sleep_then_succeed():
    time.sleep(0.1)


def _fail():
    raise RuntimeError("intentional failure")


def _return_value():
    return 42


def test_submit_callable_succeeds(client: LocalClient):
    handle = client.submit(JobRequest(name="ok", entrypoint=Entrypoint.from_callable(_noop)))
    status = handle.wait()
    assert status == JobStatus.SUCCEEDED


def test_submit_callable_failure(client: LocalClient):
    handle = client.submit(JobRequest(name="fail", entrypoint=Entrypoint.from_callable(_fail)))
    status = handle.wait(raise_on_failure=False)
    assert status == JobStatus.FAILED


def test_submit_callable_failure_raises(client: LocalClient):
    handle = client.submit(JobRequest(name="fail", entrypoint=Entrypoint.from_callable(_fail)))
    with pytest.raises(RuntimeError, match="intentional failure"):
        handle.wait(raise_on_failure=True)


def test_job_id_contains_name(client: LocalClient):
    handle = client.submit(JobRequest(name="my-job", entrypoint=Entrypoint.from_callable(_noop)))
    assert "my-job" in handle.job_id
    handle.wait()


def test_status_transitions(client: LocalClient):
    handle = client.submit(JobRequest(name="slow", entrypoint=Entrypoint.from_callable(_sleep_then_succeed)))
    # Should be running or pending initially
    initial = handle.status()
    assert initial in (JobStatus.PENDING, JobStatus.RUNNING)
    handle.wait()
    assert handle.status() == JobStatus.SUCCEEDED


def test_terminate():
    """Terminate marks a job as STOPPED even if the underlying thread is still running."""
    stop = threading.Event()

    def hang():
        stop.wait(10)

    c = LocalClient(max_threads=4)
    handle = c.submit(JobRequest(name="hang", entrypoint=Entrypoint.from_callable(hang)))
    time.sleep(0.05)
    handle.terminate()
    assert handle.status() == JobStatus.STOPPED
    stop.set()  # unblock the thread so the executor can shut down
    c.shutdown(wait=True)


def test_wait_all_all_succeed(client: LocalClient):
    handles = [client.submit(JobRequest(name=f"ok-{i}", entrypoint=Entrypoint.from_callable(_noop))) for i in range(3)]
    statuses = wait_all(handles)
    assert all(s == JobStatus.SUCCEEDED for s in statuses)


def test_wait_all_mixed_failure(client: LocalClient):
    h_ok = client.submit(JobRequest(name="ok", entrypoint=Entrypoint.from_callable(_noop)))
    h_fail = client.submit(JobRequest(name="fail", entrypoint=Entrypoint.from_callable(_fail)))
    with pytest.raises(JobFailed):
        wait_all([h_ok, h_fail], raise_on_failure=True)


def test_wait_all_no_raise(client: LocalClient):
    h_ok = client.submit(JobRequest(name="ok", entrypoint=Entrypoint.from_callable(_noop)))
    h_fail = client.submit(JobRequest(name="fail", entrypoint=Entrypoint.from_callable(_fail)))
    statuses = wait_all([h_ok, h_fail], raise_on_failure=False)
    assert statuses[0] == JobStatus.SUCCEEDED
    assert statuses[1] == JobStatus.FAILED


def test_wait_all_empty():
    assert wait_all([]) == []


def test_wait_all_timeout():
    stop = threading.Event()

    def hang():
        stop.wait(10)

    c = LocalClient(max_threads=4)
    handle = c.submit(JobRequest(name="hang", entrypoint=Entrypoint.from_callable(hang)))
    with pytest.raises(TimeoutError):
        wait_all([handle], timeout=0.2)
    handle.terminate()
    stop.set()
    c.shutdown(wait=True)


def test_job_already_exists_carries_handle():
    """JobAlreadyExists can carry a handle for the caller to adopt."""

    class FakeHandle:
        @property
        def job_id(self) -> str:
            return "existing-job"

        def wait(self, timeout=None, *, raise_on_failure=True):
            return JobStatus.SUCCEEDED

        def status(self):
            return JobStatus.RUNNING

        def terminate(self):
            pass

    handle = FakeHandle()
    exc = JobAlreadyExists("my-job", handle=handle)
    assert exc.job_name == "my-job"
    assert exc.handle is handle
    assert "my-job" in str(exc)


def test_job_already_exists_without_handle():
    """JobAlreadyExists without a handle should still be raisable."""
    exc = JobAlreadyExists("orphan-job")
    assert exc.handle is None
    assert exc.job_name == "orphan-job"


def test_submit_with_adopt_existing_false_default(client: LocalClient):
    """By default, adopt_existing=True and LocalClient doesn't enforce uniqueness."""
    # LocalClient doesn't track job names, so calling submit twice with the same name
    # creates two separate jobs (no exception)
    h1 = client.submit(JobRequest(name="same-name", entrypoint=Entrypoint.from_callable(_noop)))
    h2 = client.submit(JobRequest(name="same-name", entrypoint=Entrypoint.from_callable(_noop)))
    # Both jobs should succeed independently
    assert h1.wait() == JobStatus.SUCCEEDED
    assert h2.wait() == JobStatus.SUCCEEDED
    assert h1.job_id != h2.job_id  # Different job IDs


def test_submit_with_adopt_existing_true(client: LocalClient):
    """When adopt_existing=True, LocalClient still doesn't enforce uniqueness."""
    # LocalClient doesn't track job names, so adopt_existing has no effect
    h1 = client.submit(JobRequest(name="same-name", entrypoint=Entrypoint.from_callable(_noop)), adopt_existing=True)
    h2 = client.submit(JobRequest(name="same-name", entrypoint=Entrypoint.from_callable(_noop)), adopt_existing=True)
    # Both jobs should succeed independently
    assert h1.wait() == JobStatus.SUCCEEDED
    assert h2.wait() == JobStatus.SUCCEEDED
    assert h1.job_id != h2.job_id  # Different job IDs
