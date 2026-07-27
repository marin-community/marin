# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import threading
import time

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.types import JobName
from iris.rpc import job_pb2

from scripts.ci.wait_for import BackoffConfig, EventKind, EventSpec, IrisJobSource, PollSource, Source, select_loop


class ReleaseSource(Source):
    def __init__(self, release: threading.Event):
        super().__init__(EventSpec(EventKind.POLL, "release", "poll release"))
        self.release = release

    def check(self) -> dict | None:
        self.release.set()
        return None


@pytest.mark.parametrize(
    ("terminal_state", "expected_state"),
    [
        (job_pb2.JOB_STATE_SUCCEEDED, "succeeded"),
        (job_pb2.JOB_STATE_FAILED, "failed"),
    ],
)
def test_iris_job_source_fires_when_job_reaches_terminal_state(
    terminal_state: job_pb2.JobState, expected_state: str
) -> None:
    spec = EventSpec(EventKind.IRIS_JOB, "/alice/training-run", "iris.job /alice/training-run")
    source = IrisJobSource(
        spec,
        lambda job_id: job_pb2.JobStatus(job_id=job_id.to_wire(), state=terminal_state),
    )

    result = select_loop(
        [source],
        deadline=time.monotonic() + 1.0,
        backoff=BackoffConfig(initial=1e-9, maximum=1e-9, factor=2.0, jitter=0.0),
    )

    assert result["result"] == {"state": expected_state}


def test_iris_job_source_does_not_block_other_sources() -> None:
    spec = EventSpec(EventKind.IRIS_JOB, "/alice/training-run", "iris.job /alice/training-run")
    started = threading.Event()
    release = threading.Event()

    def blocking_waiter(job_id: JobName) -> job_pb2.JobStatus:
        started.set()
        if not release.wait(timeout=5.0):
            raise TimeoutError(f"Test did not release wait for {job_id}")
        return job_pb2.JobStatus(job_id=job_id.to_wire(), state=job_pb2.JOB_STATE_SUCCEEDED)

    iris_source = IrisJobSource(spec, blocking_waiter)
    poll_source = PollSource(
        EventSpec(EventKind.POLL, "true", "poll true"),
        poll_timeout=1.0,
    )

    assert started.wait(timeout=1.0)
    try:
        result = select_loop(
            [iris_source, poll_source],
            deadline=time.monotonic() + 1.0,
            backoff=BackoffConfig(initial=1e-9, maximum=1e-9, factor=2.0, jitter=0.0),
        )
    finally:
        release.set()

    assert result["event"] == EventKind.POLL


def test_iris_job_source_completion_wakes_selector_before_backoff() -> None:
    spec = EventSpec(EventKind.IRIS_JOB, "/alice/training-run", "iris.job /alice/training-run")
    started = threading.Event()
    release = threading.Event()

    def blocking_waiter(job_id: JobName) -> job_pb2.JobStatus:
        started.set()
        if not release.wait(timeout=5.0):
            raise TimeoutError(f"Test did not release wait for {job_id}")
        return job_pb2.JobStatus(job_id=job_id.to_wire(), state=job_pb2.JOB_STATE_SUCCEEDED)

    iris_source = IrisJobSource(spec, blocking_waiter)

    assert started.wait(timeout=1.0)
    result = select_loop(
        [iris_source, ReleaseSource(release)],
        deadline=time.monotonic() + 1.0,
        backoff=BackoffConfig(initial=60.0, maximum=60.0, factor=2.0, jitter=0.0),
    )

    assert result["event"] == EventKind.IRIS_JOB
    assert result["result"] == {"state": "succeeded"}


def test_iris_job_source_fails_fast_when_job_is_missing() -> None:
    spec = EventSpec(EventKind.IRIS_JOB, "/alice/missing-run", "iris.job /alice/missing-run")

    def missing_waiter(job_id: JobName) -> job_pb2.JobStatus:
        raise ConnectError(Code.NOT_FOUND, f"Job {job_id} not found")

    source = IrisJobSource(spec, missing_waiter)

    with pytest.raises(ConnectError) as exc_info:
        select_loop(
            [source],
            deadline=time.monotonic() + 1.0,
            backoff=BackoffConfig(initial=1e-9, maximum=1e-9, factor=2.0, jitter=0.0),
        )

    assert exc_info.value.code is Code.NOT_FOUND
