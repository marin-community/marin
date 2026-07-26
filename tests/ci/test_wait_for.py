# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest

from iris.cluster.types import JobName
from iris.rpc import job_pb2
from scripts.ci.wait_for import EventKind, EventSpec, IrisJobSource


class FakeIrisJobClient:
    def __init__(self, states: Iterator[int]):
        self._states = states

    def job_state(self, job_id: JobName) -> int:
        return next(self._states)


@pytest.mark.parametrize(
    ("terminal_state", "expected_state"),
    [
        (job_pb2.JOB_STATE_SUCCEEDED, "succeeded"),
        (job_pb2.JOB_STATE_FAILED, "failed"),
    ],
)
def test_iris_job_source_fires_when_job_reaches_terminal_state(terminal_state: int, expected_state: str) -> None:
    spec = EventSpec(EventKind.IRIS_JOB, "/alice/training-run", "iris.job /alice/training-run")
    client = FakeIrisJobClient(iter([job_pb2.JOB_STATE_RUNNING, terminal_state]))
    source = IrisJobSource(spec, client)

    assert source.check() is None
    assert source.check() == {"state": expected_state}
