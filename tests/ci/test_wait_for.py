# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.types import JobName
from iris.rpc import job_pb2

from scripts.ci.wait_for import BackoffConfig, EventKind, EventSpec, IrisJobSource, select_loop


class FakeIrisJobClient:
    def __init__(self, states: Iterator[job_pb2.JobState | ConnectError]):
        self._states = states

    def get_job_states_once(self, job_ids: list[JobName]) -> dict[str, job_pb2.JobState]:
        state = next(self._states)
        if isinstance(state, ConnectError):
            raise state
        return {job_ids[0].to_wire(): state}


class MissingIrisJobClient:
    def get_job_states_once(self, job_ids: list[JobName]) -> dict[str, job_pb2.JobState]:
        return {}


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
    client = FakeIrisJobClient(iter([job_pb2.JOB_STATE_RUNNING, terminal_state]))
    source = IrisJobSource(spec, client)

    assert source.check() is None
    assert source.check() == {"state": expected_state}


def test_iris_job_source_retries_transient_rpc_error_on_selector_backoff() -> None:
    spec = EventSpec(EventKind.IRIS_JOB, "/alice/training-run", "iris.job /alice/training-run")
    client = FakeIrisJobClient(
        iter(
            [
                ConnectError(Code.UNAVAILABLE, "controller unavailable"),
                job_pb2.JOB_STATE_SUCCEEDED,
            ]
        )
    )
    source = IrisJobSource(spec, client)

    result = select_loop(
        [source],
        deadline=None,
        backoff=BackoffConfig(initial=1e-9, maximum=1e-9, factor=2.0, jitter=0.0),
    )

    assert result["result"] == {"state": "succeeded"}


def test_iris_job_source_fails_fast_when_job_is_missing() -> None:
    spec = EventSpec(EventKind.IRIS_JOB, "/alice/missing-run", "iris.job /alice/missing-run")
    source = IrisJobSource(spec, MissingIrisJobClient())

    with pytest.raises(ConnectError) as exc_info:
        select_loop(
            [source],
            deadline=None,
            backoff=BackoffConfig(initial=1.0, maximum=1.0, factor=2.0, jitter=0.0),
        )

    assert exc_info.value.code is Code.NOT_FOUND
