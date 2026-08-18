# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import JobName

from levanter import distributed
from levanter.distributed import DistributedConfig, _square_brace_expand


class _RecordingIrisClient:
    def __init__(self) -> None:
        self.terminated_jobs: list[JobName] = []

    def terminate(self, job_id: JobName) -> None:
        self.terminated_jobs.append(job_id)


def test_square_brace_expand():
    custom_sequence = "node[001-004,007]suffix"
    expanded_nodes = _square_brace_expand(custom_sequence)
    assert expanded_nodes == ["node001suffix", "node002suffix", "node003suffix", "node004suffix", "node007suffix"]

    custom_sequence_2 = "prefix[001-002]node[005-006]suffix"
    expanded_nodes_2 = _square_brace_expand(custom_sequence_2)
    assert expanded_nodes_2 == [
        "prefix001node005suffix",
        "prefix001node006suffix",
        "prefix002node005suffix",
        "prefix002node006suffix",
    ]

    custom_sequence_3 = "node[1-11]suffix"
    expanded_nodes_3 = _square_brace_expand(custom_sequence_3)
    assert expanded_nodes_3 == [f"node{i}suffix" for i in range(1, 12)]

    custom_sequence_3 = "node[1-11,21]suffix"
    expanded_nodes_3 = _square_brace_expand(custom_sequence_3)
    assert expanded_nodes_3 == [f"node{i}suffix" for i in range(1, 12)] + ["node21suffix"]


@pytest.mark.parametrize(
    ("process_index", "last_exc", "should_terminate"),
    [
        pytest.param(0, None, True, id="rank-zero-clean-exit"),
        pytest.param(1, None, False, id="nonzero-rank"),
        pytest.param(0, RuntimeError("training failed"), False, id="rank-zero-failure"),
    ],
)
def test_iris_distributed_exit_terminates_job_on_clean_rank_zero(
    monkeypatch: pytest.MonkeyPatch,
    process_index: int,
    last_exc: BaseException | None,
    should_terminate: bool,
) -> None:
    job_info = JobInfo(task_id=JobName.from_wire("/test-user/training/0"))
    client = _RecordingIrisClient()
    exit_callbacks: list[tuple[Callable[..., Any], tuple[Any, ...]]] = []

    monkeypatch.setattr(distributed, "get_job_info", lambda: job_info)
    monkeypatch.setattr(distributed, "configure_megascale_from_iris", lambda: None)
    monkeypatch.setattr(distributed, "initialize_iris_jax", lambda: None)
    monkeypatch.setattr(distributed, "iris_ctx", lambda: SimpleNamespace(client=client), raising=False)
    monkeypatch.setattr(distributed.jax, "process_index", lambda: process_index)
    monkeypatch.setattr(sys, "last_exc", last_exc, raising=False)
    monkeypatch.setattr(
        distributed.atexit,
        "register",
        lambda callback, *args: exit_callbacks.append((callback, args)),
    )

    DistributedConfig().initialize()
    for callback, args in reversed(exit_callbacks):
        callback(*args)

    expected = [job_info.job_id] if should_terminate else []
    assert client.terminated_jobs == expected
