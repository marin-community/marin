# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iris.cluster.types import JobName

from levanter import distributed
from levanter.distributed import _square_brace_expand


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


class _RecordingIrisClient:
    """Records the jobs that the process-exit hook completes."""

    def __init__(self):
        self.completed: list[JobName] = []

    def complete(self, job_id: JobName) -> None:
        self.completed.append(job_id)


@pytest.fixture
def unmarked_run(monkeypatch):
    """Reset the success mark, so one test cannot arm another."""
    monkeypatch.setattr(distributed, "_run_succeeded", False)


def test_exit_without_a_success_mark_keeps_the_iris_job_open(unmarked_run):
    # A training failure exits through the Iris callable runner's `sys.exit(1)`, which
    # runs this hook. The job must stay open for Iris to fail it from the exit codes.
    client = _RecordingIrisClient()

    distributed._complete_iris_job_after_successful_run(client, JobName.root("alice", "train"))

    assert client.completed == []


def test_exit_after_a_success_mark_completes_the_iris_job(unmarked_run):
    client = _RecordingIrisClient()
    job_id = JobName.root("alice", "train")

    distributed.mark_run_succeeded()
    distributed._complete_iris_job_after_successful_run(client, job_id)

    assert client.completed == [job_id]
