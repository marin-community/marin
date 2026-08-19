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
def iris_job(monkeypatch):
    """Install an unmarked completion, as JAX process 0 of an Iris job does."""
    completion = distributed._IrisJobCompletion(_RecordingIrisClient(), JobName.root("alice", "train"))
    monkeypatch.setattr(distributed, "_iris_job_completion", completion)
    return completion


def test_exit_without_a_success_mark_keeps_the_iris_job_open(iris_job):
    # A training failure exits through the Iris callable runner's `sys.exit(1)`, which
    # runs this hook. The job must stay open for Iris to fail it from the exit codes.
    distributed._complete_iris_job_after_successful_run()

    assert iris_job.client.completed == []


def test_exit_after_a_success_mark_completes_the_iris_job(iris_job):
    distributed.mark_run_succeeded()
    distributed._complete_iris_job_after_successful_run()

    assert iris_job.client.completed == [iris_job.job_id]


def test_success_mark_outside_an_iris_job_completes_nothing(monkeypatch):
    # Every local run and every non-zero JAX process takes this path.
    monkeypatch.setattr(distributed, "_iris_job_completion", None)

    distributed.mark_run_succeeded()
    distributed._complete_iris_job_after_successful_run()

    assert distributed._iris_job_completion is None
