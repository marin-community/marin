# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Endpoint naming for the standalone telltale server."""

import pytest
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import JobName
from iris.runtime import telltale
from iris.runtime.multigpu import IRIS_MULTIGPU_PROCESS_INDEX_ENV


@pytest.mark.parametrize(
    ("task_id", "process_index", "expected"),
    [
        ("/alice/train/worker/3", None, "telltale/worker/3"),
        ("/alice/train/0", None, "telltale/0"),
        ("/alice/train/0", "2", "telltale/0/2"),
    ],
)
def test_endpoint_name_identifies_the_process_within_its_namespace(monkeypatch, task_id, process_index, expected):
    monkeypatch.setattr(telltale, "get_job_info", lambda: JobInfo(task_id=JobName.from_wire(task_id)))
    if process_index is not None:
        monkeypatch.setenv(IRIS_MULTIGPU_PROCESS_INDEX_ENV, "2")

    assert telltale._endpoint_name() == expected
