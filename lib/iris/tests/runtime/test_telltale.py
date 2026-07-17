# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Endpoint naming for the standalone telltale server."""

import pytest
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import JobName
from iris.runtime import telltale
from iris.runtime.multigpu import IRIS_MULTIGPU_PROCESS_INDEX_ENV


@pytest.fixture
def job(monkeypatch):
    def _job(task_id: str) -> None:
        monkeypatch.setattr(telltale, "get_job_info", lambda: JobInfo(task_id=JobName.from_wire(task_id)))

    return _job


def test_endpoint_name_is_scoped_below_the_shared_namespace(job):
    """Every task in a hierarchy shares one namespace, so the name must not be bare.

    register() prefixes with the namespace (user/root-job), and endpoint
    resolution picks arbitrarily among live endpoints of the same name — a bare
    "telltale" would resolve to a random peer task.
    """
    job("/alice/train/worker/3")

    assert telltale._endpoint_name() == "telltale/worker/3"


def test_endpoint_name_for_a_task_directly_under_the_root_job(job):
    job("/alice/train/0")

    assert telltale._endpoint_name() == "telltale/0"


def test_endpoint_name_separates_processes_sharing_a_host(job, monkeypatch):
    """Multi-process hosts run one registry per child, so each needs its own name."""
    job("/alice/train/0")
    monkeypatch.setenv(IRIS_MULTIGPU_PROCESS_INDEX_ENV, "2")

    assert telltale._endpoint_name() == "telltale/0/2"
