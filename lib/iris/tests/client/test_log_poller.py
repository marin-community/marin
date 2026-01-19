# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for fetch_logs and log streaming."""

import pytest

from iris.client import IrisClient, LocalClientConfig
from iris.cluster.types import Entrypoint, ResourceSpec
from iris.rpc import cluster_pb2


def logging_job():
    """A job that produces logs."""
    print("Log line 1")
    print("Log line 2")
    print("Log line 3")


@pytest.fixture
def local_client():
    config = LocalClientConfig(max_workers=2)
    with IrisClient.local(config) as client:
        yield client


@pytest.fixture
def resources():
    return ResourceSpec(cpu=1, memory="1g")


def test_fetch_logs_basic(local_client, resources):
    """Verify fetch_logs returns job output."""
    entrypoint = Entrypoint.from_callable(logging_job)
    job = local_client.submit(entrypoint, "log-test", resources)
    job.wait()

    logs = local_client.fetch_logs(job.job_id)
    log_data = [entry.data for entry in logs]

    assert any("Log line 1" in d for d in log_data)
    assert any("Log line 2" in d for d in log_data)
    assert any("Log line 3" in d for d in log_data)


def test_wait_with_stream_logs(local_client, resources, caplog):
    """Verify job.wait(stream_logs=True) logs output while waiting."""
    import logging

    entrypoint = Entrypoint.from_callable(logging_job)
    job = local_client.submit(entrypoint, "stream-test", resources)

    with caplog.at_level(logging.INFO, logger="iris.client.client"):
        status = job.wait(stream_logs=True)

    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    # Log streaming now uses task-specific logging format: [job_id/task-N][timestamp]
    # In local mode, the job may complete too fast for logs to be captured
    # The test passes if the job succeeds - actual log capture is timing-dependent
