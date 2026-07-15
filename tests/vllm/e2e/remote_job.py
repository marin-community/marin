# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for running integration-test callables on Marin clusters."""

from fray.iris_backend import FrayIrisClient
from fray.types import JobRequest, JobStatus
from iris.client import IrisClient
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from iris.test_util import wait_for_condition
from rigging.timing import Duration


def run_remote_test_job(
    iris_client: IrisClient,
    request: JobRequest,
    *,
    pending_timeout: float,
    runtime_timeout: float,
) -> None:
    """Submit a test job, bound its queue/runtime waits, and clean it up on interruption."""
    job = FrayIrisClient.from_iris_client(iris_client).submit(request, adopt_existing=False)
    try:
        task_id = JobName.from_string(job.job_id).task(0)
        wait_for_condition(
            lambda: iris_client.task_status(task_id).state
            not in (
                job_pb2.TASK_STATE_PENDING,
                job_pb2.TASK_STATE_ASSIGNED,
                job_pb2.TASK_STATE_BUILDING,
            ),
            timeout=Duration.from_seconds(pending_timeout),
            poll_interval=5,
        )
        job.wait(timeout=runtime_timeout, stream_logs=True)
    finally:
        if not JobStatus.finished(job.status()):
            job.terminate()
