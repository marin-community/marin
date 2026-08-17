# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public workload API behavior over the deployed controller wire."""

import operator

import pytest
from iris.client import IrisClient, Job, JobFailedError, JobState, TaskState
from iris.client.workload import DeviceKind
from iris.cluster.types import JobName
from iris.rpc import job_pb2


def _task_proto() -> job_pb2.TaskStatus:
    task = job_pb2.TaskStatus(
        task_id="/alice/train/0",
        state=job_pb2.TASK_STATE_RUNNING,
        worker_id="worker-1",
        current_attempt_id=2,
        ports={"http": 8080},
        backend_id="gpu",
        cluster="local",
    )
    task.submitted_at.epoch_ms = 1_000
    task.attempts.add(
        attempt_id=2,
        attempt_uid="attempt-uid",
        state=job_pb2.TASK_STATE_RUNNING,
        worker_id="worker-1",
        pod_name="train-0",
    )
    return task


def _job_proto(state: int = job_pb2.JOB_STATE_RUNNING) -> job_pb2.JobStatus:
    job = job_pb2.JobStatus(
        job_id="/alice/train",
        state=state,
        task_count=1,
        task_state_counts={"running": 1},
        tasks=[_task_proto()],
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=4_000,
            memory_bytes=16_000_000_000,
            disk_bytes=20_000_000_000,
            device=job_pb2.DeviceConfig(gpu=job_pb2.GpuDevice(variant="H100", count=8)),
        ),
    )
    job.submitted_at.epoch_ms = 1_000
    return job


class WorkloadCluster:
    def __init__(self, job: job_pb2.JobStatus | None = None):
        self.job = job or _job_proto()
        self.last_query = None

    def get_job_status(self, _job_id):
        return self.job

    def get_job_states(self, job_ids):
        return {str(job_id): self.job.state for job_id in job_ids}

    def list_jobs(self, *, query, **_kwargs):
        self.last_query = query
        return [self.job]

    def get_task_status(self, _task_id):
        return self.job.tasks[0]

    def list_tasks(self, _job_id):
        return list(self.job.tasks)

    def wait_for_job(self, _job_id, _timeout, _poll_interval):
        return self.job


def test_public_workload_handles_return_native_snapshots():
    cluster = WorkloadCluster()
    client = IrisClient(cluster)

    job = Job(client, JobName.from_wire("/alice/train"))
    status = job.status()
    task = job.tasks()[0]
    task_status = task.status()

    assert status.state is JobState.RUNNING
    assert status.job_id == JobName.from_wire("/alice/train")
    assert status.resources.device is not None
    assert status.resources.device.kind is DeviceKind.GPU
    assert status.resources.device.variant == "H100"
    assert status.task_state_counts == {TaskState.RUNNING: 1}

    assert task_status.state is TaskState.RUNNING
    assert task_status.current_attempt_number == 2
    assert task_status.attempts[0].attempt_uid == "attempt-uid"
    with pytest.raises(TypeError):
        operator.setitem(task_status.ports, "debug", 9000)


def test_list_jobs_filters_on_wire_and_returns_native_values():
    cluster = WorkloadCluster()
    client = IrisClient(cluster)

    jobs = client.list_jobs(state=JobState.RUNNING, prefix="/alice/train", limit=1)

    assert jobs[0].state is JobState.RUNNING
    assert cluster.last_query.state_filter == "running"
    assert cluster.last_query.job_id_prefix == "/alice/train"


def test_wait_failure_exposes_native_terminal_snapshot():
    failed = _job_proto(job_pb2.JOB_STATE_FAILED)
    failed.error = "process exited 1"
    client = IrisClient(WorkloadCluster(failed))

    with pytest.raises(JobFailedError) as raised:
        Job(client, JobName.from_wire("/alice/train")).wait(timeout=1)

    assert raised.value.status.state is JobState.FAILED
    assert raised.value.status.error_message == "process exited 1"
