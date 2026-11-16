# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cluster.types import ResourceSpec
from iris.rpc import job_pb2

from scripts.iris.dev_cpu import HolderJob
from scripts.iris.dev_gpu import Priority


class RecordingClient:
    def __init__(self):
        self.submitted: dict = {}

    def submit(self, **kwargs):
        self.submitted = kwargs
        return "job"


def test_holder_job_submits_privileged_non_preemptible_cpu_task_with_lifetime():
    holder = HolderJob(
        name="ingest-debug",
        resources=ResourceSpec(cpu=16, memory="64GB", disk="200GB"),
        priority=Priority.PRODUCTION,
        max_lifetime=259200,
    )
    client = RecordingClient()
    holder.submit(client)
    submitted = client.submitted
    assert submitted["container_profile"] == job_pb2.CONTAINER_PROFILE_PRIVILEGED
    assert submitted["timeout"].to_seconds() == 259200
    resources = submitted["resources"].to_proto()
    assert not resources.HasField("device")
    (constraint,) = [c.to_proto() for c in submitted["constraints"]]
    assert constraint.mode == job_pb2.CONSTRAINT_MODE_REQUIRED
    assert constraint.value.string_value == "false"
