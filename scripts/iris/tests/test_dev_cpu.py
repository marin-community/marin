# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cluster.types import ResourceSpec
from iris.rpc import job_pb2

from scripts.iris.dev_cpu import (
    HolderJob,
    kubectl_copy_from_cmd,
    kubectl_copy_to_cmd,
    kubectl_exec_cmd,
)
from scripts.iris.dev_gpu import PRIORITY_BANDS, CoreweaveTarget, PodRef, Priority

TARGET = CoreweaveTarget(namespace="iris", kubeconfig_path="/k/cfg", kube_context="cw")
POD = PodRef(namespace="iris", pod_name="dev-cpu-pod", container="task")
KUBECTL = ["kubectl", "--kubeconfig=/k/cfg", "--context=cw", "--namespace=iris"]


class RecordingClient:
    def __init__(self):
        self.submitted: dict = {}

    def submit(self, **kwargs):
        self.submitted = kwargs
        return "job"


def test_holder_job_is_a_privileged_non_preemptible_cpu_task_with_a_lifetime():
    holder = HolderJob(
        name="ingest-debug",
        resources=ResourceSpec(cpu=16, memory="64GB", disk="200GB"),
        priority=Priority.PRODUCTION,
        max_lifetime=259200,
    )
    client = RecordingClient()
    holder.submit(client)
    submitted = client.submitted
    assert submitted["name"] == "dev-cpu-ingest-debug"
    assert submitted["container_profile"] == job_pb2.CONTAINER_PROFILE_PRIVILEGED
    assert submitted["priority_band"] == PRIORITY_BANDS[Priority.PRODUCTION]
    assert submitted["timeout"].to_seconds() == 259200
    resources = submitted["resources"].to_proto()
    assert resources.cpu_millicores == 16_000
    assert not resources.HasField("device")
    (constraint,) = [c.to_proto() for c in submitted["constraints"]]
    # A dev box that a spot eviction can take away loses its interactive session, so the rule is hard.
    assert constraint.mode == job_pb2.CONSTRAINT_MODE_REQUIRED
    assert constraint.value.string_value == "false"
    assert submitted["entrypoint"].to_proto().run_command.argv[:2] == ["python", "-c"]


def test_kubectl_commands_target_the_task_container_of_the_recorded_pod():
    assert kubectl_exec_cmd(TARGET, POD, ("bwrap", "--version")) == [
        *KUBECTL,
        "exec",
        "dev-cpu-pod",
        "-c",
        "task",
        "--",
        "bwrap",
        "--version",
    ]
    assert kubectl_copy_to_cmd(TARGET, POD, "grader.py", "/tmp/grader.py") == [
        *KUBECTL,
        "cp",
        "-c",
        "task",
        "grader.py",
        "dev-cpu-pod:/tmp/grader.py",
    ]
    assert kubectl_copy_from_cmd(TARGET, POD, "/tmp/out", "out") == [
        *KUBECTL,
        "cp",
        "-c",
        "task",
        "dev-cpu-pod:/tmp/out",
        "out",
    ]
