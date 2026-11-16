# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cluster.types import ResourceSpec
from iris.rpc import job_pb2

from scripts.iris.coreweave_dev_pod import CoreweaveTarget, PodRef, Priority, state_path
from scripts.iris.dev_cpu import DevCpuState, HolderJob, current_pod, load_state

TARGET = CoreweaveTarget(namespace="iris", kubeconfig_path="/k/cfg", kube_context="cw")
POD = PodRef(namespace="iris", pod_name="dev-cpu-pod", container="task")


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


def test_commands_resolve_the_task_pod_afresh_after_a_retry(tmp_path, monkeypatch):
    state = DevCpuState(
        session_name="ingest-debug",
        config_file="/k/iris.yaml",
        job_id="job-1",
        task_id="task-9",
        priority=Priority.PRODUCTION,
        target=TARGET,
    )
    state_file = state_path(tmp_path, "ingest-debug")
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(state.to_json())

    resolved = []
    monkeypatch.setattr(
        "scripts.iris.dev_cpu.wait_for_running_pod",
        lambda target, task_id, *, timeout: (resolved.append(task_id), POD)[1],
    )
    # Re-resolution reads the persisted task id, so a retried holder stays reachable.
    assert current_pod(load_state(state_file)) == POD
    assert resolved == ["task-9"]


def test_recorded_state_round_trips_through_json():
    state = DevCpuState(
        session_name="ingest-debug",
        config_file="/k/iris.yaml",
        job_id="job-1",
        task_id="task-9",
        priority=Priority.PRODUCTION,
        target=TARGET,
    )
    restored = DevCpuState.from_json(state.to_json())
    assert restored == state
