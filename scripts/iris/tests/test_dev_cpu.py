# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib

from click.testing import CliRunner
from iris.cluster.types import ResourceSpec
from iris.rpc import job_pb2

from scripts.iris.coreweave_dev_pod import CoreweaveTarget, PodRef, Priority, state_path
from scripts.iris.dev_cpu import DevCpuState, HolderJob, cli

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


def test_status_reports_an_ended_box_without_resolving_a_pod(tmp_path, monkeypatch):
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
    monkeypatch.setattr("scripts.iris.dev_cpu.load_state", lambda path: state)
    monkeypatch.setattr("scripts.iris.dev_cpu.controller_client", lambda config_file: contextlib.nullcontext(object()))
    monkeypatch.setattr("scripts.iris.dev_cpu.is_job_active", lambda client, job_id: False)
    resolved = []
    monkeypatch.setattr("scripts.iris.dev_cpu.wait_for_running_pod", lambda *args, **kwargs: resolved.append(1))

    result = CliRunner().invoke(cli, ["--name", "ingest-debug", "status"])

    assert result.exit_code == 0
    assert "Active: False" in result.output
    assert "Pod: none" in result.output
    assert resolved == []


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
