# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
from types import SimpleNamespace

from click.testing import CliRunner
from iris.cluster.types import ResourceSpec
from iris.resources.state import JobState
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
    assert submitted["max_retries_preemption"] == 0
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
    monkeypatch.setattr("scripts.iris.dev_cpu.STATE_DIR", tmp_path)
    state_path(tmp_path, "ingest-debug").write_text(state.to_json())
    monkeypatch.setattr("scripts.iris.dev_cpu.controller_client", lambda config_file: contextlib.nullcontext(object()))
    monkeypatch.setattr("scripts.iris.dev_cpu.is_job_active", lambda client, job_id: False)
    resolved = []
    monkeypatch.setattr("scripts.iris.dev_cpu.wait_for_running_pod", lambda *args, **kwargs: resolved.append(1))

    result = CliRunner().invoke(cli, ["--name", "ingest-debug", "status"])

    assert result.exit_code == 0
    assert "Active: False" in result.output
    assert "Pod: none" in result.output
    assert resolved == []


def test_exec_refuses_an_ended_box(tmp_path, monkeypatch):
    state = DevCpuState(
        session_name="ingest-debug",
        config_file="/k/iris.yaml",
        job_id="job-1",
        task_id="task-9",
        priority=Priority.PRODUCTION,
        target=TARGET,
    )
    monkeypatch.setattr("scripts.iris.dev_cpu.STATE_DIR", tmp_path)
    state_path(tmp_path, "ingest-debug").write_text(state.to_json())
    monkeypatch.setattr("scripts.iris.dev_cpu.controller_client", lambda config_file: contextlib.nullcontext(object()))
    monkeypatch.setattr("scripts.iris.dev_cpu.is_job_active", lambda client, job_id: False)

    result = CliRunner().invoke(cli, ["--name", "ingest-debug", "exec", "--", "bwrap", "--version"])

    assert result.exit_code != 0
    assert "has ended; release it" in result.output


def test_failed_allocation_cancels_the_holder_and_leaves_no_state(tmp_path, monkeypatch):
    cancelled = []

    class FailingJob:
        # The wire format prefixes the user namespace, as real Job objects do.
        job_id = "/ben/dev-cpu-ingest-debug"

        def state_only(self):
            return JobState.FAILED

        def status(self):
            return SimpleNamespace(error_message="scheduling rejected the task")

        def tasks(self):
            return []

    class FailingClient:
        def submit(self, **kwargs):
            return FailingJob()

        def cancel_job(self, job_name):
            cancelled.append(job_name)

    monkeypatch.setattr("scripts.iris.dev_cpu.STATE_DIR", tmp_path)
    monkeypatch.setattr("scripts.iris.dev_cpu.load_config", lambda path: None)
    monkeypatch.setattr("scripts.iris.dev_cpu.require_coreweave_platform", lambda config: TARGET)
    monkeypatch.setattr(
        "scripts.iris.dev_cpu.controller_client", lambda config_file: contextlib.nullcontext(FailingClient())
    )

    result = CliRunner().invoke(
        cli,
        [
            "--config",
            "/k/iris.yaml",
            "--name",
            "ingest-debug",
            "allocate",
            "--cpu",
            "16",
            "--memory",
            "64GB",
            "--disk",
            "200GB",
            "--priority",
            "interactive",
            "--max-lifetime",
            "259200",
            "--timeout",
            "2",
        ],
    )

    assert result.exit_code != 0
    assert len(cancelled) == 1 and "dev-cpu-ingest-debug" in str(cancelled[0])
    assert not state_path(tmp_path, "ingest-debug").exists()


def test_release_keeps_the_state_file_when_the_cancel_fails(tmp_path, monkeypatch):
    state = DevCpuState(
        session_name="ingest-debug",
        config_file="/k/iris.yaml",
        job_id="job-1",
        task_id="task-9",
        priority=Priority.PRODUCTION,
        target=TARGET,
    )
    monkeypatch.setattr("scripts.iris.dev_cpu.STATE_DIR", tmp_path)
    state_file = state_path(tmp_path, "ingest-debug")
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(state.to_json())

    class FailingCancelClient:
        def cancel_job(self, job_name):
            raise RuntimeError("tunnel dropped")

    monkeypatch.setattr(
        "scripts.iris.dev_cpu.controller_client", lambda config_file: contextlib.nullcontext(FailingCancelClient())
    )

    result = CliRunner().invoke(cli, ["--name", "ingest-debug", "release"])
    assert result.exit_code != 0
    assert state_file.exists()

    forced = CliRunner().invoke(cli, ["--name", "ingest-debug", "release", "--force"])
    assert forced.exit_code == 0
    assert not state_file.exists()


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
