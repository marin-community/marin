# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from fray.iris_backend import FrayIrisClient, convert_constraints, convert_environment
from fray.types import GpuConfig, JobStatus
from rigging.timing import Duration

from experiments.post_training.gpu_packet_submitter import (
    GpuJobPacket,
    _read_launch_action,
    _submit_coordinator,
    coordinate_packet,
    submit_packet,
)


def _packet() -> GpuJobPacket:
    return GpuJobPacket.from_mapping(
        {
            "schema_version": 1,
            "job_name": "async-rl-v2-e61-stage2-a1",
            "command": ["bash", "-c", "python -m experiments.post_training.async_rl --run e61"],
            "gpu_variant": "H100",
            "gpus_per_task": 8,
            "replicas": 5,
            "cpu": 80,
            "ram": "950g",
            "disk": "400g",
            "receipt_uri": "s3://marin-us-east-02a/marin/users/ahmad/e61/admission.json",
            "timeout_seconds": 21600,
            "environment": {
                "env_vars": {"MARIN_PREFIX": "s3://marin-us-east-02a/marin"},
                "extras": ["gpu"],
                "setup_scripts": None,
                "sync_packages": [],
            },
        }
    )


_TEST_CREDENTIALS = {
    "DAYTONA_API_KEY": "daytona-test-secret",
    "HF_TOKEN": "hf-test-secret",
    "WANDB_API_KEY": "wandb-test-secret",
}


def _packet_with_credentials() -> GpuJobPacket:
    raw = json.loads(_packet().canonical_json())
    raw["credential_placeholders"] = list(_TEST_CREDENTIALS)
    raw["environment"]["env_vars"].update({name: f"${{ENV:{name}}}" for name in _TEST_CREDENTIALS})
    return GpuJobPacket.from_mapping(raw)


def test_minimal_experimentctl_packet_uses_one_h100_defaults():
    packet = GpuJobPacket.from_mapping(
        {"command": ["python3", "dummy_train.py"], "fixture": True},
        job_name="dummy-gpu-job",
        receipt_uri="/tmp/dummy-gpu-job-receipt.json",
    )

    request = packet.child_request(target_cluster=None)
    assert isinstance(request.resources.device, GpuConfig)
    assert request.resources.device.variant == "H100"
    assert request.resources.device.count == 1
    assert request.replicas == 1
    assert request.name == "dummy-gpu-job"


def test_unspecified_preemptible_matches_native_absent_constraint():
    request = _packet().child_request(target_cluster=None)

    assert convert_constraints(request.resources) == []


def test_packet_execution_timeout_reaches_iris_task_deadline():
    fake_iris = MagicMock()
    fake_iris.submit.return_value = MagicMock(job_id="/atqamar/coordinator/child")

    FrayIrisClient.from_iris_client(fake_iris).submit(_packet().child_request(target_cluster=None))

    assert fake_iris.submit.call_args.kwargs["timeout"] == Duration.from_seconds(21600)


def test_generic_gpu_child_pins_empty_jax_platform_device_default():
    request = _packet().child_request(target_cluster=None)

    environment = convert_environment(request.environment, request.resources.device)
    assert environment.to_proto().env_vars["JAX_PLATFORMS"] == ""


class _Child:
    job_id = "/atqamar/e61-coordinator-0/e61-stage2"

    def wait(self, *, timeout, **_kwargs):
        assert timeout is None
        return JobStatus.SUCCEEDED


class _RecordingClient:
    def __init__(self):
        self.requests = []

    def submit(self, request, *, adopt_existing):
        assert not adopt_existing
        self.requests.append(request)
        return _Child()


class _QueuedPastExecutionTimeoutChild:
    job_id = "/atqamar/e61-coordinator-0/e61-stage2-queued"

    def __init__(self, queued_seconds):
        self.queued_seconds = queued_seconds
        self.waits = 0

    def wait(self, *, timeout, **_kwargs):
        self.waits += 1
        if timeout is not None and timeout < self.queued_seconds:
            return JobStatus.RUNNING
        if self.waits == 1:
            return JobStatus.RUNNING
        return JobStatus.SUCCEEDED


class _QueuedChildClient:
    def __init__(self, child):
        self.child = child
        self.requests = []

    def submit(self, request, *, adopt_existing):
        assert not adopt_existing
        self.requests.append(request)
        return self.child


def test_coordinator_submits_unpinned_gpu_child_and_records_target_local_queue(capsys):
    packet = _packet()
    client = _RecordingClient()
    written = {}

    def queue_reader(job_id):
        assert job_id == _Child.job_id
        return {
            "job_state": "PENDING",
            "task_count": 5,
            "tasks": [
                {
                    "task_id": f"{job_id}/{index}",
                    "state": "PENDING",
                    "pending_reason": "SchedulingGated",
                    "status_message": "waiting for H100 quota",
                    "execution_cluster_id": "local",
                }
                for index in range(5)
            ],
        }

    receipt = coordinate_packet(
        packet,
        target_cluster="cw-rno2a",
        fallback_cluster="cw-us-east-02a",
        client=client,
        queue_reader=queue_reader,
        receipt_writer=lambda uri, value: written.update(uri=uri, value=value),
        coordinator_job_id="/atqamar/e61-coordinator",
    )

    request = client.requests[0]
    assert isinstance(request.resources.device, GpuConfig)
    assert request.resources.device.variant == "H100"
    assert request.resources.device.count == 8
    assert request.replicas == 5
    assert request.timeout_seconds == 21600
    assert request.resources.target_cluster is None
    assert receipt["route"] == "coordinator"
    assert receipt["target_cluster"] == "cw-rno2a"
    assert receipt["fallback_cluster"] == "cw-us-east-02a"
    assert receipt["capacity_source"] == "cluster_queue"
    assert receipt["capacity_observation_source"] == "target_cluster_local_queue"
    assert receipt["coordinator_job_id"] == "/atqamar/e61-coordinator"
    assert receipt["terminal_result"] == "succeeded"
    assert written == {"uri": packet.receipt_uri, "value": receipt}
    assert (
        "GPU_PACKET_COORDINATOR_ADMISSION_PASS route=coordinator target=cw-rno2a " "fallback=cw-us-east-02a"
    ) in capsys.readouterr().out


def test_coordinator_waits_past_execution_timeout_for_terminal_receipt(monkeypatch):
    packet = replace(_packet(), timeout_seconds=1)
    child = _QueuedPastExecutionTimeoutChild(queued_seconds=2)
    client = _QueuedChildClient(child)
    written = []
    backoffs = []
    monkeypatch.setattr("experiments.post_training.gpu_packet_submitter.time.sleep", backoffs.append)

    def queue_reader(job_id):
        return {
            "job_state": "RUNNING",
            "task_count": 5,
            "tasks": [
                {
                    "task_id": f"{job_id}/{index}",
                    "state": "PENDING",
                    "pending_reason": "SchedulingGated",
                    "status_message": "waiting for H100 quota",
                    "execution_cluster_id": "local",
                }
                for index in range(5)
            ],
        }

    def write_receipt(uri, value):
        assert child.waits == 2
        written.append((uri, value))

    receipt = coordinate_packet(
        packet,
        target_cluster="cw-us-east-02a",
        fallback_cluster="cw-rno2a",
        client=client,
        queue_reader=queue_reader,
        receipt_writer=write_receipt,
        coordinator_job_id="/atqamar/e61-coordinator",
    )

    assert client.requests[0].timeout_seconds == 1
    assert receipt["selected_cluster"] == "cw-us-east-02a"
    assert receipt["fallback_used"] is False
    assert receipt["terminal_result"] == "succeeded"
    assert backoffs == [1.0]
    assert written == [(packet.receipt_uri, receipt)]


def test_coordinator_records_terminal_failure_before_propagating():
    class FailedChild:
        job_id = "/atqamar/e61-coordinator-0/e61-stage2-failed"

        def wait(self, **_kwargs):
            return JobStatus.FAILED

    packet = _packet()
    written = []

    with pytest.raises(RuntimeError, match="finished with failed"):
        coordinate_packet(
            packet,
            target_cluster="cw-rno2a",
            fallback_cluster="cw-us-east-02a",
            client=_QueuedChildClient(FailedChild()),
            queue_reader=lambda _job_id: {"job_state": "FAILED", "task_count": 5, "tasks": []},
            receipt_writer=lambda uri, value: written.append((uri, value)),
            coordinator_job_id="/atqamar/e61-coordinator",
        )

    assert len(written) == 1
    assert written[0][0] == packet.receipt_uri
    assert written[0][1]["terminal_result"] == "failed"


def test_coordinator_resolves_named_credentials_without_artifact_disclosure(monkeypatch, capsys):
    packet = _packet_with_credentials()
    for name, value in _TEST_CREDENTIALS.items():
        monkeypatch.setenv(name, value)
    client = _RecordingClient()
    written = []

    receipt = coordinate_packet(
        packet,
        target_cluster="cw-rno2a",
        fallback_cluster="cw-us-east-02a",
        client=client,
        queue_reader=lambda _job_id: {"job_state": "RUNNING", "task_count": 5, "tasks": []},
        receipt_writer=lambda uri, value: written.append((uri, value)),
        coordinator_job_id="/atqamar/e61-coordinator",
    )

    request_env = client.requests[0].environment.env_vars
    assert {name: request_env[name] for name in _TEST_CREDENTIALS} == _TEST_CREDENTIALS
    artifact_text = packet.canonical_json() + json.dumps(receipt) + capsys.readouterr().out
    assert all(value not in artifact_text for value in _TEST_CREDENTIALS.values())
    assert written == [(packet.receipt_uri, receipt)]


def test_coordinator_envelope_forwards_only_declared_credentials(monkeypatch, capsys):
    packet = _packet_with_credentials()
    for name, value in _TEST_CREDENTIALS.items():
        monkeypatch.setenv(name, value)
    captured = {}

    class RecordingIrisClient:
        def submit(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(job_id="/atqamar/e61-coordinator")

    packet_bytes = packet.canonical_json().encode()
    receipt = submit_packet(
        packet,
        target_cluster="cw-rno2a",
        fallback_cluster="cw-us-east-02a",
        route="coordinator",
        coordinator_job_name="async-rl-v2-e61-coordinator",
        client=RecordingIrisClient(),
        source_packet_sha256=packet.sha256,
        source_packet_bytes=packet_bytes,
    )

    parent_env = captured["environment"].env_vars
    assert {name: parent_env[name] for name in _TEST_CREDENTIALS} == _TEST_CREDENTIALS
    assert set(parent_env) - set(_TEST_CREDENTIALS) == {
        "GPU_PACKET_FALLBACK_CLUSTER",
        "GPU_PACKET_JOB_NAME",
        "GPU_PACKET_RECEIPT_URI",
        "GPU_PACKET_ROUTE",
        "GPU_PACKET_SELECTED_CLUSTER",
        "GPU_PACKET_SOURCE_SHA256",
        "GPU_PACKET_TARGET_CLUSTER",
    }
    assert captured["entrypoint"].workdir_files["gpu-packet.json"] == packet_bytes
    artifact_bytes = packet_bytes + json.dumps(receipt).encode() + capsys.readouterr().out.encode()
    assert all(value.encode() not in artifact_bytes for value in _TEST_CREDENTIALS.values())


def test_coordinator_rejects_missing_named_credentials(monkeypatch):
    packet = _packet_with_credentials()
    for name in packet.credential_placeholders:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(
        ValueError,
        match="GPU packet requires ambient environment variables: DAYTONA_API_KEY, HF_TOKEN, WANDB_API_KEY",
    ):
        packet.child_request(target_cluster=None)


def test_packet_rejects_undeclared_environment_reference():
    raw = json.loads(_packet().canonical_json())
    raw["environment"]["env_vars"]["WANDB_API_KEY"] = "${ENV:WANDB_API_KEY}"

    with pytest.raises(ValueError, match="credential_placeholders must exactly name"):
        GpuJobPacket.from_mapping(raw)


def test_packet_rejects_malformed_environment_reference():
    raw = json.loads(_packet().canonical_json())
    raw["credential_placeholders"] = ["WANDB_API_KEY"]
    raw["environment"]["env_vars"]["WANDB_API_KEY"] = "prefix-${ENV:WANDB_API_KEY}"

    with pytest.raises(ValueError, match=r"must use exact \$\{ENV:VAR\} syntax"):
        GpuJobPacket.from_mapping(raw)


def test_direct_route_pins_gpu_child_and_records_federated_capacity_source():
    packet = _packet()
    client = _RecordingClient()

    receipt = submit_packet(
        packet,
        target_cluster="cw-us-east-02a",
        fallback_cluster="cw-rno2a",
        route="direct",
        coordinator_job_name="unused-for-direct-route",
        client=object(),
        direct_client=client,
        route_receipt={
            "path": "/probe/east.json",
            "sha256": "a" * 64,
            "line": "GPU_PACKET_DIRECT_ONE_H100_ADMISSION_PASS",
        },
        receipt_writer=lambda uri, value: None,
    )

    assert client.requests[0].resources.target_cluster == "cw-us-east-02a"
    assert receipt["route"] == "direct"
    assert receipt["target_cluster"] == "cw-us-east-02a"
    assert receipt["fallback_cluster"] == "cw-rno2a"
    assert receipt["capacity_source"] == "cluster_queue"
    assert receipt["capacity_observation_source"] == "direct_route_admission_receipt"
    assert receipt["terminal_result"] == "succeeded"


def test_coordinator_route_selects_target_and_embeds_packet():
    packet = _packet()
    captured = {}

    def submit_coordinator(client, actual_packet, target, fallback, selected, name, sha256, packet_bytes):
        captured.update(
            client=client,
            packet=actual_packet,
            target=target,
            fallback=fallback,
            selected=selected,
            name=name,
            sha256=sha256,
            packet_bytes=packet_bytes,
        )
        return SimpleNamespace(job_id="/atqamar/e61-coordinator")

    receipt = submit_packet(
        packet,
        target_cluster="cw-rno2a",
        fallback_cluster="cw-us-east-02a",
        route="coordinator",
        coordinator_job_name="async-rl-v2-e61-coordinator",
        client="hub-client",
        coordinator_submitter=submit_coordinator,
    )

    assert captured == {
        "client": "hub-client",
        "packet": packet,
        "target": "cw-rno2a",
        "fallback": "cw-us-east-02a",
        "selected": "cw-rno2a",
        "name": "async-rl-v2-e61-coordinator",
        "sha256": packet.sha256,
        "packet_bytes": packet.canonical_json().encode(),
    }
    assert receipt["submitted_job_id"] == "/atqamar/e61-coordinator"
    assert receipt["route"] == "coordinator"
    assert receipt["capacity_source"] == "cluster_queue"


def test_coordinator_envelope_pins_cpu_parent_and_preserves_packet():
    packet = _packet()
    captured = {}

    class RecordingIrisClient:
        def submit(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(job_id="/atqamar/e61-coordinator")

    _submit_coordinator(
        RecordingIrisClient(),
        packet,
        "cw-rno2a",
        "cw-us-east-02a",
        "cw-rno2a",
        "async-rl-v2-e61-coordinator",
        "a" * 64,
        b'{"command":["echo","pass"],"fixture":true}',
    )

    assert captured["name"] == "async-rl-v2-e61-coordinator"
    assert captured["constraints"][0].key == "cluster"
    assert captured["constraints"][0].values[0].value == "cw-rno2a"
    assert captured["entrypoint"].workdir_files["gpu-packet.json"] == b'{"command":["echo","pass"],"fixture":true}'
    assert captured["timeout"] is None
    assert captured["environment"].env_vars == {
        "GPU_PACKET_ROUTE": "coordinator",
        "GPU_PACKET_TARGET_CLUSTER": "cw-rno2a",
        "GPU_PACKET_FALLBACK_CLUSTER": "cw-us-east-02a",
        "GPU_PACKET_SELECTED_CLUSTER": "cw-rno2a",
        "GPU_PACKET_SOURCE_SHA256": "a" * 64,
        "GPU_PACKET_JOB_NAME": "async-rl-v2-e61-stage2-a1",
        "GPU_PACKET_RECEIPT_URI": "s3://marin-us-east-02a/marin/users/ahmad/e61/admission.json",
    }


def test_experimentctl_launch_action_verifies_exact_packet_binding(tmp_path):
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(_packet().canonical_json())
    packet_sha256 = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    action = {
        "action": "launch_gpu",
        "handoff": "generic_gpu_submitter",
        "job_name": "/atqamar/async-rl-v2-e61-stage2-a1",
        "launch": {
            "submitter_interface": "generic-gpu-coordinator/v1",
            "packet": {"path": str(packet_path), "sha256": packet_sha256},
            "target_cluster": "cw-rno2a",
            "fallback_cluster": "cw-us-east-02a",
            "capacity_source": "cluster_queue",
            "route": "coordinator",
            "route_receipt": None,
            "receipt_required": True,
        },
    }
    action_path = tmp_path / "action.json"
    action_path.write_text(json.dumps(action))

    assert _read_launch_action(action_path) == (
        packet_path,
        packet_sha256,
        "async-rl-v2-e61-stage2-a1",
        "cw-rno2a",
        "cw-us-east-02a",
        "coordinator",
        None,
    )

    action["launch"]["capacity_source"] = "controller_peer_availability"
    action_path.write_text(json.dumps(action))
    with pytest.raises(ValueError, match="capacity_source must be cluster_queue"):
        _read_launch_action(action_path)


def test_retryable_target_submission_uses_fallback_and_records_selection():
    packet = _packet()
    attempts = []

    def submit_coordinator(_client, _packet, _target, _fallback, selected, _name, _sha256, _packet_bytes):
        attempts.append(selected)
        if selected == "cw-rno2a":
            raise ConnectError(Code.FAILED_PRECONDITION, "target cluster unavailable")
        return SimpleNamespace(job_id="/atqamar/e61-fallback-coordinator")

    receipt = submit_packet(
        packet,
        target_cluster="cw-rno2a",
        fallback_cluster="cw-us-east-02a",
        route="coordinator",
        coordinator_job_name="async-rl-v2-e61-coordinator",
        client="hub-client",
        coordinator_submitter=submit_coordinator,
    )

    assert attempts == ["cw-rno2a", "cw-us-east-02a"]
    assert receipt["selected_cluster"] == "cw-us-east-02a"
    assert receipt["fallback_used"] is True
