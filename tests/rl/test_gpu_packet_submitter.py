# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from types import SimpleNamespace

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from fray.types import GpuConfig, JobStatus

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


class _Child:
    job_id = "/atqamar/e61-coordinator-0/e61-stage2"

    def wait(self, **_kwargs):
        return JobStatus.SUCCEEDED


class _RecordingClient:
    def __init__(self):
        self.requests = []

    def submit(self, request, *, adopt_existing):
        assert not adopt_existing
        self.requests.append(request)
        return _Child()


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
