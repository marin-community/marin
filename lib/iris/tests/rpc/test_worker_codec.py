# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral coverage for the legacy WorkerService wire boundary."""

from iris.rpc import job_pb2, worker_pb2
from iris.rpc.worker_codec import (
    attempt_launch_from_proto,
    attempt_launch_to_proto,
    process_info_from_proto,
    process_info_to_proto,
    reconcile_request_from_proto,
    worker_metadata_from_proto,
    worker_metadata_to_proto,
)


def test_worker_launch_wire_round_trip_preserves_execution_contract() -> None:
    wire = job_pb2.RunTaskRequest(
        task_id="/alice/train/7",
        num_tasks=8,
        attempt_id=3,
        attempt_uid="abc123",
        entrypoint=job_pb2.RuntimeEntrypoint(
            setup_commands=["uv sync"],
            run_command=job_pb2.CommandEntrypoint(argv=["python", "train.py"]),
            workdir_files={"config.json": b"{}"},
            workdir_file_refs={"weights": "sha256:abc"},
        ),
        environment=job_pb2.EnvironmentConfig(env_vars={"A": "b"}, setup_scripts=["uv sync"]),
        bundle_id="bundle",
        resources=job_pb2.ResourceSpecProto(cpu_millicores=2500, memory_bytes=8_000, disk_bytes=9_000),
        ports=["http"],
        task_image="image@sha256:digest",
        priority=job_pb2.PRIORITY_BAND_BATCH,
        container_profile=job_pb2.CONTAINER_PROFILE_RESTRICTED,
    )
    wire.timeout.milliseconds = 12_000
    wire.coscheduling.group_by = "tpu-name"
    wire.constraints.add(
        key="region",
        op=job_pb2.CONSTRAINT_OP_EQ,
        value=job_pb2.AttributeValue(string_value="us-central1"),
        mode=job_pb2.CONSTRAINT_MODE_REQUIRED,
    )

    assert attempt_launch_to_proto(attempt_launch_from_proto(wire)) == wire


def test_worker_metadata_wire_round_trip_preserves_typed_attributes_and_provenance() -> None:
    wire = job_pb2.WorkerMetadata(
        hostname="worker-1",
        ip_address="10.0.0.1",
        cpu_count=8,
        memory_bytes=64_000,
        disk_bytes=128_000,
        device=job_pb2.DeviceConfig(gpu=job_pb2.GpuDevice(variant="H100", count=8)),
        attributes={
            "pool": job_pb2.AttributeValue(string_value="train"),
            "ordinal": job_pb2.AttributeValue(int_value=3),
        },
        provenance=job_pb2.Provenance(
            tree_hash="tree",
            base_commit="commit",
            dirty=True,
            branch="feature",
            built_by="alice",
        ),
    )

    native = worker_metadata_from_proto(wire)

    assert worker_metadata_from_proto(worker_metadata_to_proto(native)) == native


def test_worker_reconcile_run_without_repeated_spec_remains_run_intent() -> None:
    wire = worker_pb2.Worker.ReconcileRequest(
        worker_id="worker-1",
        desired=[worker_pb2.Worker.DesiredAttempt(attempt_uid="abc", run=worker_pb2.Worker.AttemptSpec())],
    )

    request = reconcile_request_from_proto(wire)

    assert request.desired[0].is_run
    assert request.desired[0].launch is None


def test_process_info_wire_round_trip_preserves_reported_runtime_identity() -> None:
    wire = job_pb2.ProcessInfo(
        hostname="worker-1",
        pid=42,
        python_version="3.12.1",
        uptime_ms=123,
        memory_rss_bytes=456,
        memory_vms_bytes=789,
        thread_count=4,
        open_fd_count=5,
        memory_total_bytes=999,
        cpu_count=8,
        cpu_millicores=250,
        provenance=job_pb2.Provenance(
            tree_hash="tree",
            base_commit="commit",
            dirty=False,
            branch="",
            built_by="",
        ),
    )

    assert process_info_to_proto(process_info_from_proto(wire)) == wire
