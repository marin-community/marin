# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the iris.worker / iris.task stats schemas."""

from datetime import datetime

from iris.cluster.worker.stats import (
    TASK_STATS_NAMESPACE,
    WORKER_STATS_NAMESPACE,
    IrisTaskStat,
    IrisWorkerStat,
    WorkerStatus,
    build_task_stat,
    build_worker_stat,
)
from iris.rpc import job_pb2


def test_namespace_constants():
    assert WORKER_STATS_NAMESPACE == "iris.worker"
    assert TASK_STATS_NAMESPACE == "iris.task"


def test_build_worker_stat_shape():
    snapshot = job_pb2.WorkerResourceSnapshot(
        host_cpu_percent=42,
        memory_used_bytes=1_000_000,
        memory_total_bytes=8_000_000,
        disk_used_bytes=500_000,
        disk_total_bytes=10_000_000,
        running_task_count=2,
        total_process_count=37,
        net_recv_bps=1234,
        net_sent_bps=5678,
    )
    metadata = job_pb2.WorkerMetadata(
        cpu_count=8,
        memory_bytes=16_000_000_000,
        tpu_name="tpu-x",
        gce_instance_name="vm-1",
        gce_zone="us-central1-a",
    )
    metadata.attributes["device-type"].string_value = "tpu"
    metadata.attributes["device-variant"].string_value = "v6e-8"

    ts = datetime(2026, 5, 1, 12, 0, 0)
    stat = build_worker_stat(
        worker_id="w-1",
        ts=ts,
        status=WorkerStatus.RUNNING,
        address="10.0.0.1:8080",
        snapshot=snapshot,
        metadata=metadata,
    )

    assert isinstance(stat, IrisWorkerStat)
    assert stat.worker_id == "w-1"
    assert stat.ts == ts
    assert stat.status == WorkerStatus.RUNNING
    assert stat.address == "10.0.0.1:8080"
    assert stat.cpu_pct == 42.0
    assert isinstance(stat.cpu_pct, float)
    assert stat.mem_bytes == 1_000_000
    assert stat.mem_total_bytes == 8_000_000
    assert stat.disk_used_bytes == 500_000
    assert stat.disk_total_bytes == 10_000_000
    assert stat.running_task_count == 2
    assert stat.total_process_count == 37
    assert stat.net_recv_bps == 1234
    assert stat.net_sent_bps == 5678
    assert stat.device_type == "tpu"
    assert stat.device_variant == "v6e-8"
    assert stat.cpu_count == 8
    assert stat.memory_bytes == 16_000_000_000
    assert stat.tpu_name == "tpu-x"
    assert stat.gce_instance_name == "vm-1"
    assert stat.zone == "us-central1-a"
    # Healthy field intentionally absent — that is a controller decision.
    assert not hasattr(stat, "healthy")


def test_build_worker_stat_zone_falls_back_to_attribute():
    """When gce_zone is empty, fall back to the ``zone`` worker attribute."""
    snapshot = job_pb2.WorkerResourceSnapshot()
    metadata = job_pb2.WorkerMetadata()
    metadata.attributes["zone"].string_value = "fallback-zone"

    stat = build_worker_stat(
        worker_id="w-1",
        ts=datetime(2026, 5, 1),
        status=WorkerStatus.IDLE,
        address="addr",
        snapshot=snapshot,
        metadata=metadata,
    )
    assert stat.zone == "fallback-zone"


def test_build_task_stat_shape():
    usage = job_pb2.ResourceUsage(
        memory_mb=512,
        memory_peak_mb=600,
        disk_mb=128,
        cpu_millicores=1500,
        process_count=3,
    )
    ts = datetime(2026, 5, 1, 12, 0, 0)
    stat = build_task_stat(
        task_id="/u/job/0",
        attempt_id=2,
        worker_id="w-1",
        ts=ts,
        usage=usage,
    )
    assert isinstance(stat, IrisTaskStat)
    assert stat.task_id == "/u/job/0"
    assert stat.attempt_id == 2
    assert stat.worker_id == "w-1"
    assert stat.ts == ts
    assert stat.cpu_millicores == 1500
    assert stat.memory_mb == 512
    assert stat.memory_peak_mb == 600
    assert stat.disk_mb == 128
    # Accelerator fields default to None.
    assert stat.accelerator_util_pct is None
    assert stat.accelerator_mem_bytes is None


def test_build_task_stat_with_accelerator():
    usage = job_pb2.ResourceUsage(memory_mb=1, memory_peak_mb=2, disk_mb=3, cpu_millicores=4, process_count=1)
    stat = build_task_stat(
        task_id="t",
        attempt_id=0,
        worker_id="w",
        ts=datetime(2026, 5, 1),
        usage=usage,
        accelerator_util_pct=88.5,
        accelerator_mem_bytes=2_000_000,
    )
    assert stat.accelerator_util_pct == 88.5
    assert stat.accelerator_mem_bytes == 2_000_000
