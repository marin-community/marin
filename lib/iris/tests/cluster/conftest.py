# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for cluster-level scheduling tests.

Provides constraint builders and resource spec fixtures used across
test_scheduler.py, test_scaling_group.py, and test_constraints.py.
"""

import pytest

from iris.cluster.constraints import WellKnownAttribute
from iris.rpc import cluster_pb2

# ---------------------------------------------------------------------------
# Constraint builders
# ---------------------------------------------------------------------------


def eq_constraint(key: str, value: str) -> cluster_pb2.Constraint:
    """Build an EQ constraint proto for the given key and string value."""
    c = cluster_pb2.Constraint(key=key, op=cluster_pb2.CONSTRAINT_OP_EQ)
    c.value.string_value = value
    return c


def in_constraint(key: str, values: list[str]) -> cluster_pb2.Constraint:
    """Build an IN constraint proto for the given key and string values."""
    c = cluster_pb2.Constraint(key=key, op=cluster_pb2.CONSTRAINT_OP_IN)
    for v in values:
        av = c.values.add()
        av.string_value = v
    return c


# ---------------------------------------------------------------------------
# Resource spec fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cpu_resource_spec() -> cluster_pb2.ResourceSpecProto:
    """Standard CPU resource spec for scheduling tests."""
    return cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=4 * 1024**3)


@pytest.fixture
def gpu_resource_spec() -> cluster_pb2.ResourceSpecProto:
    """GPU resource spec with device type constraint."""
    spec = cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=4 * 1024**3)
    spec.device.gpu.CopyFrom(cluster_pb2.GpuDevice(variant="h100", count=1))
    return spec


# ---------------------------------------------------------------------------
# Worker attribute helpers
# ---------------------------------------------------------------------------


def make_worker_attrs(
    region: str = "us-central1",
    device_type: str = "cpu",
    device_variant: str = "",
    preemptible: str | None = None,
    zone: str | None = None,
    tpu_name: str | None = None,
    tpu_worker_id: int | None = None,
    **extras: str,
) -> dict[str, cluster_pb2.AttributeValue]:
    """Build a worker attributes dict for scheduling tests.

    Returns a dict suitable for setting on WorkerMetadata.attributes.
    """
    attrs: dict[str, cluster_pb2.AttributeValue] = {
        WellKnownAttribute.DEVICE_TYPE: cluster_pb2.AttributeValue(string_value=device_type),
    }
    if region:
        attrs[WellKnownAttribute.REGION] = cluster_pb2.AttributeValue(string_value=region)
    if device_variant:
        attrs[WellKnownAttribute.DEVICE_VARIANT] = cluster_pb2.AttributeValue(string_value=device_variant)
    if preemptible is not None:
        attrs[WellKnownAttribute.PREEMPTIBLE] = cluster_pb2.AttributeValue(string_value=preemptible)
    if zone is not None:
        attrs[WellKnownAttribute.ZONE] = cluster_pb2.AttributeValue(string_value=zone)
    if tpu_name is not None:
        attrs[WellKnownAttribute.TPU_NAME] = cluster_pb2.AttributeValue(string_value=tpu_name)
    if tpu_worker_id is not None:
        attrs[WellKnownAttribute.TPU_WORKER_ID] = cluster_pb2.AttributeValue(int_value=tpu_worker_id)
    for key, val in extras.items():
        attrs[key] = cluster_pb2.AttributeValue(string_value=val)
    return attrs
