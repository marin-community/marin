# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ConstraintDescriptor registry and constraint evaluation."""

import pytest

from iris.cluster.constraints import (
    CONSTRAINT_REGISTRY,
    AttributeValue,
    Constraint,
    ConstraintOp,
    DeviceType,
    INHERITED_CONSTRAINT_KEYS,
    PlacementRequirements,
    VALID_LOCALITY_TIERS,
    WellKnownAttribute,
    evaluate_constraint,
    is_cpu_device_type_constraint,
    locality_constraint,
    locality_topology_key,
    merge_constraints,
    extract_placement_requirements,
    routing_constraints,
)
from iris.rpc import cluster_pb2


def _eq_constraint(key: str, value: str) -> cluster_pb2.Constraint:
    c = cluster_pb2.Constraint(key=key, op=cluster_pb2.CONSTRAINT_OP_EQ)
    c.value.string_value = value
    return c


def _in_constraint(key: str, values: list[str]) -> cluster_pb2.Constraint:
    c = cluster_pb2.Constraint(key=key, op=cluster_pb2.CONSTRAINT_OP_IN)
    for v in values:
        av = c.values.add()
        av.string_value = v
    return c


# --- is_cpu_device_type_constraint ---


def test_is_cpu_device_type_constraint():
    assert is_cpu_device_type_constraint(_eq_constraint("device-type", "cpu"))
    assert is_cpu_device_type_constraint(_eq_constraint("device-type", "CPU"))
    assert not is_cpu_device_type_constraint(_eq_constraint("device-type", "gpu"))
    assert not is_cpu_device_type_constraint(_eq_constraint("device-variant", "cpu"))


# --- routing_constraints ---


def test_routing_constraints_strips_cpu_and_non_routing():
    constraints = [
        _eq_constraint("device-type", "cpu"),
        _eq_constraint("region", "us-central1"),
        _eq_constraint("tpu-name", "my-pod"),
    ]
    result = routing_constraints(constraints)
    assert len(result) == 1
    assert result[0].key == "region"


def test_routing_constraints_keeps_gpu_device_type():
    constraints = [
        _eq_constraint("device-type", "gpu"),
        _eq_constraint("device-variant", "h100"),
    ]
    result = routing_constraints(constraints)
    assert len(result) == 2


# --- evaluate_constraint for routing ---


def test_evaluate_constraint_eq():
    attr = AttributeValue("gpu")
    c = _eq_constraint("device-type", "gpu")
    assert evaluate_constraint(attr, c)
    assert not evaluate_constraint(AttributeValue("tpu"), c)


def test_evaluate_constraint_in():
    attr = AttributeValue("us-central1")
    c = _in_constraint("region", ["us-central1", "us-east1"])
    assert evaluate_constraint(attr, c)
    assert not evaluate_constraint(AttributeValue("eu-west1"), c)


# --- Normalization: proto constraints → PlacementRequirements ---


@pytest.mark.parametrize(
    "constraints, expected",
    [
        (
            [_eq_constraint("device-type", "gpu")],
            PlacementRequirements(DeviceType.GPU, None, None, None, None),
        ),
        (
            [_eq_constraint("preemptible", "true")],
            PlacementRequirements(None, None, True, None, None),
        ),
        (
            [_eq_constraint("region", "us-central1")],
            PlacementRequirements(None, None, None, frozenset({"us-central1"}), None),
        ),
        (
            [_in_constraint("zone", ["us-central1-a", "us-central1-b"])],
            PlacementRequirements(None, None, None, None, frozenset({"us-central1-a", "us-central1-b"})),
        ),
        (
            [
                _eq_constraint("device-type", "tpu"),
                _eq_constraint("device-variant", "v5litepod-16"),
                _eq_constraint("preemptible", "false"),
            ],
            PlacementRequirements(DeviceType.TPU, frozenset({"v5litepod-16"}), False, None, None),
        ),
    ],
)
def test_extract_placement_requirements_parameterized(constraints, expected):
    result = extract_placement_requirements(constraints)
    assert result == expected


# --- merge_constraints canonical override ---


def test_merge_canonical_key_child_overrides_parent():
    parent = [Constraint(key="device-type", op=ConstraintOp.EQ, value="gpu")]
    child = [Constraint(key="device-type", op=ConstraintOp.EQ, value="tpu")]
    merged = merge_constraints(parent, child)
    assert len(merged) == 1
    assert merged[0].value == "tpu"


def test_merge_non_canonical_key_appends():
    parent = [Constraint(key="tpu-name", op=ConstraintOp.EQ, value="pod-a")]
    child = [Constraint(key="tpu-name", op=ConstraintOp.EQ, value="pod-b")]
    merged = merge_constraints(parent, child)
    assert len(merged) == 2


# --- INHERITED_CONSTRAINT_KEYS ---


def test_inherited_keys_strips_device_and_preemptible():
    """Parent H100 constraints must not leak into child job inheritance."""
    constraints = [
        _eq_constraint(WellKnownAttribute.DEVICE_TYPE, "gpu"),
        _eq_constraint(WellKnownAttribute.DEVICE_VARIANT, "h100"),
        _eq_constraint(WellKnownAttribute.PREEMPTIBLE, "true"),
        _eq_constraint(WellKnownAttribute.REGION, "us-central1"),
        _eq_constraint(WellKnownAttribute.ZONE, "us-central1-a"),
    ]

    inherited = [c for c in constraints if c.key in INHERITED_CONSTRAINT_KEYS]

    assert len(inherited) == 2
    keys = {c.key for c in inherited}
    assert keys == {WellKnownAttribute.REGION, WellKnownAttribute.ZONE}


# ---------------------------------------------------------------------------
# Locality constraint helpers
# ---------------------------------------------------------------------------


def test_locality_constraint_same_slice():
    c = locality_constraint("same-slice")
    assert c.key == WellKnownAttribute.LOCALITY
    assert c.op == ConstraintOp.EQ
    assert c.value == "same-slice"


def test_locality_constraint_all_tiers():
    for tier in VALID_LOCALITY_TIERS:
        c = locality_constraint(tier)
        assert c.value == tier


def test_locality_constraint_invalid_tier():
    with pytest.raises(ValueError, match="Invalid locality tier"):
        locality_constraint("invalid")


def test_locality_topology_key_mapping():
    assert locality_topology_key("same-slice") == "ib.coreweave.cloud/spine-switch"
    assert locality_topology_key("same-rack") == "topology.kubernetes.io/rack"
    assert locality_topology_key("same-superpod") == "backend.coreweave.cloud/superpod"


def test_locality_topology_key_invalid():
    with pytest.raises(ValueError, match="Invalid locality tier"):
        locality_topology_key("invalid")


def test_locality_registered_in_constraint_registry():
    desc = CONSTRAINT_REGISTRY.get("locality")
    assert desc is not None
    assert desc.canonical is True
    assert desc.routing is False


def test_locality_not_routing():
    """Locality is scheduler-only, not used for autoscaler routing."""
    proto_constraint = _eq_constraint(WellKnownAttribute.LOCALITY, "same-slice")
    filtered = routing_constraints([proto_constraint])
    assert filtered == []
