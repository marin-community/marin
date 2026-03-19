# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ConstraintDescriptor registry and constraint evaluation."""

import pytest

from iris.cluster.constraints import (
    AttributeValue,
    Constraint,
    ConstraintOp,
    DeviceType,
    INHERITED_CONSTRAINT_KEYS,
    PlacementRequirements,
    WellKnownAttribute,
    evaluate_constraint,
    is_cpu_device_type_constraint,
    merge_constraints,
    extract_placement_requirements,
    routing_constraints,
)

from .conftest import eq_constraint, in_constraint

# --- is_cpu_device_type_constraint ---


def test_is_cpu_device_type_constraint():
    assert is_cpu_device_type_constraint(eq_constraint("device-type", "cpu"))
    assert is_cpu_device_type_constraint(eq_constraint("device-type", "CPU"))
    assert not is_cpu_device_type_constraint(eq_constraint("device-type", "gpu"))
    assert not is_cpu_device_type_constraint(eq_constraint("device-variant", "cpu"))


# --- routing_constraints ---


def test_routing_constraints_strips_cpu_and_non_routing():
    constraints = [
        eq_constraint("device-type", "cpu"),
        eq_constraint("region", "us-central1"),
        eq_constraint("tpu-name", "my-pod"),
    ]
    result = routing_constraints(constraints)
    assert len(result) == 1
    assert result[0].key == "region"


def test_routing_constraints_keeps_gpu_device_type():
    constraints = [
        eq_constraint("device-type", "gpu"),
        eq_constraint("device-variant", "h100"),
    ]
    result = routing_constraints(constraints)
    assert len(result) == 2


# --- evaluate_constraint for routing ---


def test_evaluate_constraint_eq():
    attr = AttributeValue("gpu")
    c = eq_constraint("device-type", "gpu")
    assert evaluate_constraint(attr, c)
    assert not evaluate_constraint(AttributeValue("tpu"), c)


def test_evaluate_constraint_in():
    attr = AttributeValue("us-central1")
    c = in_constraint("region", ["us-central1", "us-east1"])
    assert evaluate_constraint(attr, c)
    assert not evaluate_constraint(AttributeValue("eu-west1"), c)


# --- Normalization: proto constraints → PlacementRequirements ---


@pytest.mark.parametrize(
    "constraints, expected",
    [
        (
            [eq_constraint("device-type", "gpu")],
            PlacementRequirements(DeviceType.GPU, None, None, None, None),
        ),
        (
            [eq_constraint("preemptible", "true")],
            PlacementRequirements(None, None, True, None, None),
        ),
        (
            [eq_constraint("region", "us-central1")],
            PlacementRequirements(None, None, None, frozenset({"us-central1"}), None),
        ),
        (
            [in_constraint("zone", ["us-central1-a", "us-central1-b"])],
            PlacementRequirements(None, None, None, None, frozenset({"us-central1-a", "us-central1-b"})),
        ),
        (
            [
                eq_constraint("device-type", "tpu"),
                eq_constraint("device-variant", "v5litepod-16"),
                eq_constraint("preemptible", "false"),
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
        eq_constraint(WellKnownAttribute.DEVICE_TYPE, "gpu"),
        eq_constraint(WellKnownAttribute.DEVICE_VARIANT, "h100"),
        eq_constraint(WellKnownAttribute.PREEMPTIBLE, "true"),
        eq_constraint(WellKnownAttribute.REGION, "us-central1"),
        eq_constraint(WellKnownAttribute.ZONE, "us-central1-a"),
    ]

    inherited = [c for c in constraints if c.key in INHERITED_CONSTRAINT_KEYS]

    assert len(inherited) == 2
    keys = {c.key for c in inherited}
    assert keys == {WellKnownAttribute.REGION, WellKnownAttribute.ZONE}
