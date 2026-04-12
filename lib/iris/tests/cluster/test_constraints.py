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
    infer_preemptible_constraint,
    is_cpu_device_type_constraint,
    looks_like_executor,
    merge_constraints,
    extract_placement_requirements,
    preemptible_constraint,
    routing_constraints,
    soft_constraint_score,
    split_hard_soft,
)

from iris.rpc import job_pb2

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


# --- looks_like_executor heuristic ---


def _make_resources(
    cpu_millicores: int = 500,
    memory_bytes: int = 1 * 1024**3,
    device: str | None = None,
) -> job_pb2.ResourceSpecProto:
    r = job_pb2.ResourceSpecProto(cpu_millicores=cpu_millicores, memory_bytes=memory_bytes)
    if device == "cpu":
        r.device.cpu.variant = "cpu"
    elif device == "tpu":
        r.device.tpu.variant = "v5litepod-16"
    elif device == "gpu":
        r.device.gpu.variant = "h100"
        r.device.gpu.count = 8
    return r


def test_looks_like_executor_small_cpu_job():
    """0.5 CPU, 1 GiB RAM, no device → executor."""
    assert looks_like_executor(_make_resources(), replicas=1)


def test_looks_like_executor_cpu_device_explicit():
    """Explicit CpuDevice still counts as no accelerator."""
    assert looks_like_executor(_make_resources(device="cpu"), replicas=1)


def test_looks_like_executor_false_with_tpu():
    assert not looks_like_executor(_make_resources(device="tpu"), replicas=1)


def test_looks_like_executor_false_with_gpu():
    assert not looks_like_executor(_make_resources(device="gpu"), replicas=1)


def test_looks_like_executor_one_full_cpu():
    """A job requesting exactly 1 CPU core is still an executor."""
    assert looks_like_executor(_make_resources(cpu_millicores=1000), replicas=1)


def test_looks_like_executor_false_high_cpu():
    assert not looks_like_executor(_make_resources(cpu_millicores=2000), replicas=1)


def test_looks_like_executor_false_high_memory():
    assert not looks_like_executor(_make_resources(memory_bytes=5 * 1024**3), replicas=1)


def test_looks_like_executor_false_multiple_replicas():
    assert not looks_like_executor(_make_resources(), replicas=2)


def test_infer_preemptible_constraint_adds_non_preemptible():
    resources = _make_resources()
    result = infer_preemptible_constraint(resources, replicas=1, existing_constraints=[])
    assert result is not None
    assert result.key == WellKnownAttribute.PREEMPTIBLE
    assert result.value == "false"


def test_infer_preemptible_constraint_noop_when_explicit():
    resources = _make_resources()
    existing = [preemptible_constraint(True)]
    result = infer_preemptible_constraint(resources, replicas=1, existing_constraints=existing)
    assert result is None


def test_infer_preemptible_constraint_noop_for_gpu():
    resources = _make_resources(device="gpu")
    result = infer_preemptible_constraint(resources, replicas=1, existing_constraints=[])
    assert result is None


# --- Soft constraints ---


def test_preemptible_constraint_soft_default_logic():
    """preemptible=True defaults to soft, preemptible=False defaults to hard."""
    c_true = preemptible_constraint(True)
    assert c_true.mode == job_pb2.CONSTRAINT_MODE_PREFERRED
    assert c_true.value == "true"

    c_false = preemptible_constraint(False)
    assert c_false.mode == job_pb2.CONSTRAINT_MODE_REQUIRED
    assert c_false.value == "false"


def test_preemptible_constraint_soft_override():
    """Explicit soft= overrides the default logic."""
    c = preemptible_constraint(True, soft=False)
    assert c.mode == job_pb2.CONSTRAINT_MODE_REQUIRED

    c2 = preemptible_constraint(False, soft=True)
    assert c2.mode == job_pb2.CONSTRAINT_MODE_PREFERRED


def test_split_hard_soft():
    hard = eq_constraint("region", "us-central1")
    soft = Constraint(
        key="preemptible",
        op=ConstraintOp.EQ,
        value="true",
        mode=job_pb2.CONSTRAINT_MODE_PREFERRED,
    )
    hard_list, soft_list = split_hard_soft([hard, soft])
    assert len(hard_list) == 1
    assert hard_list[0].key == "region"
    assert len(soft_list) == 1
    assert soft_list[0].key == "preemptible"


def test_split_hard_soft_all_required():
    c1 = eq_constraint("region", "us-central1")
    c2 = eq_constraint("device-type", "tpu")
    hard_list, soft_list = split_hard_soft([c1, c2])
    assert len(hard_list) == 2
    assert len(soft_list) == 0


def _soft_eq(key: str, value: str) -> Constraint:
    return Constraint(key=key, op=ConstraintOp.EQ, value=value.lower(), mode=job_pb2.CONSTRAINT_MODE_PREFERRED)


def test_soft_constraint_score_counts_matches():
    attrs = {
        "preemptible": AttributeValue("true"),
        "region": AttributeValue("us-central1"),
    }
    soft1 = _soft_eq("preemptible", "true")
    soft2 = _soft_eq("region", "eu-west1")
    # Only preemptible matches
    assert soft_constraint_score(attrs, [soft1, soft2]) == 1
    # Both match
    soft2_match = _soft_eq("region", "us-central1")
    assert soft_constraint_score(attrs, [soft1, soft2_match]) == 2


def test_soft_constraint_score_zero_when_no_match():
    attrs = {"preemptible": AttributeValue("false")}
    soft = _soft_eq("preemptible", "true")
    assert soft_constraint_score(attrs, [soft]) == 0
