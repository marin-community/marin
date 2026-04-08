# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.cluster.types — Entrypoint, EnvironmentSpec, and constraint helpers."""

import pytest

from iris.cluster.constraints import (
    Constraint,
    ConstraintOp,
    WellKnownAttribute,
    constraints_from_resources,
    device_variant_constraint,
    extract_placement_requirements,
    merge_constraints,
    preemptible_constraint,
    region_constraint,
)
from iris.cluster.types import (
    Entrypoint,
    JobName,
    TaskAttempt,
    adjust_tpu_replicas,
    gpu_device,
    tpu_device,
)
from iris.rpc import job_pb2


def _add(a, b):
    return a + b


def test_entrypoint_from_callable_resolve_roundtrip():
    ep = Entrypoint.from_callable(_add, 3, b=4)
    fn, args, kwargs = ep.resolve()
    assert fn(*args, **kwargs) == 7


def test_entrypoint_proto_roundtrip_preserves_bytes():
    """Bytes survive to_proto -> from_proto without deserialization."""
    ep = Entrypoint.from_callable(_add, 1, 2)
    original_files = ep.workdir_files

    proto = ep.to_proto()
    ep2 = Entrypoint.from_proto(proto)

    assert ep2.workdir_files == original_files
    fn, args, kwargs = ep2.resolve()
    assert fn(*args, **kwargs) == 3


def test_entrypoint_command():
    ep = Entrypoint.from_command("echo", "hello")
    assert not ep.workdir_files
    assert ep.command == ["echo", "hello"]


def test_entrypoint_callable_has_workdir_files():
    ep = Entrypoint.from_callable(_add, 1, 2)
    assert "_callable.pkl" in ep.workdir_files
    assert "_callable_runner.py" in ep.workdir_files
    assert ep.command is not None


def test_job_name_roundtrip_and_hierarchy():
    job = JobName.root("test-user", "root")
    child = job.child("child")
    task = child.task(0)

    assert str(job) == "/test-user/root"
    assert str(child) == "/test-user/root/child"
    assert str(task) == "/test-user/root/child/0"
    assert job.user == "test-user"
    assert child.root_job == job
    assert task.parent == child
    assert child.parent == job
    assert job.parent is None

    parsed = JobName.from_string("/test-user/root/child/0")
    assert parsed == task
    assert parsed.namespace == "test-user/root"
    assert parsed.is_task
    assert parsed.task_index == 0
    assert JobName.root("test-user", "root").is_ancestor_of(parsed)
    assert not parsed.is_ancestor_of(JobName.root("test-user", "root"), include_self=False)


@pytest.mark.parametrize(
    "value",
    ["", "root", "/root", "/test-user//child", "/test-user/root/ ", "/test-user/root/", "/test-user/root//0"],
)
def test_job_name_rejects_invalid_inputs(value: str):
    with pytest.raises(ValueError):
        JobName.from_string(value)


def test_job_name_require_task_errors_on_non_task():
    with pytest.raises(ValueError):
        JobName.from_string("/test-user/root/child").require_task()


def test_job_name_to_safe_token_and_deep_nesting():
    import hashlib

    job = JobName.from_string("/test-user/a/b/c/d/e/0")
    expected_hash = hashlib.sha256(str(job).encode()).hexdigest()
    assert job.to_safe_token() == f"test-user-{expected_hash}"
    assert job.require_task()[1] == 0

    # Deeply nested names still produce short tokens (no ENAMETOOLONG).
    deep = JobName.from_string("/alice/" + "/".join(f"layer-{i}" for i in range(20)) + "/0")
    token = deep.to_safe_token()
    assert token.startswith("alice-")
    assert len(token) < 128  # well under the 255-byte filesystem limit

    # Different names produce different tokens.
    job2 = JobName.from_string("/test-user/a/b/c/d/e/1")
    assert job.to_safe_token() != job2.to_safe_token()


def test_job_name_depth():
    """Job depth increases with hierarchy; tasks inherit parent depth."""
    assert JobName.root("test-user", "train").depth == 1
    assert JobName.from_string("/test-user/train/eval").depth == 2
    assert JobName.from_string("/test-user/train/eval/score").depth == 3
    # Task depth equals parent job depth
    assert JobName.from_string("/test-user/train/0").depth == 1
    assert JobName.from_string("/test-user/train/eval/0").depth == 2


# ---------------------------------------------------------------------------
# TaskAttempt: structured task_id:attempt_id
# ---------------------------------------------------------------------------


def test_task_name_roundtrip_without_attempt():
    tn = TaskAttempt.from_wire("/alice/job/0")
    assert tn.task_id == JobName.from_string("/alice/job/0")
    assert tn.attempt_id is None
    assert tn.to_wire() == "/alice/job/0"
    assert str(tn) == "/alice/job/0"


def test_task_name_roundtrip_with_attempt():
    tn = TaskAttempt.from_wire("/alice/job/0:3")
    assert tn.task_id == JobName.from_string("/alice/job/0")
    assert tn.attempt_id == 3
    assert tn.to_wire() == "/alice/job/0:3"


def test_task_name_require_attempt():
    tn = TaskAttempt.from_wire("/alice/job/0:5")
    assert tn.require_attempt() == 5

    tn_no_attempt = TaskAttempt.from_wire("/alice/job/0")
    with pytest.raises(ValueError, match="no attempt_id"):
        tn_no_attempt.require_attempt()


def test_task_name_job_id_and_task_index():
    tn = TaskAttempt.from_wire("/alice/parent/child/0:2")
    assert tn.job_id == JobName.from_string("/alice/parent/child")
    assert tn.task_index == 0


def test_task_name_with_and_without_attempt():
    tn = TaskAttempt.from_wire("/alice/job/0")
    with_attempt = tn.with_attempt(7)
    assert with_attempt.attempt_id == 7
    assert with_attempt.task_id == tn.task_id

    without = with_attempt.without_attempt()
    assert without.attempt_id is None
    assert without.task_id == tn.task_id


def test_task_name_from_components():
    task_id = JobName.from_string("/alice/job/0")
    tn = TaskAttempt(task_id=task_id, attempt_id=2)
    assert tn.to_wire() == "/alice/job/0:2"


@pytest.mark.parametrize("value", ["", "not-a-path", "/alice/job/0:notanint"])
def test_task_name_rejects_invalid_inputs(value: str):
    with pytest.raises(ValueError):
        TaskAttempt.from_wire(value)


def test_task_name_attempt_zero():
    """Attempt 0 is a valid attempt_id and must round-trip."""
    tn = TaskAttempt.from_wire("/alice/job/0:0")
    assert tn.attempt_id == 0
    assert tn.to_wire() == "/alice/job/0:0"


# ---------------------------------------------------------------------------
# Helpers for building proto constraints used by the extraction functions.
# ---------------------------------------------------------------------------


def _proto_constraint(key: str, string_value: str, op: int = job_pb2.CONSTRAINT_OP_EQ) -> job_pb2.Constraint:
    """Build a proto Constraint with a string value."""
    return job_pb2.Constraint(
        key=key,
        op=op,
        value=job_pb2.AttributeValue(string_value=string_value),
    )


# ---------------------------------------------------------------------------
# region_constraint (returns a Python Constraint dataclass)
# ---------------------------------------------------------------------------


def test_region_constraint_single_region_produces_eq():
    c = region_constraint(["us-west4"])
    assert c.key == WellKnownAttribute.REGION
    assert c.op == ConstraintOp.EQ
    assert c.value == "us-west4"


def test_region_constraint_multiple_regions_produces_in():
    c = region_constraint(["us-central1", "us-central2"])
    assert c.key == WellKnownAttribute.REGION
    assert c.op == ConstraintOp.IN
    assert c.values == ("us-central1", "us-central2")
    assert c.value is None


def test_region_constraint_empty_list_raises():
    with pytest.raises(ValueError, match="non-empty"):
        region_constraint([])


def test_region_constraint_empty_string_raises():
    with pytest.raises(ValueError, match="non-empty"):
        region_constraint([""])


# ---------------------------------------------------------------------------
# extract_placement_requirements: preemptible field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("true", True),
        ("false", False),
    ],
)
def test_preemptible_preference_returns_bool(raw_value: str, expected: bool):
    constraints = [_proto_constraint(WellKnownAttribute.PREEMPTIBLE, raw_value)]
    assert extract_placement_requirements(constraints).preemptible is expected


def test_preemptible_preference_none_when_absent():
    constraints = [_proto_constraint(WellKnownAttribute.REGION, "us-west4")]
    assert extract_placement_requirements(constraints).preemptible is None


def test_preemptible_preference_conflicting_raises():
    constraints = [
        _proto_constraint(WellKnownAttribute.PREEMPTIBLE, "true"),
        _proto_constraint(WellKnownAttribute.PREEMPTIBLE, "false"),
    ]
    with pytest.raises(ValueError, match="conflicting"):
        extract_placement_requirements(constraints)


def test_preemptible_preference_invalid_value_raises():
    constraints = [_proto_constraint(WellKnownAttribute.PREEMPTIBLE, "maybe")]
    with pytest.raises(ValueError, match="'true' or 'false'"):
        extract_placement_requirements(constraints)


# ---------------------------------------------------------------------------
# extract_placement_requirements: required_regions field
# ---------------------------------------------------------------------------


def test_required_regions_single():
    constraints = [_proto_constraint(WellKnownAttribute.REGION, "eu-west4")]
    assert extract_placement_requirements(constraints).required_regions == frozenset({"eu-west4"})


def test_required_regions_none_when_absent():
    constraints = [_proto_constraint(WellKnownAttribute.PREEMPTIBLE, "true")]
    assert extract_placement_requirements(constraints).required_regions is None


def test_required_regions_conflicting_raises():
    constraints = [
        _proto_constraint(WellKnownAttribute.REGION, "us-west4"),
        _proto_constraint(WellKnownAttribute.REGION, "eu-west4"),
    ]
    with pytest.raises(ValueError, match="conflicting"):
        extract_placement_requirements(constraints)


def test_required_regions_empty_string_raises():
    constraints = [_proto_constraint(WellKnownAttribute.REGION, "")]
    with pytest.raises(ValueError, match="non-empty"):
        extract_placement_requirements(constraints)


# ---------------------------------------------------------------------------
# extract_placement_requirements: required_zones field
# ---------------------------------------------------------------------------


def test_required_zones_single():
    constraints = [_proto_constraint(WellKnownAttribute.ZONE, "us-central2-b")]
    assert extract_placement_requirements(constraints).required_zones == frozenset({"us-central2-b"})


def test_required_zones_none_when_absent():
    constraints = [_proto_constraint(WellKnownAttribute.PREEMPTIBLE, "true")]
    assert extract_placement_requirements(constraints).required_zones is None


def test_required_zones_conflicting_raises():
    constraints = [
        _proto_constraint(WellKnownAttribute.ZONE, "us-central2-a"),
        _proto_constraint(WellKnownAttribute.ZONE, "us-central2-b"),
    ]
    with pytest.raises(ValueError, match="conflicting"):
        extract_placement_requirements(constraints)


def test_required_zones_empty_string_raises():
    constraints = [_proto_constraint(WellKnownAttribute.ZONE, "")]
    with pytest.raises(ValueError, match="non-empty"):
        extract_placement_requirements(constraints)


# ---------------------------------------------------------------------------
# extract_placement_requirements (proto inputs, combines both extractors)
# ---------------------------------------------------------------------------


def test_extract_placement_requirements_combines_fields():
    constraints = [
        _proto_constraint(WellKnownAttribute.PREEMPTIBLE, "true"),
        _proto_constraint(WellKnownAttribute.REGION, "us-central1"),
        _proto_constraint(WellKnownAttribute.ZONE, "us-central1-a"),
    ]
    nc = extract_placement_requirements(constraints)
    assert nc.preemptible is True
    assert nc.required_regions == frozenset({"us-central1"})
    assert nc.required_zones == frozenset({"us-central1-a"})


# ---------------------------------------------------------------------------
# merge_constraints (Python Constraint dataclass inputs)
# ---------------------------------------------------------------------------


def test_merge_parent_only():
    """Child has no constraints -- parent constraints pass through."""
    parent = [region_constraint(["us-west4"]), preemptible_constraint(True)]
    result = merge_constraints(parent, [])
    assert set(result) == set(parent)


def test_merge_child_overrides_region():
    parent = [region_constraint(["us-west4"])]
    child = [region_constraint(["eu-west4"])]
    result = merge_constraints(parent, child)
    regions = [c for c in result if c.key == WellKnownAttribute.REGION]
    assert len(regions) == 1
    assert regions[0].value == "eu-west4"


def test_merge_child_overrides_preemptible():
    parent = [preemptible_constraint(True)]
    child = [preemptible_constraint(False)]
    result = merge_constraints(parent, child)
    preemptibles = [c for c in result if c.key == WellKnownAttribute.PREEMPTIBLE]
    assert len(preemptibles) == 1
    assert preemptibles[0].value == "false"


def test_merge_child_overrides_zone():
    """Child zone constraint replaces parent's (zone is canonical)."""
    parent = [Constraint(key=WellKnownAttribute.ZONE, op=ConstraintOp.EQ, value="a")]
    child = [Constraint(key=WellKnownAttribute.ZONE, op=ConstraintOp.EQ, value="b")]
    result = merge_constraints(parent, child)
    zone_constraints = [c for c in result if c.key == WellKnownAttribute.ZONE]
    assert len(zone_constraints) == 1
    assert zone_constraints[0].value == "b"


def test_merge_non_canonical_key_both_present():
    """Non-canonical keys from parent and child are both kept."""
    parent = [Constraint(key=WellKnownAttribute.TPU_NAME, op=ConstraintOp.EQ, value="a")]
    child = [Constraint(key=WellKnownAttribute.TPU_NAME, op=ConstraintOp.EQ, value="b")]
    result = merge_constraints(parent, child)
    tpu_constraints = [c for c in result if c.key == WellKnownAttribute.TPU_NAME]
    assert len(tpu_constraints) == 2
    assert {c.value for c in tpu_constraints} == {"a", "b"}


def test_merge_non_canonical_key_dedup():
    """Duplicate non-canonical constraints are deduplicated."""
    shared = Constraint(key=WellKnownAttribute.TPU_NAME, op=ConstraintOp.EQ, value="a")
    result = merge_constraints([shared], [shared])
    tpu_constraints = [c for c in result if c.key == WellKnownAttribute.TPU_NAME]
    assert len(tpu_constraints) == 1


def test_merge_multiple_canonical_keys_partial_override():
    """Child overrides region but inherits preemptible from parent."""
    parent = [region_constraint(["us-west4"]), preemptible_constraint(True)]
    child = [region_constraint(["eu-west4"])]
    result = merge_constraints(parent, child)

    regions = [c for c in result if c.key == WellKnownAttribute.REGION]
    assert len(regions) == 1
    assert regions[0].value == "eu-west4"

    preemptibles = [c for c in result if c.key == WellKnownAttribute.PREEMPTIBLE]
    assert len(preemptibles) == 1
    assert preemptibles[0].value == "true"


def test_region_constraint_empty_string_in_multi_raises():
    with pytest.raises(ValueError, match="non-empty"):
        region_constraint(["us-central1", ""])


# ---------------------------------------------------------------------------
# Constraint.to_proto / from_proto round-trip for IN operator
# ---------------------------------------------------------------------------


def test_constraint_in_proto_roundtrip():
    """IN constraint survives a proto round-trip."""
    original = Constraint(key=WellKnownAttribute.REGION, op=ConstraintOp.IN, values=("us-central1", "eu-west4"))
    proto = original.to_proto()
    assert proto.op == job_pb2.CONSTRAINT_OP_IN
    assert len(proto.values) == 2
    restored = Constraint.from_proto(proto)
    assert restored == original


# ---------------------------------------------------------------------------
# extract_placement_requirements: IN operator for regions (proto inputs)
# ---------------------------------------------------------------------------


def _proto_in_constraint(key: str, string_values: list[str]) -> job_pb2.Constraint:
    """Build a proto Constraint with IN op and multiple string values."""
    c = job_pb2.Constraint(key=key, op=job_pb2.CONSTRAINT_OP_IN)
    for sv in string_values:
        c.values.append(job_pb2.AttributeValue(string_value=sv))
    return c


def test_required_regions_in_multiple():
    constraints = [_proto_in_constraint(WellKnownAttribute.REGION, ["us-central1", "us-central2"])]
    result = extract_placement_requirements(constraints).required_regions
    assert result == frozenset({"us-central1", "us-central2"})


def test_required_regions_in_single():
    constraints = [_proto_in_constraint(WellKnownAttribute.REGION, ["eu-west4"])]
    result = extract_placement_requirements(constraints).required_regions
    assert result == frozenset({"eu-west4"})


def test_required_regions_in_empty_values_raises():
    """IN constraint with no values is invalid."""
    c = job_pb2.Constraint(key=WellKnownAttribute.REGION, op=job_pb2.CONSTRAINT_OP_IN)
    with pytest.raises(ValueError, match="at least one value"):
        extract_placement_requirements([c])


def test_extract_placement_requirements_with_in_region():
    """extract_placement_requirements works with IN region constraints."""
    constraints = [
        _proto_constraint(WellKnownAttribute.PREEMPTIBLE, "false"),
        _proto_in_constraint(WellKnownAttribute.REGION, ["us-central1", "us-central2"]),
        _proto_constraint(WellKnownAttribute.ZONE, "us-central2-b"),
    ]
    nc = extract_placement_requirements(constraints)
    assert nc.preemptible is False
    assert nc.required_regions == frozenset({"us-central1", "us-central2"})
    assert nc.required_zones == frozenset({"us-central2-b"})


# ---------------------------------------------------------------------------
# constraints_from_resources
# ---------------------------------------------------------------------------


def test_constraints_from_resources_tpu():
    """TPU resource spec produces device-type and device-variant constraints."""
    resources = job_pb2.ResourceSpecProto(cpu_millicores=2000)
    resources.device.CopyFrom(tpu_device("v5litepod-16"))
    result = constraints_from_resources(resources)
    keys = {c.key for c in result}
    assert WellKnownAttribute.DEVICE_TYPE in keys
    assert WellKnownAttribute.DEVICE_VARIANT in keys
    type_c = next(c for c in result if c.key == WellKnownAttribute.DEVICE_TYPE)
    assert type_c.value == "tpu"
    variant_c = next(c for c in result if c.key == WellKnownAttribute.DEVICE_VARIANT)
    assert variant_c.value == "v5litepod-16"


def test_constraints_from_resources_gpu():
    resources = job_pb2.ResourceSpecProto(cpu_millicores=2000)
    resources.device.CopyFrom(gpu_device("H100", count=8))
    result = constraints_from_resources(resources)
    type_c = next(c for c in result if c.key == WellKnownAttribute.DEVICE_TYPE)
    assert type_c.value == "gpu"
    variant_c = next(c for c in result if c.key == WellKnownAttribute.DEVICE_VARIANT)
    assert variant_c.value == "h100"


def test_constraints_from_resources_cpu_produces_nothing():
    """CPU-only resource spec produces no device constraints."""
    resources = job_pb2.ResourceSpecProto(cpu_millicores=2000)
    resources.device.CopyFrom(job_pb2.DeviceConfig(cpu=job_pb2.CpuDevice()))
    result = constraints_from_resources(resources)
    assert result == []


def test_constraints_from_resources_no_device():
    """Resource spec with no device field produces no constraints."""
    resources = job_pb2.ResourceSpecProto(cpu_millicores=2000)
    result = constraints_from_resources(resources)
    assert result == []


def test_constraints_from_resources_auto_variant_skipped():
    """Variant 'auto' is not emitted as a constraint."""
    resources = job_pb2.ResourceSpecProto()
    resources.device.CopyFrom(job_pb2.DeviceConfig(gpu=job_pb2.GpuDevice(variant="auto", count=1)))
    result = constraints_from_resources(resources)
    assert len(result) == 1
    assert result[0].key == WellKnownAttribute.DEVICE_TYPE


# ---------------------------------------------------------------------------
# merge_constraints: device-type is a canonical key
# ---------------------------------------------------------------------------


def test_merge_child_overrides_device_type():
    """Child device-type constraint replaces parent's."""
    parent = [Constraint(key=WellKnownAttribute.DEVICE_TYPE, op=ConstraintOp.EQ, value="tpu")]
    child = [Constraint(key=WellKnownAttribute.DEVICE_TYPE, op=ConstraintOp.EQ, value="gpu")]
    result = merge_constraints(parent, child)
    dt = [c for c in result if c.key == WellKnownAttribute.DEVICE_TYPE]
    assert len(dt) == 1
    assert dt[0].value == "gpu"


# ---------------------------------------------------------------------------
# adjust_tpu_replicas
# ---------------------------------------------------------------------------


def test_adjust_tpu_replicas_auto_scales_to_vm_count():
    """replicas=1 auto-scales to vm_count for multi-host topologies."""
    assert adjust_tpu_replicas(tpu_device("v6e-32"), replicas=1) == 8
    assert adjust_tpu_replicas(tpu_device("v5litepod-16"), replicas=1) == 4


def test_adjust_tpu_replicas_rejects_invalid_count():
    with pytest.raises(ValueError, match="replicas must be a multiple of 4"):
        adjust_tpu_replicas(tpu_device("v5litepod-16"), replicas=3)


def test_adjust_tpu_replicas_correct_counts_pass_through():
    assert adjust_tpu_replicas(tpu_device("v6e-32"), replicas=8) == 8
    assert adjust_tpu_replicas(tpu_device("v6e-32"), replicas=16) == 16


def test_adjust_tpu_replicas_single_host_and_edge_cases():
    """Single-host, no device, and unknown topology return replicas unchanged."""
    assert adjust_tpu_replicas(tpu_device("v6e-4"), replicas=1) == 1
    assert adjust_tpu_replicas(None, replicas=1) == 1
    assert adjust_tpu_replicas(tpu_device("v99-unknown", count=4), replicas=1) == 1


def test_merge_auto_constraints_with_user_variant_override():
    """User multi-variant IN constraint replaces auto-generated single-variant EQ."""
    auto = constraints_from_resources(
        job_pb2.ResourceSpecProto(
            cpu_millicores=2000,
            device=tpu_device("v5litepod-16"),
        )
    )
    user = [device_variant_constraint(["v5litepod-16", "v6e-16"])]
    merged = merge_constraints(auto, user)

    # device-type from auto should be kept
    dt = [c for c in merged if c.key == WellKnownAttribute.DEVICE_TYPE]
    assert len(dt) == 1
    assert dt[0].value == "tpu"

    # device-variant should be the user's IN constraint, not auto's EQ
    dv = [c for c in merged if c.key == WellKnownAttribute.DEVICE_VARIANT]
    assert len(dv) == 1
    assert dv[0].op == ConstraintOp.IN
    assert dv[0].values == ("v5litepod-16", "v6e-16")
