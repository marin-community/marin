# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.cluster.types — Entrypoint, EnvironmentSpec, and constraint helpers."""

import pytest

from iris.cluster.types import (
    Constraint,
    ConstraintOp,
    Entrypoint,
    JobName,
    merge_constraints,
    normalize_constraints,
    preemptible_constraint,
    preemptible_preference_from_constraints,
    region_constraint,
    required_regions_from_constraints,
)
from iris.rpc import cluster_pb2


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
    job = JobName.root("root")
    child = job.child("child")
    task = child.task(0)

    assert str(job) == "/root"
    assert str(child) == "/root/child"
    assert str(task) == "/root/child/0"
    assert task.parent == child
    assert child.parent == job
    assert job.parent is None

    parsed = JobName.from_string("/root/child/0")
    assert parsed == task
    assert parsed.namespace == "root"
    assert parsed.is_task
    assert parsed.task_index == 0
    assert JobName.root("root").is_ancestor_of(parsed)
    assert not parsed.is_ancestor_of(JobName.root("root"), include_self=False)


@pytest.mark.parametrize(
    "value",
    ["", "root", "/root//child", "/root/ ", "/root/child/", "/root/child//0"],
)
def test_job_name_rejects_invalid_inputs(value: str):
    with pytest.raises(ValueError):
        JobName.from_string(value)


def test_job_name_require_task_errors_on_non_task():
    with pytest.raises(ValueError):
        JobName.from_string("/root/child").require_task()


def test_job_name_to_safe_token_and_deep_nesting():
    job = JobName.from_string("/a/b/c/d/e/0")
    assert job.to_safe_token() == "job__a__b__c__d__e__0"
    assert job.require_task()[1] == 0


def test_job_name_depth():
    """Job depth increases with hierarchy; tasks inherit parent depth."""
    assert JobName.root("train").depth == 1
    assert JobName.from_string("/train/eval").depth == 2
    assert JobName.from_string("/train/eval/score").depth == 3
    # Task depth equals parent job depth
    assert JobName.from_string("/train/0").depth == 1
    assert JobName.from_string("/train/eval/0").depth == 2


# ---------------------------------------------------------------------------
# Helpers for building proto constraints used by the extraction functions.
# ---------------------------------------------------------------------------


def _proto_constraint(key: str, string_value: str, op: int = cluster_pb2.CONSTRAINT_OP_EQ) -> cluster_pb2.Constraint:
    """Build a proto Constraint with a string value."""
    return cluster_pb2.Constraint(
        key=key,
        op=op,
        value=cluster_pb2.AttributeValue(string_value=string_value),
    )


# ---------------------------------------------------------------------------
# region_constraint (returns a Python Constraint dataclass)
# ---------------------------------------------------------------------------


def test_region_constraint_happy_path():
    c = region_constraint("us-west4")
    assert c.key == "region"
    assert c.op == ConstraintOp.EQ
    assert c.value == "us-west4"


def test_region_constraint_empty_string_raises():
    with pytest.raises(ValueError, match="non-empty"):
        region_constraint("")


# ---------------------------------------------------------------------------
# preemptible_preference_from_constraints (proto inputs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("true", True),
        ("false", False),
    ],
)
def test_preemptible_preference_returns_bool(raw_value: str, expected: bool):
    constraints = [_proto_constraint("preemptible", raw_value)]
    assert preemptible_preference_from_constraints(constraints) is expected


def test_preemptible_preference_none_when_absent():
    constraints = [_proto_constraint("region", "us-west4")]
    assert preemptible_preference_from_constraints(constraints) is None


def test_preemptible_preference_conflicting_raises():
    constraints = [
        _proto_constraint("preemptible", "true"),
        _proto_constraint("preemptible", "false"),
    ]
    with pytest.raises(ValueError, match="conflicting"):
        preemptible_preference_from_constraints(constraints)


def test_preemptible_preference_invalid_value_raises():
    constraints = [_proto_constraint("preemptible", "maybe")]
    with pytest.raises(ValueError, match="'true' or 'false'"):
        preemptible_preference_from_constraints(constraints)


# ---------------------------------------------------------------------------
# required_regions_from_constraints (proto inputs)
# ---------------------------------------------------------------------------


def test_required_regions_single():
    constraints = [_proto_constraint("region", "eu-west4")]
    assert required_regions_from_constraints(constraints) == frozenset({"eu-west4"})


def test_required_regions_none_when_absent():
    constraints = [_proto_constraint("preemptible", "true")]
    assert required_regions_from_constraints(constraints) is None


def test_required_regions_conflicting_raises():
    constraints = [
        _proto_constraint("region", "us-west4"),
        _proto_constraint("region", "eu-west4"),
    ]
    with pytest.raises(ValueError, match="conflicting"):
        required_regions_from_constraints(constraints)


def test_required_regions_empty_string_raises():
    constraints = [_proto_constraint("region", "")]
    with pytest.raises(ValueError, match="non-empty"):
        required_regions_from_constraints(constraints)


# ---------------------------------------------------------------------------
# normalize_constraints (proto inputs, combines both extractors)
# ---------------------------------------------------------------------------


def test_normalize_constraints_combines_fields():
    constraints = [
        _proto_constraint("preemptible", "true"),
        _proto_constraint("region", "us-central1"),
    ]
    nc = normalize_constraints(constraints)
    assert nc.preemptible is True
    assert nc.required_regions == frozenset({"us-central1"})


# ---------------------------------------------------------------------------
# merge_constraints (Python Constraint dataclass inputs)
# ---------------------------------------------------------------------------


def test_merge_parent_only():
    """Child has no constraints -- parent constraints pass through."""
    parent = [region_constraint("us-west4"), preemptible_constraint(True)]
    result = merge_constraints(parent, [])
    assert set(result) == set(parent)


def test_merge_child_overrides_region():
    parent = [region_constraint("us-west4")]
    child = [region_constraint("eu-west4")]
    result = merge_constraints(parent, child)
    regions = [c for c in result if c.key == "region"]
    assert len(regions) == 1
    assert regions[0].value == "eu-west4"


def test_merge_child_overrides_preemptible():
    parent = [preemptible_constraint(True)]
    child = [preemptible_constraint(False)]
    result = merge_constraints(parent, child)
    preemptibles = [c for c in result if c.key == "preemptible"]
    assert len(preemptibles) == 1
    assert preemptibles[0].value == "false"


def test_merge_non_canonical_key_both_present():
    """Non-canonical keys from parent and child are both kept."""
    parent = [Constraint(key="zone", op=ConstraintOp.EQ, value="a")]
    child = [Constraint(key="zone", op=ConstraintOp.EQ, value="b")]
    result = merge_constraints(parent, child)
    zone_constraints = [c for c in result if c.key == "zone"]
    assert len(zone_constraints) == 2
    assert {c.value for c in zone_constraints} == {"a", "b"}


def test_merge_non_canonical_key_dedup():
    """Duplicate non-canonical constraints are deduplicated."""
    shared = Constraint(key="zone", op=ConstraintOp.EQ, value="a")
    result = merge_constraints([shared], [shared])
    zone_constraints = [c for c in result if c.key == "zone"]
    assert len(zone_constraints) == 1


def test_merge_multiple_canonical_keys_partial_override():
    """Child overrides region but inherits preemptible from parent."""
    parent = [region_constraint("us-west4"), preemptible_constraint(True)]
    child = [region_constraint("eu-west4")]
    result = merge_constraints(parent, child)

    regions = [c for c in result if c.key == "region"]
    assert len(regions) == 1
    assert regions[0].value == "eu-west4"

    preemptibles = [c for c in result if c.key == "preemptible"]
    assert len(preemptibles) == 1
    assert preemptibles[0].value == "true"
