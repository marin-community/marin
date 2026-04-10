# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.cli.job — validate_region_zone, executor heuristic, and related CLI validation."""

import click
import pytest

from iris.cli.job import (
    _parse_tpu_alternatives,
    build_resources,
    build_tpu_alternatives,
    validate_region_zone,
)
from iris.cluster.constraints import (
    Constraint,
    WellKnownAttribute,
    infer_preemptible_constraint,
    preemptible_constraint,
    region_constraint,
)
from iris.rpc import config_pb2


def _make_config_with_zones(zones: list[str]) -> config_pb2.IrisClusterConfig:
    """Build a minimal IrisClusterConfig with scale groups for the given zones."""
    config = config_pb2.IrisClusterConfig()
    for zone in zones:
        region = zone.rsplit("-", 1)[0]
        sg = config.scale_groups[f"sg-{zone}"]
        sg.worker.attributes["zone"] = zone
        sg.worker.attributes["region"] = region
    return config


def test_validate_region_zone_valid_region():
    config = _make_config_with_zones(["us-central2-b", "europe-west4-a"])
    validate_region_zone(("us-central2",), None, config)


def test_validate_region_zone_valid_zone():
    config = _make_config_with_zones(["us-central2-b", "europe-west4-a"])
    validate_region_zone(None, "europe-west4-a", config)


def test_validate_region_zone_invalid_region_raises():
    config = _make_config_with_zones(["us-central2-b", "europe-west4-a"])
    with pytest.raises(click.BadParameter, match=r"eu-west4.*not a known region"):
        validate_region_zone(("eu-west4",), None, config)


def test_validate_region_zone_invalid_region_suggests_closest():
    config = _make_config_with_zones(["us-central2-b", "europe-west4-a"])
    with pytest.raises(click.BadParameter, match="Did you mean 'europe-west4'"):
        validate_region_zone(("eu-west4",), None, config)


def test_validate_region_zone_invalid_zone_raises():
    config = _make_config_with_zones(["us-central2-b", "europe-west4-a"])
    with pytest.raises(click.BadParameter, match=r"us-central2-a.*not a known zone"):
        validate_region_zone(None, "us-central2-a", config)


def test_validate_region_zone_invalid_zone_suggests_closest():
    config = _make_config_with_zones(["us-central2-b", "europe-west4-a"])
    with pytest.raises(click.BadParameter, match="Did you mean 'us-central2-b'"):
        validate_region_zone(None, "us-central2-a", config)


def test_validate_region_zone_no_config_skips():
    validate_region_zone(("nonexistent",), "nonexistent", None)


def test_validate_region_zone_no_constraints_skips():
    config = _make_config_with_zones(["us-central2-b"])
    validate_region_zone(None, None, config)


# ---------------------------------------------------------------------------
# Executor heuristic tests (mirrors the logic in run_iris_job)
# ---------------------------------------------------------------------------


def test_executor_heuristic_small_cpu_job_gets_non_preemptible():
    resources = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB")
    resources_proto = resources.to_proto()
    replicas = 1
    constraints: list[Constraint] = []

    preemptible = infer_preemptible_constraint(resources_proto, replicas, constraints)
    assert preemptible is not None
    assert preemptible.key == WellKnownAttribute.PREEMPTIBLE
    assert preemptible.value == "false"


def test_executor_heuristic_skipped_for_gpu_job():
    resources = build_resources(tpu=None, gpu="H100", cpu=0.5, memory="1GB", disk="5GB")
    resources_proto = resources.to_proto()
    replicas = 1
    constraints: list[Constraint] = []

    preemptible = infer_preemptible_constraint(resources_proto, replicas, constraints)
    assert preemptible is None


def test_executor_heuristic_skipped_for_large_cpu_job():
    resources = build_resources(tpu=None, gpu=None, cpu=4.0, memory="16GB", disk="5GB")
    resources_proto = resources.to_proto()
    replicas = 1
    constraints: list[Constraint] = []

    preemptible = infer_preemptible_constraint(resources_proto, replicas, constraints)
    assert preemptible is None


def test_executor_heuristic_skipped_when_user_sets_preemptible():
    resources = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB")
    resources_proto = resources.to_proto()
    replicas = 1
    constraints: list[Constraint] = [preemptible_constraint(True)]

    preemptible = infer_preemptible_constraint(resources_proto, replicas, constraints)
    assert preemptible is None


def test_executor_heuristic_with_region_constraint():
    resources = build_resources(tpu=None, gpu=None, cpu=0.5, memory="1GB", disk="5GB")
    resources_proto = resources.to_proto()
    replicas = 1
    constraints: list[Constraint] = [region_constraint(["us-central2"])]

    preemptible = infer_preemptible_constraint(resources_proto, replicas, constraints)
    assert preemptible is not None
    assert preemptible.value == "false"


# ---------------------------------------------------------------------------
# --tpu multi-variant parsing tests
# ---------------------------------------------------------------------------


def test_parse_tpu_single_variant():
    primary, alternatives = _parse_tpu_alternatives("v6e-4")
    assert primary == "v6e-4"
    assert alternatives == []


def test_parse_tpu_multiple_variants():
    primary, alternatives = _parse_tpu_alternatives("v6e-4,v5litepod-4,v5p-8")
    assert primary == "v6e-4"
    assert alternatives == ["v5litepod-4", "v5p-8"]


def test_parse_tpu_strips_whitespace():
    primary, alternatives = _parse_tpu_alternatives(" v6e-4 , v5litepod-4 ")
    assert primary == "v6e-4"
    assert alternatives == ["v5litepod-4"]


def test_parse_tpu_empty_raises():
    with pytest.raises(click.BadParameter, match="at least one"):
        _parse_tpu_alternatives("")


def test_parse_tpu_only_commas_raises():
    with pytest.raises(click.BadParameter, match="at least one"):
        _parse_tpu_alternatives(", ,")


def test_parse_tpu_mismatched_vm_count_raises():
    # v5p-8 has vm_count=1 but v5p-16 has vm_count=2 — multinode mismatch.
    with pytest.raises(click.BadParameter, match="vm_count"):
        _parse_tpu_alternatives("v5p-8,v5p-16")


def test_build_tpu_alternatives_none():
    assert build_tpu_alternatives(None) == []


def test_build_tpu_alternatives_returns_full_list():
    assert build_tpu_alternatives("v6e-4,v5litepod-4,v5p-8") == ["v6e-4", "v5litepod-4", "v5p-8"]


def test_build_resources_uses_first_variant_as_primary():
    spec = build_resources(tpu="v6e-4,v5litepod-4,v5p-8", gpu=None, cpu=8.0, memory="32GB", disk="50GB")
    assert spec.device is not None
    assert spec.device.HasField("tpu")
    assert spec.device.tpu.variant == "v6e-4"
