# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.cli.job — validate_region_zone, executor heuristic, and related CLI validation."""

import click
import pytest

from iris.cli.job import build_resources, validate_extra_resources, validate_region_zone
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
# validate_extra_resources tests
# ---------------------------------------------------------------------------


def test_extra_resources_normal_job_passes():
    # Default CPU-only job with small resources should always pass.
    validate_extra_resources(tpu=None, gpu=None, memory="1GB", disk="5GB", enable_extra_resources=False)


def test_extra_resources_tpu_without_flag_raises():
    with pytest.raises(click.UsageError, match="--tpu requires --enable-extra-resources"):
        validate_extra_resources(tpu="v5litepod-16", gpu=None, memory="1GB", disk="5GB", enable_extra_resources=False)


def test_extra_resources_gpu_without_flag_raises():
    with pytest.raises(click.UsageError, match="--gpu requires --enable-extra-resources"):
        validate_extra_resources(tpu=None, gpu="H100x8", memory="1GB", disk="5GB", enable_extra_resources=False)


def test_extra_resources_large_memory_without_flag_raises():
    with pytest.raises(click.UsageError, match=r"--memory 4GB.*--enable-extra-resources"):
        validate_extra_resources(tpu=None, gpu=None, memory="4GB", disk="5GB", enable_extra_resources=False)


def test_extra_resources_memory_just_below_threshold_passes():
    # 3.9 GB is below 4 GB threshold.
    validate_extra_resources(tpu=None, gpu=None, memory="3900MB", disk="5GB", enable_extra_resources=False)


def test_extra_resources_large_disk_without_flag_raises():
    with pytest.raises(click.UsageError, match=r"--disk 10GB.*--enable-extra-resources"):
        validate_extra_resources(tpu=None, gpu=None, memory="1GB", disk="10GB", enable_extra_resources=False)


def test_extra_resources_disk_just_below_threshold_passes():
    # 9.9 GB is below 10 GB threshold.
    validate_extra_resources(tpu=None, gpu=None, memory="1GB", disk="9900MB", enable_extra_resources=False)


def test_extra_resources_flag_bypasses_tpu_check():
    # With --enable-extra-resources, TPU is allowed.
    validate_extra_resources(tpu="v5litepod-16", gpu=None, memory="1GB", disk="5GB", enable_extra_resources=True)


def test_extra_resources_flag_bypasses_large_memory():
    validate_extra_resources(tpu=None, gpu=None, memory="64GB", disk="5GB", enable_extra_resources=True)


def test_extra_resources_flag_bypasses_large_disk():
    validate_extra_resources(tpu=None, gpu=None, memory="1GB", disk="100GB", enable_extra_resources=True)


def test_extra_resources_hint_mentions_coordinator():
    # Error message should explain the entrypoint/coordinator concept.
    with pytest.raises(click.UsageError, match="coordinator"):
        validate_extra_resources(tpu="v5litepod-16", gpu=None, memory="1GB", disk="5GB", enable_extra_resources=False)
