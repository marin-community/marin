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


def test_validate_extra_resources():
    # Normal CPU-only job passes without the flag.
    validate_extra_resources(tpu=None, gpu=None, memory="1GB", disk="5GB", enable_extra_resources=False)

    # TPU and GPU blocked without the flag; error names the coordinator pattern.
    with pytest.raises(click.UsageError, match="--tpu requires --enable-extra-resources"):
        validate_extra_resources(tpu="v5litepod-16", gpu=None, memory="1GB", disk="5GB", enable_extra_resources=False)
    with pytest.raises(click.UsageError, match="--gpu requires --enable-extra-resources"):
        validate_extra_resources(tpu=None, gpu="H100x8", memory="1GB", disk="5GB", enable_extra_resources=False)
    with pytest.raises(click.UsageError, match="coordinator"):
        validate_extra_resources(tpu="v5litepod-16", gpu=None, memory="1GB", disk="5GB", enable_extra_resources=False)

    # Memory threshold: >= 4 GB blocked, < 4 GB allowed.
    with pytest.raises(click.UsageError, match=r"--memory 4GB.*--enable-extra-resources"):
        validate_extra_resources(tpu=None, gpu=None, memory="4GB", disk="5GB", enable_extra_resources=False)
    validate_extra_resources(tpu=None, gpu=None, memory="3900MB", disk="5GB", enable_extra_resources=False)

    # Disk threshold: >= 10 GB blocked, < 10 GB allowed.
    with pytest.raises(click.UsageError, match=r"--disk 10GB.*--enable-extra-resources"):
        validate_extra_resources(tpu=None, gpu=None, memory="1GB", disk="10GB", enable_extra_resources=False)
    validate_extra_resources(tpu=None, gpu=None, memory="1GB", disk="9900MB", enable_extra_resources=False)

    # --enable-extra-resources bypasses all checks.
    validate_extra_resources(tpu="v5litepod-16", gpu=None, memory="1GB", disk="5GB", enable_extra_resources=True)
    validate_extra_resources(tpu=None, gpu=None, memory="64GB", disk="5GB", enable_extra_resources=True)
    validate_extra_resources(tpu=None, gpu=None, memory="1GB", disk="100GB", enable_extra_resources=True)
