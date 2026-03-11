# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.cli.job — validate_region_zone and related CLI validation."""

import click
import pytest

from iris.cli.job import validate_region_zone
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
