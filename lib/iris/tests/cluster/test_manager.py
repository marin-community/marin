# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for cluster manager lifecycle functions."""

from unittest.mock import patch

from iris.cluster.manager import stop_all
from iris.cluster.platform.base import CloudSliceState
from iris.rpc import config_pb2
from iris.time_utils import Timestamp
from tests.cluster.platform.fakes import FakePlatform, FakePlatformConfig

DEFAULT_RESOURCES = config_pb2.ScaleGroupResources(
    cpu=128,
    memory_bytes=128 * 1024**3,
    disk_bytes=100 * 1024**3,
    tpu_count=8,
)


def _make_scale_group_config(name: str, zone: str = "us-central1-a") -> config_pb2.ScaleGroupConfig:
    """Create a scale group config for testing."""
    config = config_pb2.ScaleGroupConfig(
        name=name,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5p-8",
        slice_size=1,
    )
    config.resources.CopyFrom(DEFAULT_RESOURCES)
    config.slice_template.gcp.zones.append(zone)
    return config


def _make_cluster_config(scale_groups: dict[str, config_pb2.ScaleGroupConfig]) -> config_pb2.IrisClusterConfig:
    """Create a minimal cluster config for testing stop_all."""
    config = config_pb2.IrisClusterConfig()
    config.controller.local.SetInParent()
    config.platform.local.SetInParent()
    for name, sg_config in scale_groups.items():
        config.scale_groups[name].CopyFrom(sg_config)
    return config


def test_stop_all_terminates_slices():
    """stop_all() terminates all slices in each scale group."""
    sg_config = _make_scale_group_config("test-group")
    platform = FakePlatform(FakePlatformConfig(config=sg_config))

    # Create two slices
    slice1 = platform.create_slice(config_pb2.SliceConfig(labels={"iris-scale-group": "test-group"}))
    slice2 = platform.create_slice(config_pb2.SliceConfig(labels={"iris-scale-group": "test-group"}))

    # Advance time to get slices ready
    platform.tick(Timestamp.now().epoch_ms())

    # Verify slices are ready before stop_all
    assert slice1.status().state == CloudSliceState.READY
    assert slice2.status().state == CloudSliceState.READY
    assert len(platform.list_slices(zones=["us-central1-a"], labels={"iris-scale-group": "test-group"})) == 2

    # Call stop_all
    cluster_config = _make_cluster_config({"test-group": sg_config})
    with patch("iris.cluster.manager.IrisConfig") as mock_iris_config_cls:
        mock_iris_config_cls.return_value.platform.return_value = platform
        stop_all(cluster_config)

    # Verify slices are terminated
    assert slice1.status().state == CloudSliceState.DELETING
    assert slice2.status().state == CloudSliceState.DELETING


def test_stop_all_filters_slices_by_scale_group():
    """stop_all() only terminates slices matching the scale group label."""
    sg_a_config = _make_scale_group_config("group-a")

    # Use group-a config for platform to avoid topology mismatch
    platform = FakePlatform(FakePlatformConfig(config=sg_a_config))

    # Create slices with different scale group labels
    slice_a = platform.create_slice(config_pb2.SliceConfig(labels={"iris-scale-group": "group-a"}))
    slice_b = platform.create_slice(config_pb2.SliceConfig(labels={"iris-scale-group": "group-b"}))

    platform.tick(Timestamp.now().epoch_ms())

    # Call stop_all with only group-a in config
    cluster_config = _make_cluster_config({"group-a": sg_a_config})
    with patch("iris.cluster.manager.IrisConfig") as mock_iris_config_cls:
        mock_iris_config_cls.return_value.platform.return_value = platform
        stop_all(cluster_config)

    # Only slice_a should be terminated
    assert slice_a.status().state == CloudSliceState.DELETING
    assert slice_b.status().state == CloudSliceState.READY


def test_stop_all_handles_empty_scale_groups():
    """stop_all() succeeds when there are no slices to terminate."""
    sg_config = _make_scale_group_config("empty-group")
    platform = FakePlatform(FakePlatformConfig(config=sg_config))

    # Don't create any slices
    cluster_config = _make_cluster_config({"empty-group": sg_config})
    with patch("iris.cluster.manager.IrisConfig") as mock_iris_config_cls:
        mock_iris_config_cls.return_value.platform.return_value = platform
        stop_all(cluster_config)

    # Should complete without error, no slices to verify
