# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for autoscaler bootstrap env propagation."""

from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.scaling_group import ScalingGroup
from tests.cluster.platform.fakes import FakePlatform, FakePlatformConfig
from iris.rpc import config_pb2
from iris.time_utils import Duration


def test_per_group_bootstrap_config_injects_accelerator_env_vars():
    scale_group = config_pb2.ScaleGroupConfig(
        name="h100-8x",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_GPU,
        accelerator_variant="H100",
        resources=config_pb2.ScaleGroupResources(
            cpu=64,
            memory_bytes=256 * 1024**3,
            disk_bytes=1024**4,
            gpu_count=8,
        ),
        num_vms=1,
    )
    platform = FakePlatform(FakePlatformConfig(config=scale_group))
    group = ScalingGroup(scale_group, platform=platform)

    base = config_pb2.BootstrapConfig(
        docker_image="test:latest",
        worker_port=10001,
        controller_address="controller:10000",
    )
    autoscaler = Autoscaler(
        scale_groups={group.name: group},
        evaluation_interval=Duration.from_seconds(1.0),
        platform=platform,
        bootstrap_config=base,
    )

    bc = autoscaler._per_group_bootstrap_config(group)

    assert bc is not None
    assert bc.env_vars["IRIS_ACCELERATOR_TYPE"] == "gpu"
    assert bc.env_vars["IRIS_ACCELERATOR_VARIANT"] == "H100"
    assert bc.env_vars["IRIS_GPU_COUNT"] == "8"
