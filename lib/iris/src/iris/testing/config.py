# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Configuration factories used by Iris tests."""

from pathlib import Path

from iris.cluster.config import (
    IrisClusterConfig,
    LocalSliceConfig,
    ScaleGroupConfig,
    ScaleGroupResources,
    SliceConfig,
    load_config,
    make_local_config,
)
from iris.cluster.types import AcceleratorType, CapacityType

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "ci-test.yaml"


def make_controller_only_config() -> IrisClusterConfig:
    """Build a null-auth local config with no auto-scaled workers."""
    config = load_config(DEFAULT_CONFIG)
    config.scale_groups = {
        "placeholder": ScaleGroupConfig(
            name="placeholder",
            num_vms=1,
            buffer_slices=0,
            max_slices=0,
            resources=ScaleGroupResources(
                cpu_millicores=1000,
                memory_bytes=1 * 1024**3,
                disk_bytes=10 * 1024**3,
                device_type=AcceleratorType.CPU,
                capacity_type=CapacityType.ON_DEMAND,
            ),
            slice_template=SliceConfig(local=LocalSliceConfig()),
        )
    }
    return make_local_config(config)
