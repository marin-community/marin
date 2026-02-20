# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ray-based cluster implementation."""

from fray.v1.cluster.ray.cluster import RayCluster
from fray.v1.cluster.ray.dashboard import (
    DashboardConfig,
    DashboardConnection,
    ray_dashboard,
)
from fray.v1.cluster.ray.resources import accelerator_descriptor, as_remote_kwargs, get_scheduling_strategy

__all__ = [
    "DashboardConfig",
    "DashboardConnection",
    "RayCluster",
    "accelerator_descriptor",
    "as_remote_kwargs",
    "get_scheduling_strategy",
    "ray_dashboard",
]
