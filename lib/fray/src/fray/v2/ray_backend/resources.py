# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ray-specific resource utilities for job scheduling."""

from typing import Any

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from fray.v2.types import GpuConfig, ResourceConfig, TpuConfig


def as_remote_kwargs(
    config: ResourceConfig,
    env_vars: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Get kwargs for ray.remote() decoration.

    Args:
        config: Resource configuration
        env_vars: Optional environment variables for runtime_env
    """
    runtime_env = {"env_vars": env_vars} if env_vars else None

    if isinstance(config.device, TpuConfig):
        out: dict[str, Any] = {"num_cpus": 8}
    elif isinstance(config.device, GpuConfig):
        out = {"num_gpus": config.device.count}
        if config.device.variant != "auto":
            out["accelerator_type"] = config.device.variant
    else:
        out = {"num_cpus": 1}

    if runtime_env:
        out["runtime_env"] = runtime_env
    return out


def get_scheduling_strategy(config: ResourceConfig) -> PlacementGroupSchedulingStrategy | None:
    """Create TPU placement group scheduling strategy."""
    if not isinstance(config.device, TpuConfig):
        return None

    tpu_type_head = f"TPU-{config.device.variant}-head"
    bundles: list[dict[str, int]] = [{"TPU": 1, "CPU": 1}] * config.chip_count()
    bundles.append({tpu_type_head: 1})

    pg = ray.util.placement_group(bundles, strategy="STRICT_PACK")
    return PlacementGroupSchedulingStrategy(pg, placement_group_capture_child_tasks=True)


def accelerator_descriptor(config: ResourceConfig) -> str | None:
    """Get accelerator type string (e.g., 'v4-8', 'H100') for logging/tracking."""
    if isinstance(config.device, TpuConfig):
        return config.device.variant
    elif isinstance(config.device, GpuConfig):
        return config.device.variant
    return None
