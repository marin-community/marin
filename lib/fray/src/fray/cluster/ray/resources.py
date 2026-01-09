# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ray-specific resource utilities for job scheduling."""

from typing import Any

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from fray.cluster.base import GpuConfig, ResourceConfig, TpuConfig


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


# Map fray device types to Ray accelerator types
FRAY_TO_RAY_ACCEL_TYPE: dict[str, str] = {
    "h100": "H100",
    "h100-pcie": "H100-PCIE",
    "h200": "H200",
    "a100-80g": "A100-80G",
    "a100-40g": "A100-40G",
    "a10": "A10G",
    "a10g": "A10G",
    "a40": "A40",
    "v100": "V100",
    "v100-sxm": "V100-SXM",
    "v100s": "V100S-PCIE",
    "t4": "T4",
    "a6000": "A6000",
    "trn1": "TRN1",
    "v3": "TPU-V3",
    "v4": "TPU-V4",
    "v5litepod": "TPU-V5LITEPOD",
    "v5p": "TPU-V5P",
    "v6e": "TPU-V6E",
    "l4": "L4",
    "l40s": "L40S",
    "gb10": "GB10",
}
