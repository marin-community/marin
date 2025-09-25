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

import dataclasses
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import ray
from levanter.utils.py_utils import logical_cpu_core_count
from levanter.utils.ray_utils import RayResources
from ray.remote_function import RemoteFunction
from ray.runtime_env import RuntimeEnv

from marin.resources_utils import AcceleratorType

logger = logging.getLogger(__name__)


class ResourceConfig(Protocol):
    """
    A configuration for hardware resources. This is used to specify the hardware resources for a task.
    Currently, this is mostly used by training.
    """

    runtime_env: RuntimeEnv
    """
    A Ray runtime environment to use. You can set environment variables and specify
    additional resources to request for this task.
    """

    device_flops_override: float | None = None
    """Optional override for device FLOPS.
    If set, this value will be used instead of looking up in device_flops_map.

    Currently, this is only used by speedrun and not forwarded to Levanter.
    """

    def accelerator_descriptor(self) -> str | None:
        """Returns the accelerator type descriptor for this hardware configuration."""
        return None

    def as_remote_kwargs(self) -> dict:
        """Returns the resource bundle for this hardware configuration."""
        return self.as_ray_resources().to_kwargs()

    def as_ray_resources(self) -> RayResources: ...

    def as_decorator(self) -> Callable[[type], ray.actor.ActorClass] | Callable[[Callable], RemoteFunction]:
        """Returns a ray.remote decorator for this hardware configuration."""
        return ray.remote(**self.as_remote_kwargs())

    def with_env_vars(self, env: dict[str, str] | None = None, /, **kwargs):
        """Returns a new hardware configuration with the given environment variables."""
        new_env = self.runtime_env.get("env_vars", {}) | (env or {}) | kwargs
        return dataclasses.replace(self, runtime_env=RuntimeEnv(**{**self.runtime_env, "env_vars": new_env}))

    def device_flops(self) -> float | None:
        """Returns the peak FLOPs/s for a single device in this configuration."""
        raise NotImplementedError(
            f"device_flops() not implemented for {self.__class__.__name__}. "
            "Either implement this method in your ResourceConfig subclass or provide a device_flops_override."
        )

    def total_device_count(self) -> int:
        """Returns the total number of devices in this configuration."""
        raise NotImplementedError(
            f"total_device_count() not implemented for {self.__class__.__name__}. "
            "This method must be implemented to determine the total number of accelerator devices available."
        )


@dataclass(frozen=True)
class CpuOnlyConfig(ResourceConfig):
    num_cpus: int = dataclasses.field(default_factory=lambda: logical_cpu_core_count())
    """Configuration for local training without specialized hardware."""

    runtime_env: RuntimeEnv = dataclasses.field(default_factory=lambda: RuntimeEnv(env_vars={"JAX_PLATFORMS": "cpu"}))

    device_flops_override: float | None = None
    """Optional override for device FLOPS. If set, this value will be used instead of looking up in device_flops_map."""

    def accelerator_descriptor(self) -> str | None:
        return None

    def as_remote_kwargs(self) -> dict:
        return dict(num_cpus=self.num_cpus, runtime_env=self.runtime_env)

    def as_ray_resources(self) -> RayResources:
        return RayResources(**self.as_remote_kwargs())

    def device_flops(self) -> float | None:
        if self.device_flops_override is not None:
            logger.info(f"Using user-provided device FLOPS override: {self.device_flops_override} for CPU")
            return self.device_flops_override
        raise NotImplementedError(
            "CPU FLOPS are not available by default. "
            "Please provide a device_flops_override to use a CPU with speedrun."
        )

    def total_device_count(self) -> int:
        return 1


@dataclass(frozen=True)
class GpuConfig(ResourceConfig):
    """Configuration for GPU-based training."""

    gpu_count: int = 1
    """Number of GPUs to use for training."""

    runtime_env: RuntimeEnv = dataclasses.field(default_factory=RuntimeEnv)

    accelerator_type: AcceleratorType | None = None
    """Type of GPU to use. Must be one of the Ray accelerator types. For instance, A100-40G, V100, etc.

    The full list is available at https://docs.ray.io/en/latest/ray-core/accelerator-types.html#accelerator-types
    """

    device_flops_override: float | None = None
    """Optional override for device FLOPS. If set, this value will be used instead of looking up in device_flops_map."""

    def __post_init__(self):
        if self.accelerator_type is not None:
            # Ray uses uppercase for accelerator types
            object.__setattr__(self, "accelerator_type", self.accelerator_type.upper())

    def get_accelerator_type(self) -> AcceleratorType | None:
        """Get the accelerator type for Ray."""
        return self.accelerator_type

    def accelerator_descriptor(self) -> str | None:
        return self.accelerator_type

    # NB that Ray doesn't like resources={"GPU": 1} so we have to do this
    def as_remote_kwargs(self) -> dict:
        out: dict[str, Any] = dict(num_gpus=self.gpu_count, runtime_env=self.runtime_env)
        if self.accelerator_type is not None:
            out["accelerator_type"] = self.accelerator_type

        return out

    def as_ray_resources(self) -> RayResources:
        return RayResources(**self.as_remote_kwargs())

    def device_flops(self) -> float:
        """Get the peak FLOPs/s for the GPU type."""
        if self.device_flops_override is not None:
            logger.info(
                f"Using user-provided device FLOPS override: {self.device_flops_override} for GPU type "
                f"{self.accelerator_type}"
            )
            return self.device_flops_override

        from marin.resources_utils import flop_count_per_device_from_accel_type, ray_device_name_to_jax_name_map

        flops = flop_count_per_device_from_accel_type(self.accelerator_type)

        if flops is None:
            raise ValueError(
                f"No FLOPs data available for accelerator type: {self.accelerator_type}. "
                "Available types: "
                + ", ".join(ray_device_name_to_jax_name_map.keys())
                + "\n"
                + "You can provide a custom FLOPS value using device_flops_override."
            )
        return flops

    def total_device_count(self) -> int:
        return self.gpu_count

    def as_json_dict(self) -> dict:
        """Convert GpuConfig to a JSON-serializable dictionary."""
        return {
            "gpu_count": self.gpu_count,
            "accelerator_type": self.accelerator_type,
            "device_flops_override": self.device_flops_override,
        }


@dataclass(frozen=True)
class TpuPodConfig(ResourceConfig):
    """
    Configuration for TPU-based training.
    """

    tpu_type: str
    """Type of TPU to use, e.g. v4-128."""
    slice_count: int | list[int] = 1
    """Number of TPU slices for training."""

    runtime_env: RuntimeEnv = dataclasses.field(default_factory=lambda: RuntimeEnv())

    device_flops_override: float | None = None
    """Optional override for device FLOPS. If set, this value will be used instead of looking up in device_flops_map."""

    def accelerator_descriptor(self) -> str | None:
        return self.tpu_type

    def as_ray_resources(self) -> RayResources:
        """
        Returns the resource bundle for this TPU configuration. We handle the TPU requirements separately
        so don't need to specify them here.
        """
        return RayResources(runtime_env=self.runtime_env, num_cpus=8)

    def device_flops(self) -> float:
        """Get the peak FLOPs/s for *each* device of the given TPU type."""
        if self.device_flops_override is not None:
            logger.info(
                f"Using user-provided device FLOPS override: "
                f"{self.device_flops_override} for TPU type {self.tpu_type}"
            )
            return self.device_flops_override

        from marin.resources_utils import get_per_device_tpu_flops

        return get_per_device_tpu_flops(self.tpu_type)

    def total_device_count(self) -> int:
        """Get the total number of TPU devices."""
        from marin.resources_utils import get_tpu_type_and_chips

        # Get the number of devices for this TPU configuration
        _, num_devices = get_tpu_type_and_chips(self.tpu_type)

        slice_count = self.slice_count if isinstance(self.slice_count, int) else max(self.slice_count)
        return slice_count * num_devices

    def as_json_dict(self) -> dict:
        """Convert TpuPodConfig to a JSON-serializable dictionary."""
        return {
            "tpu_type": self.tpu_type,
            "slice_count": self.slice_count,
            "device_flops_override": self.device_flops_override,
        }
