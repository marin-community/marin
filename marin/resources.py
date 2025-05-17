import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import ray
import ray.util.accelerators.accelerators as ray_accel_types
from levanter.utils.py_utils import logical_cpu_core_count
from levanter.utils.ray_utils import RayResources
from levanter.utils.flop_utils import DEVICE_AVAILABLE_FLOPS
from ray.remote_function import RemoteFunction
from ray.runtime_env import RuntimeEnv

# ray just declares a bunch of constants, so we read them out via reflection
_ACCEL_TYPES: list[str] = [
    getattr(ray_accel_types, name) for name in dir(ray_accel_types) if name.isupper() and name != "TPU"
]
assert all(isinstance(x, str) for x in _ACCEL_TYPES), "Expected all accelerator types to be strings"

AcceleratorType: TypeAlias = Literal[tuple(_ACCEL_TYPES)]
"""
https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html
"""


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

    def device_flops(self) -> float:
        """Returns the peak FLOPs/s for a single device in this configuration."""
        return 0.0

    def total_device_count(self) -> int:
        """Returns the total number of devices in this configuration."""
        return 1


@dataclass(frozen=True)
class CpuOnlyConfig(ResourceConfig):
    num_cpus: int = dataclasses.field(default_factory=lambda: logical_cpu_core_count())
    """Configuration for local training without specialized hardware."""
    runtime_env: RuntimeEnv = dataclasses.field(default_factory=lambda: RuntimeEnv(env_vars={"JAX_PLATFORMS": "cpu"}))

    def accelerator_descriptor(self) -> str | None:
        return None

    def as_remote_kwargs(self) -> dict:
        return dict(num_cpus=self.num_cpus, runtime_env=self.runtime_env)

    def as_ray_resources(self) -> RayResources:
        return RayResources(**self.as_remote_kwargs())

    def total_device_count(self) -> int:
        return self.num_cpus


@dataclass(frozen=True)
class GpuConfig(ResourceConfig):
    """Configuration for GPU-based training."""

    gpu_count: int = 1
    """Number of GPUs to use for training."""

    runtime_env: RuntimeEnv = dataclasses.field(default_factory=RuntimeEnv)

    accelerator_type: AcceleratorType | None = None
    """Type of GPU accelerator to use. If None, will use any available GPU."""

    def accelerator_descriptor(self) -> str | None:
        return self.accelerator_type

    # NB that Ray doesn't like resources={"GPU": 1} so we have to do this
    def as_remote_kwargs(self) -> dict:
        out = dict(num_gpus=self.gpu_count, runtime_env=self.runtime_env)
        if self.accelerator_type is not None:
            out["accelerator_type"] = self.accelerator_type

        return out

    def as_ray_resources(self) -> RayResources:
        return RayResources(**self.as_remote_kwargs())

    def device_flops(self) -> float:
        """Get the peak FLOPs/s for the GPU type."""
        device_type = self.accelerator_type or "a100"
        # Try bf16 first, then fp16
        flops = DEVICE_AVAILABLE_FLOPS.get(device_type, {}).get("bf16")
        if flops is None:
            flops = DEVICE_AVAILABLE_FLOPS.get(device_type, {}).get("fp16", 0)
        return flops

    def total_device_count(self) -> int:
        return self.gpu_count


@dataclass(frozen=True)
class TpuPodConfig(ResourceConfig):
    """
    Configuration for TPU-based training.
    """

    tpu_type: str
    """Type of TPU to use, e.g. v4-128."""
    slice_count: int = 1
    """Number of TPU slices for training."""

    runtime_env: RuntimeEnv = dataclasses.field(default_factory=lambda: RuntimeEnv())

    def accelerator_descriptor(self) -> str | None:
        return self.tpu_type

    def as_ray_resources(self) -> RayResources:
        return RayResources(runtime_env=self.runtime_env, num_cpus=8)

    def device_flops(self) -> float:
        """Get the peak FLOPs/s for the TPU type."""
        if "v4" in self.tpu_type:
            return DEVICE_AVAILABLE_FLOPS["tpu v4"]["bf16"]
        elif "v5" in self.tpu_type:
            return DEVICE_AVAILABLE_FLOPS["tpu v5"]["bf16"]
        elif "v6" in self.tpu_type:
            return DEVICE_AVAILABLE_FLOPS["tpu v6 lite"]["bf16"]
        return 0.0

    def total_device_count(self) -> int:
        """Get the total number of TPU chips."""
        if "v4-128" in self.tpu_type:
            return self.slice_count * 64  # v4 has 64 chips per slice
        elif "v5" in self.tpu_type:
            return self.slice_count * 64  # v5 has 64 chips per slice
        elif "v6" in self.tpu_type:
            return self.slice_count * 64  # v6 has 64 chips per slice
        return self.slice_count  # fallback
