import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import ray
import ray.util.accelerators.accelerators as ray_accel_types
from levanter.utils.py_utils import logical_cpu_core_count
from levanter.utils.ray_utils import RayResources
from ray.remote_function import RemoteFunction
from ray.runtime_env import RuntimeEnv

# ray just declares a bunch of constants, so we read them out via reflection
_ACCEL_TYPES: list[str] = [
    getattr(ray_accel_types, name) for name in dir(ray_accel_types) if name.isupper() and name != "TPU"
]
assert all(isinstance(x, str) for x in _ACCEL_TYPES), "Expected all accelerator types to be strings"

AcceleratorType: TypeAlias = Literal[*_ACCEL_TYPES]
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
        print(new_env)
        return dataclasses.replace(self, runtime_env=RuntimeEnv(**{**self.runtime_env, "env_vars": new_env}))


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
        out = {"num_gpus": self.gpu_count}
        if self.accelerator_type is not None:
            out["accelerator_type"] = self.accelerator_type
        return out

    def as_ray_resources(self) -> RayResources:
        return RayResources(**self.as_remote_kwargs())


@dataclass(frozen=True)
class TpuPodConfig(ResourceConfig):
    """
    Configuration for TPU-based training.
    """

    tpu_type: str
    """Type of TPU to use, e.g. v4-128."""
    node_count: int = 1
    """Number of TPU slices for training."""

    runtime_env: RuntimeEnv = dataclasses.field(default_factory=lambda: RuntimeEnv())

    def accelerator_descriptor(self) -> str | None:
        return self.tpu_type

    def as_ray_resources(self) -> RayResources:
        return RayResources(resources={self.tpu_type: self.node_count})
