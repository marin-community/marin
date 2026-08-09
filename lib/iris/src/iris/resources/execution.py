# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native execution records used by public Job specifications."""

import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import cloudpickle
import humanfriendly
from rigging.provenance import LAUNCH_PROVENANCE_ENV, launch_provenance
from rigging.telemetry.probes.nccl_client import NCCL_RAS_ENABLE_ENV

from iris.cluster.setup_scripts import (
    cuda_toolchain_setup_script,
    default_setup_script,
    iris_runtime_setup_script,
    wants_gpu_extra,
)
from iris.cluster.tpu_topology import get_tpu_topology

IRIS_SLICE_COUNT = "IRIS_SLICE_COUNT"
IRIS_TASKS_PER_SLICE = "IRIS_TASKS_PER_SLICE"


@dataclass(frozen=True, slots=True)
class CpuDevice:
    variant: str = ""


@dataclass(frozen=True, slots=True)
class GpuDevice:
    variant: str
    count: int = 1

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"GPU count must be a positive integer, got {self.count}")


@dataclass(frozen=True, slots=True)
class TpuDevice:
    variant: str
    topology: str = ""
    count: int = 0


type Device = CpuDevice | GpuDevice | TpuDevice


def parse_memory_string(value: str) -> int:
    """Parse a human-readable byte quantity."""
    candidate = value.strip()
    if not candidate or candidate == "0":
        return 0
    try:
        return humanfriendly.parse_size(candidate, binary=True)
    except humanfriendly.InvalidSize as exc:
        raise ValueError(str(exc)) from exc


@dataclass(frozen=True, slots=True, init=False)
class ResourceSpec:
    """Exact normalized resources requested by a Job."""

    cpu: float
    memory: int
    disk: int
    device: Device | None

    MIN_ACCELERATOR_CPU_MILLICORES = 4_000

    def __init__(
        self,
        cpu: float = 0.0,
        memory: int | str = 0,
        disk: int | str = 0,
        device: Device | None = None,
    ) -> None:
        object.__setattr__(self, "cpu", int(cpu * 1_000) / 1_000)
        object.__setattr__(self, "memory", memory if isinstance(memory, int) else parse_memory_string(memory))
        object.__setattr__(self, "disk", disk if isinstance(disk, int) else parse_memory_string(disk))
        object.__setattr__(self, "device", device)

    @property
    def cpu_millicores(self) -> int:
        return int(self.cpu * 1_000)


def with_accelerator_cpu_default(resources: ResourceSpec) -> ResourceSpec:
    """Apply the client-side accelerator CPU floor exactly once at submission."""
    if resources.device is None or resources.cpu_millicores >= resources.MIN_ACCELERATOR_CPU_MILLICORES:
        return resources
    return replace(resources, cpu=resources.MIN_ACCELERATOR_CPU_MILLICORES / 1_000)


def tpu_device(variant: str, count: int | None = None) -> TpuDevice:
    """Create a TPU device, inferring its per-VM chip count when known."""
    chip_count = count
    if chip_count is None:
        try:
            chip_count = get_tpu_topology(variant).chips_per_vm
        except ValueError:
            chip_count = 0
    return TpuDevice(variant=variant, count=chip_count)


def gpu_device(variant: str, count: int = 1) -> GpuDevice:
    """Create a GPU device."""
    return GpuDevice(variant=variant, count=count)


def get_gpu_count(device: Device) -> int:
    return device.count if isinstance(device, GpuDevice) else 0


def get_tpu_count(device: Device) -> int:
    return device.count if isinstance(device, TpuDevice) else 0


def adjust_tpu_replicas(device: Device | None, replicas: int) -> int:
    """Expand a single requested replica to the TPU topology's VM count."""
    if not isinstance(device, TpuDevice) or not device.variant:
        return replicas
    try:
        topology = get_tpu_topology(device.variant)
    except ValueError:
        return replicas
    if topology.vm_count <= 1:
        return replicas
    if replicas == 1:
        return topology.vm_count
    if replicas % topology.vm_count != 0:
        raise ValueError(
            f"TPU type '{device.variant}' requires {topology.vm_count} VMs per slice, "
            f"so replicas must be a multiple of {topology.vm_count} (got replicas={replicas}). "
            f"For a single slice, use replicas={topology.vm_count}. "
            f"For N slices, use replicas=N*{topology.vm_count}."
        )
    return replicas


@dataclass(frozen=True, slots=True)
class CommandEntrypoint:
    argv: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "argv", tuple(self.argv))


def _immutable_mapping[K, V](value: Mapping[K, V]) -> Mapping[K, V]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class RuntimeEntrypoint:
    setup_commands: tuple[str, ...]
    run_command: CommandEntrypoint
    workdir_files: Mapping[str, bytes]
    workdir_file_refs: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "setup_commands", tuple(self.setup_commands))
        object.__setattr__(self, "workdir_files", _immutable_mapping(self.workdir_files))
        object.__setattr__(self, "workdir_file_refs", _immutable_mapping(self.workdir_file_refs))


@dataclass(frozen=True, slots=True)
class Environment:
    env_vars: Mapping[str, str]
    setup_scripts: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "env_vars", _immutable_mapping(self.env_vars))
        object.__setattr__(self, "setup_scripts", tuple(self.setup_scripts))


CALLABLE_RUNNER = """\
import cloudpickle
import os
import sys
import traceback
import logging

_LEVEL_PREFIX = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

class _LevelPrefixFormatter(logging.Formatter):
    def format(self, record):
        record.levelprefix = _LEVEL_PREFIX.get(record.levelname, "?")
        return super().format(record)

_root = logging.getLogger()
_root.handlers.clear()
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(_LevelPrefixFormatter(
    fmt="%(levelprefix)s%(asctime)s %(name)s %(message)s",
    datefmt="%Y%m%d %H:%M:%S",
))
_root.addHandler(_handler)
_root.setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("aiobotocore").setLevel(logging.WARNING)

workdir = os.environ["IRIS_WORKDIR"]

try:
    with open(os.path.join(workdir, "_callable.pkl"), "rb") as f:
        fn, args, kwargs = cloudpickle.loads(f.read())
    fn(*args, **kwargs)
except Exception:
    traceback.print_exc()
    sys.exit(1)
"""


@dataclass(frozen=True, slots=True)
class Entrypoint:
    """User-facing command and files before runtime setup is resolved."""

    command: tuple[str, ...]
    workdir_files: Mapping[str, bytes] = MappingProxyType({})
    workdir_file_refs: Mapping[str, str] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("Command must have at least one argument")
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(self, "workdir_files", _immutable_mapping(self.workdir_files))
        object.__setattr__(self, "workdir_file_refs", _immutable_mapping(self.workdir_file_refs))

    def resolve(self) -> tuple[Callable[..., Any], tuple, dict[str, Any]]:
        payload = self.workdir_files.get("_callable.pkl")
        if payload is None:
            raise ValueError("Not a callable entrypoint")
        return cloudpickle.loads(payload)

    @classmethod
    def from_callable(cls, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> "Entrypoint":
        module = sys.modules.get(fn.__module__)
        module_path = Path(module.__file__).parts if module and getattr(module, "__file__", None) else ()
        if module and (module.__package__ is None or module.__spec__ is None or "tests" in module_path):
            cloudpickle.register_pickle_by_value(module)
        return cls(
            command=("bash", "-c", "exec $IRIS_PYTHON -u $IRIS_WORKDIR/_callable_runner.py"),
            workdir_files={
                "_callable.pkl": cloudpickle.dumps((fn, args, kwargs)),
                "_callable_runner.py": CALLABLE_RUNNER.encode(),
            },
        )

    @classmethod
    def from_command(cls, *argv: str) -> "Entrypoint":
        return cls(command=tuple(argv))


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    """User-facing environment builder resolved before Job submission."""

    pip_packages: tuple[str, ...] | None = None
    env_vars: Mapping[str, str] | None = None
    extras: tuple[str, ...] | None = None
    setup_scripts: tuple[str, ...] | None = None
    sync_packages: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pip_packages", tuple(self.pip_packages) if self.pip_packages is not None else None)
        object.__setattr__(self, "env_vars", _immutable_mapping(self.env_vars or {}))
        object.__setattr__(self, "extras", tuple(self.extras) if self.extras is not None else None)
        object.__setattr__(self, "setup_scripts", tuple(self.setup_scripts) if self.setup_scripts is not None else None)
        object.__setattr__(self, "sync_packages", tuple(self.sync_packages) if self.sync_packages is not None else None)

    def resolve(self) -> Environment:
        default_env_vars = {
            "HF_DATASETS_TRUST_REMOTE_CODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_TOKEN": os.getenv("HF_TOKEN"),
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
            LAUNCH_PROVENANCE_ENV: launch_provenance().to_json(),
        }
        if wants_gpu_extra(self.extras or ()):
            default_env_vars.update(
                {
                    NCCL_RAS_ENABLE_ENV: "1",
                    "NCCL_DEBUG": "INFO",
                    "NCCL_DEBUG_SUBSYS": "INIT,BOOTSTRAP,ENV,NET,GRAPH,TUNING,RAS",
                    "NCCL_DEBUG_TIMESTAMP": "[%F %T.%3f]",
                }
            )
        env_vars = {
            key: value for key, value in {**default_env_vars, **(self.env_vars or {})}.items() if value is not None
        }
        if self.setup_scripts is None:
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            extras = list(self.extras or ())
            scripts = [
                default_setup_script(
                    extras=extras,
                    pip_packages=list(self.pip_packages or ()),
                    python_version=python_version,
                    packages=list(self.sync_packages or ()) or None,
                )
            ]
            if wants_gpu_extra(extras):
                scripts.append(cuda_toolchain_setup_script())
        else:
            scripts = [script for script in self.setup_scripts if script.strip()]
        return Environment(env_vars=env_vars, setup_scripts=tuple(scripts))


def build_runtime_entrypoint(entrypoint: Entrypoint, environment: Environment) -> RuntimeEntrypoint:
    """Resolve setup and a user command into the exact runtime entrypoint."""
    user_scripts = tuple(script for script in environment.setup_scripts if script.strip())
    setup_commands = (*user_scripts, iris_runtime_setup_script()) if user_scripts else ()
    return RuntimeEntrypoint(
        setup_commands=setup_commands,
        run_command=CommandEntrypoint(entrypoint.command),
        workdir_files=entrypoint.workdir_files,
        workdir_file_refs=entrypoint.workdir_file_refs,
    )


def slice_topology_environment(resources: ResourceSpec, num_tasks: int) -> Mapping[str, str]:
    """Return TPU multi-slice environment variables for one Job."""
    if num_tasks <= 1 or not isinstance(resources.device, TpuDevice) or not resources.device.variant:
        return MappingProxyType({})
    try:
        tasks_per_slice = get_tpu_topology(resources.device.variant).vm_count
    except ValueError:
        return MappingProxyType({})
    if tasks_per_slice <= 0:
        return MappingProxyType({})
    if num_tasks % tasks_per_slice != 0:
        raise ValueError(
            f"TPU task count ({num_tasks}) must be divisible by TPU VM count ({tasks_per_slice}) "
            f"for variant {resources.device.variant!r}"
        )
    num_slices = num_tasks // tasks_per_slice
    if num_slices <= 1:
        return MappingProxyType({})
    return MappingProxyType({IRIS_SLICE_COUNT: str(num_slices), IRIS_TASKS_PER_SLICE: str(tasks_per_slice)})


def with_slice_topology_environment(
    environment: Environment,
    resources: ResourceSpec,
    num_tasks: int,
) -> Environment:
    """Return an Environment with canonical TPU multi-slice variables."""
    env_vars = dict(environment.env_vars)
    env_vars.pop(IRIS_SLICE_COUNT, None)
    env_vars.pop(IRIS_TASKS_PER_SLICE, None)
    env_vars.update(slice_topology_environment(resources, num_tasks))
    return replace(environment, env_vars=env_vars)
