# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure accelerator selection from a model budget and explicit fleet policy."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from fray.types import tpu_family, tpu_hbm_capacity_bytes

from marin.evaluation.model_config import ModelConfig

_BYTES_PER_GIB = 1024**3
_GPU_OVERRIDE = re.compile(r"^(?P<type>[A-Za-z0-9]+)x(?P<count>\d+)$")


class Platform(StrEnum):
    TPU = "tpu"
    GPU = "gpu"


@dataclass(frozen=True)
class AcceleratorChoice:
    """Resolved serving slice and placement."""

    platform: Platform
    tpu_type: str | None = None
    gpu_type: str | None = None
    gpu_count: int = 0
    region: str | None = None
    target_cluster: str | None = None

    @property
    def label(self) -> str:
        if self.platform is Platform.TPU:
            assert self.tpu_type is not None
            return self.tpu_type
        return f"{self.gpu_type}x{self.gpu_count}"


@dataclass(frozen=True)
class GpuProfile:
    """Capacity and placement for one GPU type."""

    hbm_gb: int
    max_count: int
    cluster: str

    def __post_init__(self) -> None:
        if self.hbm_gb <= 0 or self.max_count <= 0:
            raise ValueError("GPU capacity and maximum count must be positive")


@dataclass(frozen=True)
class HardwarePolicy:
    """Fleet values consumed by the generic sizing algorithm."""

    utilization: float
    tpu_slices: tuple[str, ...]
    tpu_family_regions: Mapping[str, str]
    gpu_preference: tuple[str, ...]
    gpu_profiles: Mapping[str, GpuProfile]

    def __post_init__(self) -> None:
        if not 0 < self.utilization <= 1:
            raise ValueError("utilization must be in (0, 1]")
        if not self.tpu_slices or not self.gpu_preference:
            raise ValueError("hardware policy requires TPU and GPU choices")
        missing = set(self.gpu_preference) - self.gpu_profiles.keys()
        if missing:
            raise ValueError(f"GPU preferences have no profile: {sorted(missing)}")

    def select(
        self,
        model: ModelConfig,
        platform: Platform,
        override: str | None,
    ) -> AcceleratorChoice:
        """Resolve one serving slice from a model's resource compatibility and a CLI choice."""
        hint = model.resource_hint
        if override:
            choice = _parse_override(override, self)
            if hint.gpu and choice.platform is not Platform.GPU:
                raise ValueError(f"model {model.name!r} requires GPU; accelerator override {override!r} selects TPU")
            _validate_parallelism(model, choice)
            return choice
        if hint.gpu:
            if platform is not Platform.GPU:
                raise ValueError(f"model {model.name!r} requires GPU; launch with --platform gpu")
            choice = _select_required_gpu(model, self)
            _validate_parallelism(model, choice)
            return choice
        if hint.hbm_gb is None:
            raise ValueError(
                f"model {model.name!r} sets neither resource_hint.hbm_gb nor resource_hint.gpu; " "cannot size a slice"
            )
        if platform is Platform.GPU:
            choice = _select_gpu(hint.hbm_gb, self)
        else:
            choice = _select_tpu(hint.hbm_gb, self)
        _validate_parallelism(model, choice)
        return choice


def default_platform(model: ModelConfig) -> Platform:
    """Choose GPU only when the model declares GPU-only compatibility."""
    if model.resource_hint.gpu:
        return Platform.GPU
    return Platform.TPU


def _select_tpu(hbm_gb: int, policy: HardwarePolicy) -> AcceleratorChoice:
    for name in policy.tpu_slices:
        family = tpu_family(name)
        capacity_gb = tpu_hbm_capacity_bytes(name) / _BYTES_PER_GIB
        if capacity_gb * policy.utilization >= hbm_gb:
            return AcceleratorChoice(
                platform=Platform.TPU,
                tpu_type=name,
                region=policy.tpu_family_regions[family],
            )
    raise ValueError(
        f"no provisioned single-host TPU slice fits {hbm_gb} GB HBM at "
        f"{policy.utilization:.0%} utilization; use --platform gpu"
    )


def _select_gpu(hbm_gb: int, policy: HardwarePolicy) -> AcceleratorChoice:
    for gpu_type in policy.gpu_preference:
        profile = policy.gpu_profiles[gpu_type]
        count = 1
        while count <= profile.max_count:
            if profile.hbm_gb * count * policy.utilization >= hbm_gb:
                return AcceleratorChoice(
                    platform=Platform.GPU,
                    gpu_type=gpu_type,
                    gpu_count=count,
                    target_cluster=profile.cluster,
                )
            count *= 2
    raise ValueError(f"no GPU slice fits {hbm_gb} GB HBM at {policy.utilization:.0%} utilization")


def _select_required_gpu(model: ModelConfig, policy: HardwarePolicy) -> AcceleratorChoice:
    hint = model.resource_hint
    unknown = set(hint.gpu) - policy.gpu_profiles.keys()
    if unknown:
        raise ValueError(f"model {model.name!r} requires GPU types absent from this fleet: {sorted(unknown)}")
    for gpu_type in policy.gpu_preference:
        count = hint.gpu.get(gpu_type)
        if count is None:
            continue
        profile = policy.gpu_profiles[gpu_type]
        if count > profile.max_count:
            raise ValueError(
                f"model {model.name!r} requires {gpu_type}x{count}, but this fleet supports at most "
                f"{gpu_type}x{profile.max_count}"
            )
        return AcceleratorChoice(
            platform=Platform.GPU,
            gpu_type=gpu_type,
            gpu_count=count,
            target_cluster=profile.cluster,
        )
    raise ValueError(f"model {model.name!r} has no GPU resource hint compatible with this fleet")


def _validate_parallelism(model: ModelConfig, choice: AcceleratorChoice) -> None:
    serve = model.serve
    if choice.platform is not Platform.GPU or serve.tensor_parallel_size is None:
        return
    ranks = serve.tensor_parallel_size * (serve.data_parallel_size or 1)
    if ranks != choice.gpu_count:
        raise ValueError(
            f"model {model.name!r} configures tensor_parallel_size={serve.tensor_parallel_size} and "
            f"data_parallel_size={serve.data_parallel_size or 1}, requiring {ranks} GPUs, but selected "
            f"{choice.label}"
        )


def _parse_override(override: str, policy: HardwarePolicy) -> AcceleratorChoice:
    text = override.strip()
    match = _GPU_OVERRIDE.match(text)
    if match:
        gpu_type = match["type"].upper()
        if gpu_type not in policy.gpu_profiles:
            raise ValueError(f"unknown GPU type {gpu_type!r} in accelerator override {override!r}")
        profile = policy.gpu_profiles[gpu_type]
        count = int(match["count"])
        if count <= 0 or count & (count - 1) or count > profile.max_count:
            raise ValueError(
                f"GPU count in accelerator override {override!r} must be a positive power of two no larger "
                f"than {profile.max_count}"
            )
        return AcceleratorChoice(
            platform=Platform.GPU,
            gpu_type=gpu_type,
            gpu_count=count,
            target_cluster=profile.cluster,
        )
    tpu_hbm_capacity_bytes(text)
    if text not in policy.tpu_slices:
        raise ValueError(
            f"accelerator override {text!r} is not a servable single-host TPU; choose one of "
            f"{', '.join(policy.tpu_slices)}"
        )
    return AcceleratorChoice(
        platform=Platform.TPU,
        tpu_type=text,
        region=policy.tpu_family_regions[tpu_family(text)],
    )
