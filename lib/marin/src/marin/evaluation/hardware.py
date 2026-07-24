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
        """Resolve one serving slice under this fleet policy."""
        if override:
            return _parse_override(override, self)
        serve = model.serve
        if serve.fixed_gpu is not None:
            gpu_type, gpu_count = serve.fixed_gpu
            profile = self.gpu_profiles.get(gpu_type)
            return AcceleratorChoice(
                platform=Platform.GPU,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                target_cluster=serve.target_cluster or (profile.cluster if profile is not None else None),
            )
        if serve.hbm_gb is None:
            raise ValueError(f"model {model.name!r} sets neither serve.hbm_gb nor serve.fixed_gpu; cannot size a slice")
        if platform is Platform.GPU:
            return _select_gpu(serve.hbm_gb, serve.target_cluster, self)
        if serve.gpu_only:
            raise ValueError(f"model {model.name!r} is gpu_only; launch with --platform gpu")
        return _select_tpu(serve.hbm_gb, self)


def default_platform(model: ModelConfig) -> Platform:
    """Choose GPU only when the model requires or pins it."""
    if model.serve.gpu_only or model.serve.fixed_gpu is not None:
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


def _select_gpu(
    hbm_gb: int,
    target_cluster: str | None,
    policy: HardwarePolicy,
) -> AcceleratorChoice:
    for gpu_type in policy.gpu_preference:
        profile = policy.gpu_profiles[gpu_type]
        count = 1
        while count <= profile.max_count:
            if profile.hbm_gb * count * policy.utilization >= hbm_gb:
                return AcceleratorChoice(
                    platform=Platform.GPU,
                    gpu_type=gpu_type,
                    gpu_count=count,
                    target_cluster=target_cluster or profile.cluster,
                )
            count *= 2
    raise ValueError(f"no GPU slice fits {hbm_gb} GB HBM at {policy.utilization:.0%} utilization")


def _parse_override(override: str, policy: HardwarePolicy) -> AcceleratorChoice:
    text = override.strip()
    match = _GPU_OVERRIDE.match(text)
    if match:
        gpu_type = match["type"].upper()
        if gpu_type not in policy.gpu_profiles:
            raise ValueError(f"unknown GPU type {gpu_type!r} in accelerator override {override!r}")
        profile = policy.gpu_profiles[gpu_type]
        return AcceleratorChoice(
            platform=Platform.GPU,
            gpu_type=gpu_type,
            gpu_count=int(match["count"]),
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
