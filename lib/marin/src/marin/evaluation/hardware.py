# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure accelerator selection from a model budget and explicit fleet policy."""

import re
from dataclasses import dataclass
from enum import StrEnum

from fray.types import TPU_HBM_BYTES_PER_CHIP, get_tpu_topology, tpu_family

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
class HardwarePolicy:
    """Fleet values consumed by the generic sizing algorithm."""

    utilization: float
    tpu_slices: tuple[str, ...]
    tpu_family_regions: dict[str, str]
    gpu_preference: tuple[str, ...]
    gpu_hbm_gb: dict[str, int]
    gpu_max_count: dict[str, int]
    gpu_clusters: dict[str, str]

    def __post_init__(self) -> None:
        if not 0 < self.utilization <= 1:
            raise ValueError("utilization must be in (0, 1]")
        if not self.tpu_slices or not self.gpu_preference:
            raise ValueError("hardware policy requires TPU and GPU choices")


def default_platform(model: ModelConfig) -> Platform:
    """Choose GPU only when the model requires or pins it."""
    if model.serve.gpu_only or model.serve.fixed_gpu is not None:
        return Platform.GPU
    return Platform.TPU


def _select_tpu(hbm_gb: int, policy: HardwarePolicy) -> AcceleratorChoice:
    for name in policy.tpu_slices:
        topology = get_tpu_topology(name)
        family = tpu_family(name)
        per_chip_gb = TPU_HBM_BYTES_PER_CHIP[family] / _BYTES_PER_GIB
        if topology.chip_count * per_chip_gb * policy.utilization >= hbm_gb:
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
        count = 1
        while count <= policy.gpu_max_count[gpu_type]:
            if policy.gpu_hbm_gb[gpu_type] * count * policy.utilization >= hbm_gb:
                return AcceleratorChoice(
                    platform=Platform.GPU,
                    gpu_type=gpu_type,
                    gpu_count=count,
                    target_cluster=target_cluster or policy.gpu_clusters[gpu_type],
                )
            count *= 2
    raise ValueError(
        f"no GPU slice fits {hbm_gb} GB HBM at {policy.utilization:.0%} utilization"
    )


def _parse_override(override: str, policy: HardwarePolicy) -> AcceleratorChoice:
    text = override.strip()
    match = _GPU_OVERRIDE.match(text)
    if match:
        gpu_type = match["type"].upper()
        if gpu_type not in policy.gpu_hbm_gb:
            raise ValueError(f"unknown GPU type {gpu_type!r} in accelerator override {override!r}")
        return AcceleratorChoice(
            platform=Platform.GPU,
            gpu_type=gpu_type,
            gpu_count=int(match["count"]),
            target_cluster=policy.gpu_clusters[gpu_type],
        )
    get_tpu_topology(text)
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


def select_accelerator(
    model: ModelConfig,
    platform: Platform,
    override: str | None,
    policy: HardwarePolicy,
) -> AcceleratorChoice:
    """Resolve one serving slice under ``policy``."""
    if override:
        return _parse_override(override, policy)
    serve = model.serve
    if serve.fixed_gpu is not None:
        gpu_type, gpu_count = serve.fixed_gpu
        return AcceleratorChoice(
            platform=Platform.GPU,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            target_cluster=serve.target_cluster or policy.gpu_clusters.get(gpu_type),
        )
    if serve.hbm_gb is None:
        raise ValueError(
            f"model {model.name!r} sets neither serve.hbm_gb nor serve.fixed_gpu; cannot size a slice"
        )
    if platform is Platform.GPU:
        return _select_gpu(serve.hbm_gb, serve.target_cluster, policy)
    if serve.gpu_only:
        raise ValueError(f"model {model.name!r} is gpu_only; launch with --platform gpu")
    return _select_tpu(serve.hbm_gb, policy)
