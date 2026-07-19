# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn a model's HBM budget into a serving slice.

:func:`select_accelerator` picks the smallest slice that fits ``model.hbm_gb`` at 85% utilization,
preferring TPU families v5e -> v6e -> v5p (single-host slices only) and GPU types H100 -> GB200. A
model may pin an exact GPU shape (``fixed_gpu``); a caller-supplied ``override`` (``v6e-8`` or
``H100x8``) wins over everything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from fray.types import TPU_HBM_BYTES_PER_CHIP, get_tpu_topology, tpu_family

from experiments.evaluation.models import EvalModelConfig

_BYTES_PER_GIB = 1024**3
UTILIZATION = 0.85

# Single-host TPU slices the marin cluster actually provisions (lib/iris/config/marin.yaml
# tpu_pools), in preference order: v5e -> v6e -> v5p, smallest fitting slice first. The serve
# path runs on one VM, so multi-host slices are out regardless of pool availability.
SERVABLE_TPU_SLICES = ("v5litepod-4", "v5litepod-8", "v6e-4", "v6e-8", "v5p-8")

# The region whose zones carry each family's pool (marin.yaml tpu_pools zones). Children of an
# eval orchestrator inherit its region unless pinned, and the orchestrator can land anywhere, so
# every TPU choice pins a region where its slice actually exists.
TPU_FAMILY_REGION: dict[str, str] = {"v5e": "us-west4", "v6e": "europe-west4", "v5p": "us-central1"}

# GPU types in preference order, with per-GPU HBM, the largest single-node count, and the CoreWeave
# peer that hosts them.
GPU_PREFERENCE = ("H100", "GB200")
GPU_HBM_GB: dict[str, int] = {"H100": 80, "GB200": 186}
GPU_MAX_COUNT: dict[str, int] = {"H100": 8, "GB200": 4}
GPU_CLUSTERS: dict[str, str] = {"H100": "cw-us-east-02a", "GB200": "cw-us-east-08a"}

_GPU_OVERRIDE = re.compile(r"^(?P<type>[A-Za-z0-9]+)x(?P<count>\d+)$")


class Platform(StrEnum):
    TPU = "tpu"
    GPU = "gpu"


@dataclass(frozen=True)
class AcceleratorChoice:
    """The resolved serving slice: exactly one of ``tpu_type`` or (``gpu_type``, ``gpu_count``) is set.

    ``region`` optionally pins workers within a cluster; ``target_cluster`` routes a GPU job to a
    CoreWeave federation peer.
    """

    platform: Platform
    tpu_type: str | None = None
    gpu_type: str | None = None
    gpu_count: int = 0
    region: str | None = None
    target_cluster: str | None = None

    @property
    def label(self) -> str:
        """A compact human-readable accelerator name, e.g. ``v6e-8`` or ``H100x8``."""
        if self.platform == Platform.TPU:
            assert self.tpu_type is not None
            return self.tpu_type
        return f"{self.gpu_type}x{self.gpu_count}"


def default_platform(model: EvalModelConfig) -> Platform:
    """The platform a model runs on absent an explicit choice: GPU when it is GPU-only or GPU-pinned."""
    if model.gpu_only or model.fixed_gpu is not None:
        return Platform.GPU
    return Platform.TPU


def _tpu_per_chip_gb(family: str) -> float:
    return TPU_HBM_BYTES_PER_CHIP[family] / _BYTES_PER_GIB


def _select_tpu(hbm_gb: int) -> AcceleratorChoice:
    for name in SERVABLE_TPU_SLICES:
        topo = get_tpu_topology(name)
        family = tpu_family(name)
        if topo.chip_count * _tpu_per_chip_gb(family) * UTILIZATION >= hbm_gb:
            return AcceleratorChoice(platform=Platform.TPU, tpu_type=name, region=TPU_FAMILY_REGION[family])
    raise ValueError(
        f"no provisioned single-host TPU slice fits {hbm_gb} GB HBM at {UTILIZATION:.0%} utilization; "
        "use --platform gpu"
    )


def _select_gpu(hbm_gb: int, target_cluster: str | None) -> AcceleratorChoice:
    for gpu_type in GPU_PREFERENCE:
        per_gpu_gb = GPU_HBM_GB[gpu_type]
        count = 1
        while count <= GPU_MAX_COUNT[gpu_type]:
            if per_gpu_gb * count * UTILIZATION >= hbm_gb:
                return AcceleratorChoice(
                    platform=Platform.GPU,
                    gpu_type=gpu_type,
                    gpu_count=count,
                    target_cluster=target_cluster or GPU_CLUSTERS[gpu_type],
                )
            count *= 2
    raise ValueError(f"no GPU slice fits {hbm_gb} GB HBM at {UTILIZATION:.0%} utilization (max GB200x4)")


def _fixed_gpu_choice(model: EvalModelConfig) -> AcceleratorChoice:
    gpu_type, gpu_count = model.fixed_gpu
    return AcceleratorChoice(
        platform=Platform.GPU,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        target_cluster=model.target_cluster or GPU_CLUSTERS.get(gpu_type),
    )


def _parse_override(override: str) -> AcceleratorChoice:
    text = override.strip()
    match = _GPU_OVERRIDE.match(text)
    if match:
        gpu_type = match["type"].upper()
        if gpu_type not in GPU_HBM_GB:
            raise ValueError(f"unknown GPU type {gpu_type!r} in accelerator override {override!r}")
        return AcceleratorChoice(
            platform=Platform.GPU,
            gpu_type=gpu_type,
            gpu_count=int(match["count"]),
            target_cluster=GPU_CLUSTERS[gpu_type],
        )
    # A non-GPU override must be a valid TPU slice name (raises ValueError otherwise).
    get_tpu_topology(text)
    if text not in SERVABLE_TPU_SLICES:
        raise ValueError(
            f"accelerator override {text!r} is a multi-host TPU slice; iris runs one task per VM for a "
            "multi-host TPU job, so serving would start independent servers fighting over one endpoint "
            f"name. Use a single-host slice: {', '.join(SERVABLE_TPU_SLICES)}"
        )
    return AcceleratorChoice(platform=Platform.TPU, tpu_type=text, region=TPU_FAMILY_REGION[tpu_family(text)])


def select_accelerator(model: EvalModelConfig, platform: Platform, override: str | None) -> AcceleratorChoice:
    """Resolve the serving slice for ``model`` on ``platform``.

    An explicit ``override`` (``v6e-8`` or ``H100x8``) wins over everything; otherwise a model's
    ``fixed_gpu`` pin wins over the platform heuristic. TPU is rejected for a GPU-only model.
    """
    if override:
        return _parse_override(override)
    if model.fixed_gpu is not None:
        return _fixed_gpu_choice(model)
    if platform == Platform.GPU:
        return _select_gpu(model.hbm_gb, model.target_cluster)
    if model.gpu_only:
        raise ValueError(f"model {model.name!r} is gpu_only; launch with --platform gpu")
    return _select_tpu(model.hbm_gb)
