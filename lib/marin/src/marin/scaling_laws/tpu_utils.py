# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU hardware utilities for memory estimation and slice selection.

This module provides utilities for estimating memory requirements and
selecting appropriate TPU slice sizes for training runs.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TpuSpec:
    """Hardware specification for a TPU generation."""

    prefix: str
    """TPU generation prefix, e.g. "v5p" or "v4"."""

    hbm_per_chip_gib: int
    """High-bandwidth memory per chip in GiB."""

    cores_per_chip: int
    """Number of cores per chip."""

    core_options: tuple[int, ...]
    """Available core configurations (slice sizes), sorted ascending."""


# ---------------- TPU Hardware Specs ----------------

V5P_SPEC = TpuSpec(
    prefix="v5p",
    hbm_per_chip_gib=95,
    cores_per_chip=2,
    core_options=(8, 16, 32, 64, 128, 256, 512, 1024, 2048),
)

V4_SPEC = TpuSpec(
    prefix="v4",
    hbm_per_chip_gib=32,
    cores_per_chip=2,
    core_options=(8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
)

# Legacy constants kept for callers that reference them directly.
HBM_PER_CHIP_GIB = V5P_SPEC.hbm_per_chip_gib
CORES_PER_CHIP = V5P_SPEC.cores_per_chip
V5P_CORE_OPTIONS = list(V5P_SPEC.core_options)
V4_HBM_PER_CHIP_GIB = V4_SPEC.hbm_per_chip_gib
V4_CORES_PER_CHIP = V4_SPEC.cores_per_chip
V4_CORE_OPTIONS = list(V4_SPEC.core_options)


def pick_tpu_type(estimated_memory_bytes: int, spec: TpuSpec) -> str:
    """Select the smallest TPU slice that fits the estimated memory.

    Args:
        estimated_memory_bytes: Estimated memory requirement in bytes.
        spec: Hardware specification for the target TPU generation.

    Returns:
        TPU slice name, e.g., "v5p-8" or "v4-32".

    Raises:
        ValueError: If the model is too large for available slices.
    """
    chip_bytes = spec.hbm_per_chip_gib * 1024**3
    chips = math.ceil(estimated_memory_bytes / chip_bytes)
    cores_req = chips * spec.cores_per_chip

    valid = [c for c in spec.core_options if c >= cores_req]
    if not valid:
        raise ValueError(f"Model too large for available {spec.prefix} slices (need {cores_req} cores).")

    return f"{spec.prefix}-{min(valid)}"


def pick_v5p_type(estimated_memory_bytes: int) -> str:
    """Select the smallest TPU v5p slice that fits the estimated memory."""
    return pick_tpu_type(estimated_memory_bytes, V5P_SPEC)


def pick_v4_type(estimated_memory_bytes: int) -> str:
    """Select the smallest TPU v4 slice that fits the estimated memory."""
    return pick_tpu_type(estimated_memory_bytes, V4_SPEC)
