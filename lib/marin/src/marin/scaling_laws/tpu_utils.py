# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU hardware utilities for memory estimation and slice selection.

This module provides utilities for estimating memory requirements and
selecting appropriate TPU slice sizes for training runs.
"""

import math

# ---------------- TPU v5p Hardware Constants ----------------
# These constants are specific to TPU v5p pods.

HBM_PER_CHIP_GIB = 95
"""High-bandwidth memory per TPU v5p chip in GiB."""

CORES_PER_CHIP = 2
"""Number of cores per TPU v5p chip."""

V5P_CORE_OPTIONS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
"""Available TPU v5p core configurations (slice sizes)."""


def pick_v5p_type(estimated_memory_bytes: int) -> str:
    """Select the smallest TPU v5p slice that fits the estimated memory.

    Args:
        estimated_memory_bytes: Estimated memory requirement in bytes.

    Returns:
        TPU slice name, e.g., "v5p-8" or "v5p-32".

    Raises:
        ValueError: If the model is too large for available v5p slices.
    """
    chip_bytes = HBM_PER_CHIP_GIB * 1024**3
    chips = math.ceil(estimated_memory_bytes / chip_bytes)
    cores_req = chips * CORES_PER_CHIP

    valid = [c for c in V5P_CORE_OPTIONS if c >= cores_req]
    if not valid:
        raise ValueError(f"Model too large for available v5p slices (need {cores_req} cores).")

    return f"v5p-{min(valid)}"
