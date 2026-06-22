# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional


# Empirical launch guardrails from Triton shared-memory launch failures.
# H100 has 232,448 bytes per-SM shared memory; kernel overhead (input tiles,
# accumulators, Triton metadata) consumes ~131 KB, leaving 101,376 bytes for
# the weight tile. The same limit applies to all NVIDIA GPUs including GB10.
NVIDIA_WEIGHT_TILE_BYTES_LIMIT = 101_376


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def max_weight_tile_bytes_for_device(device_kind: str) -> Optional[int]:
    """Return the device weight-tile byte limit, or None when no limit is known."""
    if "nvidia" in device_kind.lower():
        return NVIDIA_WEIGHT_TILE_BYTES_LIMIT
    return None
