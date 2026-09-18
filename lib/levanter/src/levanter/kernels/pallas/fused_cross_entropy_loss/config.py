# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

# Empirical guardrail from Triton shared-memory launch failures: H100's 232,448 bytes of
# per-SM shared memory minus ~131 KB of kernel overhead leaves 101,376 bytes for the
# weight tile. Same limit applies to all NVIDIA GPUs including GB10.
NVIDIA_WEIGHT_TILE_BYTES_LIMIT = 101_376


def max_weight_tile_bytes_for_device(device_kind: str) -> int | None:
    """Weight-tile budget for a lowercased ``device_kind``, or ``None`` if unknown.

    Shared by the batched_xla launch check and block-size inference so an inferred
    configuration is never one the launch check refuses."""
    if "nvidia" in device_kind:
        return NVIDIA_WEIGHT_TILE_BYTES_LIMIT
    return None


@dataclass(frozen=True, slots=True)
class BlockSizes:
    """Block sizes for fused linear softmax cross-entropy kernels.

    Note:
        Pallas TPU kernels require block sizes to be multiples of 128. This is
        validated at runtime when using the Pallas backend.
    """

    b_block_size: int = 1024
    h_block_size: int = 512
    v_block_size: int = 1024

    @classmethod
    def get_default(cls) -> "BlockSizes":
        return cls()
