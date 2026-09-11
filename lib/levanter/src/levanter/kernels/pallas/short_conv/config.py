# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Block-size configuration for the fused short-convolution kernels."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShortConvBlockSizes:
    """Tile sizes for the GPU Pallas short-conv kernels.

    ``s_block_size`` tiles the sequence axis and ``c_block_size`` the channel axis. Both
    must be powers of two -- a Pallas Triton lowering constraint on tile shapes, not a
    preference.

    Defaults are the measured winner of a six-point sweep on one GB200 at the EP64 hero
    per-layer shapes (65,536 tokens x {1536, 6144} channels, bf16, kernel 4):

    ==========================  ==================  =================
    tile (s, c, warps)          hero-scaled ms/step  vs reference
    ==========================  ==================  =================
    reference (pad-and-shift)               427.2   --
    **64, 64, 4**                           **231.8**  **-195.4 ms**
    32, 64, 4                               234.1   -193.1 ms
    64, 128, 8                              237.8   -189.4 ms
    32, 128, 4                              244.6   -182.5 ms
    128, 64, 4                              244.9   -182.3 ms
    64, 128, 4                              284.1   -143.0 ms
    ==========================  ==================  =================

    The gradient is shallow across the good configs and steeply bad above ``c_block_size``
    128 with large ``s_block_size`` (256x256 was 19x slower than the reference): each
    program holds up to ``kernel_size`` live ``[s, c]`` tiles plus an fp32 accumulator, so
    big tiles spill and occupancy collapses. Small tiles win because the kernel is
    bandwidth-bound and wants occupancy, not reuse.
    """

    s_block_size: int = 64
    c_block_size: int = 64
    num_warps: int = 4
    num_stages: int = 2

    @classmethod
    def get_default(cls) -> "ShortConvBlockSizes":
        return cls()

    def as_key(self) -> str:
        return f"s{self.s_block_size}_c{self.c_block_size}_w{self.num_warps}_st{self.num_stages}"
