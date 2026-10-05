# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Block-size configuration for the fused short-convolution kernels."""

from dataclasses import dataclass

#: Segment id for positions outside the sequence. The reference pads its shifted segment ids
#: with it and every backend must match, which is what makes the first ``kernel_size - 1``
#: positions of a sequence agree bit for bit.
OOB_SEGMENT = -1


@dataclass(frozen=True, slots=True)
class ShortConvTiles:
    """Launch shape of one direction's GPU Pallas short-conv kernel.

    Attributes:
      s_block_size: rows of one sequence block, which a program walks in order. A power of
        two dividing the sequence length.
      c_block_size: channels per program, a power of two. It is halved until it divides the
        channel count.
      num_warps: warps per program.
      rows_per_step: rows a program loads together before it stores any, a power of two no
        larger than ``s_block_size``.
      num_stages: Triton software-pipelining depth of the row loop.
    """

    s_block_size: int
    c_block_size: int
    num_warps: int
    rows_per_step: int
    num_stages: int = 1


@dataclass(frozen=True, slots=True)
class ShortConvBlockSizes:
    """Launch shapes of the forward and backward GPU Pallas short-conv kernels.

    Defaults are the measured winners of a sweep on one GB200 at the EP64 hero per-layer
    shapes (``[16, 4096, C]`` bf16, kernel 4), in kernel microseconds at C = 6144 / 1536:

    ===========  ===========================  =================
    direction    tile (s, c, warps, rows)     us
    ===========  ===========================  =================
    forward      **32, 512, 4, 16**           **229.6 / 59.7**
    forward      32, 512, 4, 8                233.0 / 61.0
    forward      32, 512, 4, 32               283.1 / 73.4
    backward     **128, 256, 2, 8**           **403.7 / 111.0**
    backward     64, 256, 2, 8                417.8 / 109.0
    backward     128, 256, 2, 4               449.7 / 122.4
    backward     128, 256, 2, 16              460.4 / 127.6
    ===========  ===========================  =================

    More rows per step put more loads in flight until the registers they hold cut occupancy.
    The backward carries twice the forward's rows and four fp32 ``dw`` accumulators, so it
    takes half as many.
    """

    forward: ShortConvTiles = ShortConvTiles(s_block_size=32, c_block_size=512, num_warps=4, rows_per_step=16)
    backward: ShortConvTiles = ShortConvTiles(s_block_size=128, c_block_size=256, num_warps=2, rows_per_step=8)

    @classmethod
    def get_default(cls) -> "ShortConvBlockSizes":
        return cls()
