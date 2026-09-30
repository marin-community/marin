# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Block-size configuration for the fused RMSNorm + GatedNorm kernels."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GatedRmsNormBlockSizes:
    """Tile sizes for the two GPU Pallas kernels.

    The statistics kernel streams a ``[t_block_size, stats_d_block_size]`` tile of ``x`` per
    step over the hidden axis. The output kernel writes one ``[t_block_size,
    out_d_block_size]`` tile per program. Every tile size must be a power of two (a Pallas
    Triton lowering constraint) and divide the hidden size.
    """

    t_block_size: int = 128
    stats_d_block_size: int = 32
    out_d_block_size: int = 128
    stats_num_warps: int = 4
    stats_num_stages: int = 4
    out_num_warps: int = 8
    out_num_stages: int = 3
    out_loop: bool = False
    """Run the output kernel as one program per row block that loops over the hidden axis
    (pipelined loads), instead of one program per ``[t, d]`` tile."""

    @classmethod
    def get_default(cls) -> "GatedRmsNormBlockSizes":
        return cls()
