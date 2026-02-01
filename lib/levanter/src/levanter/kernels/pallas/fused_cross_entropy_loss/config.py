# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class BlockSizes:
    """Block sizes for fused linear softmax cross-entropy kernels.

    Note:
        Pallas TPU kernels require block sizes to be multiples of 128. This is
        validated at runtime when using the Pallas backend.
    """

    b_block_size: int = 1024
    h_block_size: int = 256
    v_block_size: int = 2048
    bwd_strategy: Literal["combined", "split"] = "combined"

    @classmethod
    def get_default(cls) -> "BlockSizes":
        return cls()
