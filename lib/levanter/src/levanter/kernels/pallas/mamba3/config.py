# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from ..ssd.config import BlockSizes


Mamba3Mode: TypeAlias = Literal["siso", "mimo"]

MAMBA3_TPU_DEFAULT_SISO_CHUNK_SIZE = 512
MAMBA3_TPU_DEFAULT_MIMO_R4_CHUNK_SIZE = 256


def mamba3_tpu_default_chunk_size(mode: Mamba3Mode, *, mimo_rank: int | None = None) -> int:
    """Return the current tuned TPU chunk-size default for a stable Mamba-3 mode."""

    if mode == "siso":
        return MAMBA3_TPU_DEFAULT_SISO_CHUNK_SIZE
    if mode == "mimo":
        if mimo_rank is None:
            mimo_rank = 4
        if mimo_rank != 4:
            raise ValueError(f"Only `mimo_rank=4` has a tuned TPU default chunk size, got {mimo_rank}.")
        return MAMBA3_TPU_DEFAULT_MIMO_R4_CHUNK_SIZE
    raise ValueError(f"Unsupported Mamba-3 mode: {mode}.")


@dataclass(frozen=True, slots=True)
class HybridModeConfig:
    """Stable mode/config surface for selecting the current SISO or MIMO TPU path."""

    mode: Mamba3Mode = "siso"
    mimo_rank: int | None = None
    chunk_size: int | None = None

    def resolved_chunk_size(self) -> int:
        """Return the user override or the tuned TPU chunk-size default for the chosen mode."""

        if self.chunk_size is not None:
            return self.chunk_size
        return mamba3_tpu_default_chunk_size(self.mode, mimo_rank=self.mimo_rank)


__all__ = [
    "BlockSizes",
    "HybridModeConfig",
    "MAMBA3_TPU_DEFAULT_MIMO_R4_CHUNK_SIZE",
    "MAMBA3_TPU_DEFAULT_SISO_CHUNK_SIZE",
    "Mamba3Mode",
    "mamba3_tpu_default_chunk_size",
]
