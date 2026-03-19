# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BlockSizes:
    """Block sizes reserved for a future TPU Pallas SSD local-block kernel."""

    group_block_size: int = 1
    query_block_size: int | None = None
    key_block_size: int | None = None
    value_block_size: int | None = None

    @classmethod
    def get_default(cls) -> "BlockSizes":
        return cls()
