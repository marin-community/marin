# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve the original general stream inside a repair/general outer mixture."""

from dataclasses import dataclass

import numpy as np
from levanter.data.mixture import MixtureDataset
from levanter.data.text.datasets import LmDataConfig, PrebuiltLmDataset
from levanter.store.cache import TreeCache


class OrderedChildMixture(MixtureDataset):
    """Randomize source tags, while preserving each child stream's row order."""

    def _get_block(self, index):
        original = super()._get_block(index)
        counts = {}
        ordered = np.empty_like(original)
        for position, encoded in enumerate(original):
            child = int(encoded) >> 16
            offset = counts.get(child, 0)
            ordered[position] = (child << 16) + offset
            counts[child] = offset + 1
        return ordered


@dataclass(frozen=True)
class E12DataConfig(LmDataConfig):
    repair_cache: str = ""
    repair_row_weight: float = 0.0
    outer_seed: int = 120012
    outer_block_size: int = 800

    def train_mixture(self, Pos, batch_schedule, *, key):
        if not self.repair_cache or not 0 < self.repair_row_weight < 1:
            raise ValueError("E12 requires a pinned cache and calibrated target-token weight")
        # Calling super with the original key preserves the general subsequence.
        general = super().train_mixture(Pos, batch_schedule, key=key)
        cache = TreeCache.load(
            self.repair_cache,
            {"input_ids": np.zeros(0, np.int32), "loss_weights": np.zeros(0, np.float32)},
        )
        repair = PrebuiltLmDataset(
            cache,
            Pos,
            input_ids_key="input_ids",
            loss_weights_key="loss_weights",
            loss_weight_transform=None,
            eos_id=128001,
            block_cross_document_attention=self.block_cross_document_attention,
        )
        return OrderedChildMixture(
            datasets={"general": general, "e12": repair},
            weights={"general": 1 - self.repair_row_weight, "e12": self.repair_row_weight},
            key=self.outer_seed,
            block_size=self.outer_block_size,
            stop_strategy="restart",
        )
