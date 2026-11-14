# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve the original general stream inside a repair/general outer mixture."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from levanter.data.mixture import MixtureDataset
from levanter.data.text.datasets import LmDataConfig, PrebuiltLmDataset
from levanter.store.cache import TreeCache
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.long_loop.runs.e12.frozen_retention import FrozenRetentionDataset


class AlignedReplacementMixture(MixtureDataset):
    """Replace selected whole general rows; preserve all kept row positions."""

    def _get_block(self, index):
        original = super()._get_block(index)
        counts = {}
        ordered = np.empty_like(original)
        for position, encoded in enumerate(original):
            child = int(encoded) >> 16
            offset = counts.get(child, 0)
            ordered[position] = (child << 16) + (position if child == 0 else offset)
            counts[child] = offset + 1
        return ordered

    def _index_into_dataset_for_id(self, encoded, block_id):
        child = int(encoded) >> 16
        if child == 0:
            return child, block_id * self.block_size + (int(encoded) & 0xFFFF)
        return super()._index_into_dataset_for_id(encoded, block_id)


@dataclass(frozen=True)
class E12DataConfig(LmDataConfig):
    repair_cache: str = ""
    repair_row_weight: float = 0.0
    outer_seed: int = 120012
    outer_block_size: int = 480
    control_only: bool = False
    repair_once: bool = True
    repair_validation_cache: str = ""
    repair_test_cache: str = ""
    retention_manifest: str = ""
    retention_manifest_sha256: str = ""
    retention_root: str = ""

    def train_mixture(self, Pos, batch_schedule, *, key):
        if not self.repair_cache or not 0 < self.repair_row_weight < 1:
            raise ValueError("E12 requires a pinned cache and calibrated target-token weight")
        # Calling super with the original key preserves each kept general row.
        # Arm0 uses the unmodified general stream; selected rows are replaced only in treatments.
        general = super().train_mixture(Pos, batch_schedule, key=key)
        if self.control_only:
            return general
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
        weights = {"general": 1 - self.repair_row_weight, "e12": self.repair_row_weight}
        if self.repair_once:
            weights = [(0, weights), (self.outer_block_size, {"general": 1.0})]
        return AlignedReplacementMixture(
            datasets={"general": general, "e12": repair},
            weights=weights,
            key=self.outer_seed,
            block_size=self.outer_block_size,
            stop_strategy="restart",
        )

    def tagged_eval_sets(self, Pos):
        if self.retention_manifest:
            payload = StoragePath(self.retention_manifest).read_bytes()
            if hashlib.sha256(payload).hexdigest() != self.retention_manifest_sha256:
                raise ValueError("Frozen retention manifest changed")
            records = json.loads(payload)
            # Singleton subsets keep global row order/batching while yielding paired row CE.
            # Original tags still aggregate each source; recompute macro over those tags only.
            result = [
                (
                    FrozenRetentionDataset(self.retention_root, [row]),
                    [*row["tags"], f"retention-row/{ordinal:03d}"],
                )
                for ordinal, row in enumerate(records)
            ]
        else:
            result = list(super().tagged_eval_sets(Pos))
        for label, path in (("validation", self.repair_validation_cache), ("test", self.repair_test_cache)):
            if path:
                cache = TreeCache.load(
                    path, {"input_ids": np.zeros(0, np.int32), "loss_weights": np.zeros(0, np.float32)}
                )
                dataset = PrebuiltLmDataset(
                    cache,
                    Pos,
                    input_ids_key="input_ids",
                    loss_weights_key="loss_weights",
                    loss_weight_transform=None,
                    eos_id=128001,
                    block_cross_document_attention=self.block_cross_document_attention,
                )
                result.append((dataset, ["e12", "e12/" + label]))
        return result
