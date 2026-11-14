# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consume the exact, hashed rows produced by the region-local retention freeze."""

import asyncio
import hashlib
import io
from pathlib import PurePosixPath

import numpy as np
from levanter.data.dataset import AsyncDataset
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from rigging.filesystem.storage_path import StoragePath


class FrozenRetentionDataset(AsyncDataset):
    def __init__(self, root, records):
        self.root = root
        self.records = sorted(records, key=lambda row: row["index"])

    async def async_len(self):
        return len(self.records)

    def is_finite(self):
        return True

    def _load(self, index):
        row = self.records[index]
        path = PurePosixPath(row["file"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe retention path")
        payload = (StoragePath(self.root) / row["file"]).read_bytes()
        if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError("Retention row checksum changed")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            fields = {key: archive[key] for key in archive.files}
        for key, expected in row["field_hashes"].items():
            if hashlib.sha256(fields[key].tobytes()).hexdigest() != expected:
                raise ValueError(f"Retention field changed: {key}")
        segment_ids = None
        if row["has_segment_ids"]:
            segment_ids = (fields["q_segment_ids"], fields["kv_segment_ids"])
        return GrugLmExample(
            tokens=fields["input_ids"],
            loss_weight=fields["loss_weights"],
            attn_mask=AttentionMask(
                is_causal=row["is_causal"],
                segment_ids=segment_ids,
                sliding_window=row["sliding_window"],
            ),
        )

    async def get_batch(self, indices):
        return await asyncio.gather(*(asyncio.to_thread(self._load, index) for index in indices))
