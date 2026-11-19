# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve the original general stream inside a repair/general outer mixture."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from haliax import Axis
from levanter.data.dataset import AsyncDataset, MappedAsyncDataset
from levanter.data.mixture import MixtureDataset
from levanter.data.text.datasets import LmDataConfig, PrebuiltLmDataset
from levanter.data.text.examples import GrugLmExample
from levanter.store.cache import TreeCache
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.long_loop.runs.e12.frozen_retention import FrozenRetentionDataset
from experiments.post_training.long_loop.sft_runtime.teacher_targets import TeacherLmExample, TeacherTopK

EOS_ID = 128001
_LM_EXEMPLAR = {"input_ids": np.zeros(0, np.int32), "loss_weights": np.zeros(0, np.float32)}
_TEACHER_EXEMPLAR = {
    **_LM_EXEMPLAR,
    "teacher_ids": np.zeros(0, np.int32),
    "teacher_logprobs": np.zeros(0, np.float32),
}


def _prebuilt_dataset(cache: TreeCache, Pos: Axis, block_cross_document_attention: bool) -> PrebuiltLmDataset:
    return PrebuiltLmDataset(
        cache,
        Pos,
        input_ids_key="input_ids",
        loss_weights_key="loss_weights",
        loss_weight_transform=None,
        eos_id=EOS_ID,
        block_cross_document_attention=block_cross_document_attention,
    )


def teacher_dataset(
    path: str, Pos: Axis, topk: int, block_cross_document_attention: bool
) -> AsyncDataset[TeacherLmExample]:
    """Read packed rows with their stored teacher top-K, one row per cache entry.

    ``teacher_ids`` and ``teacher_logprobs`` are flat ``[Pos * topk]`` per entry, row-major
    ``[Pos, topk]``. A position carries teacher targets where its loss weight is positive.
    """
    cache = TreeCache.load(path, _TEACHER_EXEMPLAR)
    lm = _prebuilt_dataset(cache, Pos, block_cross_document_attention)
    shape = (Pos.size, topk)

    def attach(row: dict) -> TeacherLmExample:
        example = lm.fn(row)
        ids = np.asarray(row["teacher_ids"])
        logprobs = np.asarray(row["teacher_logprobs"])
        if ids.size != Pos.size * topk or logprobs.size != Pos.size * topk:
            raise ValueError(f"Teacher arrays have {ids.size} and {logprobs.size} entries, expected {Pos.size * topk}")
        mask = (np.asarray(example.loss_weight) > 0).astype(np.float32)
        teacher = TeacherTopK(ids=ids.reshape(shape), logprobs=logprobs.reshape(shape), mask=mask)
        return TeacherLmExample(
            tokens=example.tokens, loss_weight=example.loss_weight, attn_mask=example.attn_mask, teacher=teacher
        )

    return MappedAsyncDataset(cache, attach)


def without_teacher(dataset: AsyncDataset[GrugLmExample], Pos: Axis, topk: int) -> AsyncDataset[TeacherLmExample]:
    """Give rows without a teacher zero targets and a zero mask, so they mix with teacher rows."""
    empty = TeacherTopK(
        ids=np.zeros((Pos.size, topk), np.int32),
        logprobs=np.zeros((Pos.size, topk), np.float32),
        mask=np.zeros(Pos.size, np.float32),
    )

    def attach(example: GrugLmExample) -> TeacherLmExample:
        return TeacherLmExample(
            tokens=example.tokens, loss_weight=example.loss_weight, attn_mask=example.attn_mask, teacher=empty
        )

    return dataset.map(attach)


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
    # Number of stored teacher top-K entries per position in the repair cache; 0 reads no
    # teacher fields. Loading them never changes mixture sampling or row order.
    teacher_topk: int = 0
    # Held-out teacher-scored cache, evaluated for NLL and KL to the teacher.
    teacher_validation_cache: str = ""

    def train_mixture(self, Pos, batch_schedule, *, key):
        if not self.repair_cache or not 0 < self.repair_row_weight < 1:
            raise ValueError("E12 requires a pinned cache and calibrated target-token weight")
        # Calling super with the original key preserves each kept general row.
        # Arm0 uses the unmodified general stream; selected rows are replaced only in treatments.
        general = super().train_mixture(Pos, batch_schedule, key=key)
        if self.control_only:
            return general
        if self.teacher_topk > 0:
            general = without_teacher(general, Pos, self.teacher_topk)
            repair = teacher_dataset(self.repair_cache, Pos, self.teacher_topk, self.block_cross_document_attention)
        else:
            cache = TreeCache.load(self.repair_cache, _LM_EXEMPLAR)
            repair = _prebuilt_dataset(cache, Pos, self.block_cross_document_attention)
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
                dataset = _prebuilt_dataset(TreeCache.load(path, _LM_EXEMPLAR), Pos, self.block_cross_document_attention)
                result.append((dataset, ["e12", "e12/" + label]))
        return result

    def teacher_eval_dataset(self, Pos) -> AsyncDataset[TeacherLmExample]:
        """The held-out teacher-scored rows, in cache order."""
        if not self.teacher_validation_cache:
            raise ValueError("No teacher_validation_cache configured")
        if self.teacher_topk <= 0:
            raise ValueError("teacher_validation_cache needs teacher_topk > 0")
        return teacher_dataset(
            self.teacher_validation_cache, Pos, self.teacher_topk, self.block_cross_document_attention
        )
