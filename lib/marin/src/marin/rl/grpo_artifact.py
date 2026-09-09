# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Portable, pickle-free captures at the rollout-to-learner boundary."""

import dataclasses
import json
from dataclasses import dataclass

import numpy as np
from rigging.filesystem.storage_path import StoragePath


@dataclass(frozen=True)
class GoldenRollout:
    """A complete admitted batch, with distinct sampling and raw-policy scores.

    Sequences and attention masks include prompts. Remaining token arrays are
    aligned to response positions, including environment tokens masked out of
    the objective. Partition IDs describe the oracle's loss normalization groups,
    independently of the execution microbatches used to replay the objective.
    """

    sequences: np.ndarray
    attention_mask: np.ndarray
    response_mask: np.ndarray
    loss_mask: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    old_logprobs: np.ndarray
    behavior_logprobs: np.ndarray
    group_ids: np.ndarray
    objective_partition_ids: np.ndarray
    reference_logprobs: np.ndarray | None = None
    current_logprobs: np.ndarray | None = None
    logprob_gradients: np.ndarray | None = None

    def validate(self) -> None:
        if self.old_logprobs.ndim != 2:
            raise ValueError("old_logprobs must be a nonempty trajectory-by-response array")
        batch, width = self.old_logprobs.shape
        if batch == 0 or width == 0:
            raise ValueError("old_logprobs must be a nonempty trajectory-by-response array")
        if self.sequences.ndim != 2 or self.sequences.shape[0] != batch:
            raise ValueError("sequences must have one row per trajectory")
        if width > self.sequences.shape[1]:
            raise ValueError("response width cannot exceed sequence width")
        if self.attention_mask.shape != self.sequences.shape:
            raise ValueError("attention_mask must align to sequences")
        for name in ("group_ids", "objective_partition_ids"):
            values = getattr(self, name)
            if values.shape != (batch,) or values.dtype.kind not in "iu":
                raise ValueError(f"{name} must be integer trajectory IDs")
        if self.sequences.dtype.kind not in "iu":
            raise ValueError("sequences must contain integer token IDs")
        for field in dataclasses.fields(self):
            if field.name in ("sequences", "attention_mask", "group_ids", "objective_partition_ids"):
                continue
            values = getattr(self, field.name)
            if values is not None and values.shape != (batch, width):
                raise ValueError(f"{field.name} must align to response logprobs")
        if np.any(self.loss_mask < 0) or np.any(self.loss_mask > self.response_mask):
            raise ValueError("loss_mask must be nonnegative and contained in response_mask")
        if np.any((self.response_mask != 0) & (self.response_mask != 1)):
            raise ValueError("response_mask must be binary")


def write_golden_rollout(uri: str, batch: GoldenRollout, manifest: dict) -> None:
    """Write arrays and their provenance together through the configured filesystem."""
    batch.validate()
    arrays = {
        field.name: value for field in dataclasses.fields(batch) if (value := getattr(batch, field.name)) is not None
    }
    arrays["manifest"] = np.asarray(json.dumps({"schema_version": 1, **manifest}, allow_nan=False, sort_keys=True))
    with StoragePath(uri).open("wb") as output:
        np.savez_compressed(output, **arrays)


def read_golden_rollout(uri: str) -> tuple[GoldenRollout, dict]:
    """Read a capture without allowing executable pickle payloads."""
    with StoragePath(uri).open("rb") as source, np.load(source, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"]))
        if manifest["schema_version"] != 1:
            raise ValueError(f"Unsupported golden rollout schema: {manifest['schema_version']}")
        batch = GoldenRollout(
            **{field.name: archive[field.name] for field in dataclasses.fields(GoldenRollout) if field.name in archive}
        )
    batch.validate()
    return batch, manifest
