# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Portable, pickle-free captures for offline GRPO learning."""

import dataclasses
import json
from dataclasses import dataclass

import numpy as np
from rigging.filesystem.storage_path import StoragePath

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CapturedRollout:
    """A complete rollout batch stored for offline GRPO learning.

    Sequences and attention masks include prompts. Response-aligned arrays
    include every sampled response position, including positions excluded from
    the objective. Group and partition IDs preserve the rollout's objective
    normalization independently of learner microbatching.
    """

    sequences: np.ndarray
    attention_mask: np.ndarray
    response_mask: np.ndarray
    loss_mask: np.ndarray
    rewards: np.ndarray
    old_logprobs: np.ndarray
    group_ids: np.ndarray
    objective_partition_ids: np.ndarray
    reference_logprobs: np.ndarray | None = None

    def validate(self) -> None:
        """Check capture dimensions, IDs, and response-mask bounds."""
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
        for name in ("response_mask", "loss_mask", "rewards", "reference_logprobs"):
            values = getattr(self, name)
            if values is not None and values.shape != (batch, width):
                raise ValueError(f"{name} must align to response logprobs")
        if np.any(self.loss_mask < 0) or np.any(self.loss_mask > self.response_mask):
            raise ValueError("loss_mask must be nonnegative and contained in response_mask")
        if np.any((self.response_mask != 0) & (self.response_mask != 1)):
            raise ValueError("response_mask must be binary")


def write_captured_rollout(uri: str, rollout: CapturedRollout, manifest: dict) -> None:
    """Write a rollout capture and its provenance through the configured filesystem."""
    rollout.validate()
    arrays = {
        field.name: value for field in dataclasses.fields(rollout) if (value := getattr(rollout, field.name)) is not None
    }
    arrays["manifest"] = np.asarray(
        json.dumps({"schema_version": SCHEMA_VERSION, **manifest}, allow_nan=False, sort_keys=True)
    )
    with StoragePath(uri).open("wb") as output:
        np.savez_compressed(output, **arrays)


def read_captured_rollout(uri: str) -> tuple[CapturedRollout, dict]:
    """Read a capture without allowing executable pickle payloads."""
    with StoragePath(uri).open("rb") as source, np.load(source, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"]))
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"Unsupported GRPO capture schema: {manifest['schema_version']}")
        rollout = CapturedRollout(
            **{field.name: archive[field.name] for field in dataclasses.fields(CapturedRollout) if field.name in archive}
        )
    rollout.validate()
    return rollout, manifest
