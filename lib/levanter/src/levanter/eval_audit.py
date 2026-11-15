# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Pin evaluation inputs and check an initial model against a reference loss."""

import hashlib
import json
import math
from dataclasses import dataclass

import jax
import numpy as np
from jax.experimental import multihost_utils
from rigging.filesystem.storage_path import StoragePath

from levanter.data.text.datasets import LmDataConfig
from levanter.eval import TaggedEvaluator
from levanter.layers.attention import AttentionMask


@dataclass(frozen=True)
class InitialEvalMatchConfig:
    data: LmDataConfig
    reference_path: str
    sequence_length: int
    max_examples_per_dataset: int
    special_token_ids: tuple[int, ...]
    absolute_tolerance: float = 1e-3
    pre_reseed_reference_path: str | None = None


def evaluation_input_audit(evaluator: TaggedEvaluator, special_token_ids: tuple[int, ...]) -> dict:
    """Hash the global token stream and masks, independently of evaluation batch size."""
    tokens_hash = hashlib.sha256()
    weights_hash = hashlib.sha256()
    segments_hash = hashlib.sha256()
    positions = 0
    scored_tokens = 0.0
    counts = {str(token): 0 for token in special_token_ids}
    for batch, _tags in evaluator.loader:
        tokens = np.asarray(multihost_utils.process_allgather(batch.tokens.array, tiled=True), dtype=np.int32)
        weights = np.asarray(multihost_utils.process_allgather(batch.loss_weight.array, tiled=True), dtype=np.float32)
        tokens_hash.update(tokens.tobytes())
        weights_hash.update(weights.tobytes())
        if not isinstance(batch.attn_mask, AttentionMask):
            raise ValueError("Evaluation identity auditing requires a structured attention mask")
        segment_ids = batch.attn_mask.segment_ids
        segments = (
            np.zeros((*tokens.shape, 2), dtype=np.int64)
            if segment_ids is None
            else np.stack(
                [
                    np.asarray(multihost_utils.process_allgather(ids.array, tiled=True), dtype=np.int64)
                    for ids in segment_ids
                ],
                axis=-1,
            )
        )
        segments_hash.update(segments.tobytes())
        positions += tokens.size
        scored_tokens += float(weights.sum(dtype=np.float64))
        for token in special_token_ids:
            counts[str(token)] += int(np.count_nonzero(tokens == token))
    if scored_tokens <= 0:
        raise ValueError("Initial evaluation has no scored tokens")
    return {
        "tokens_sha256": tokens_hash.hexdigest(),
        "weights_sha256": weights_hash.hexdigest(),
        "segments_sha256": segments_hash.hexdigest(),
        "positions": positions,
        "scored_tokens": scored_tokens,
        "special_token_counts": counts,
    }


def initial_eval_match(reference: dict, metrics: dict, audit: dict, absolute_tolerance: float) -> dict:
    """Reject changed evaluation inputs or an initial loss outside the specified tolerance."""
    if reference["input_audit"] != audit:
        raise ValueError("Initial evaluation inputs do not match the reference token stream and masks")
    expected = float(reference["metrics"]["eval/loss"])
    actual = float(metrics["eval/loss"])
    delta = actual - expected
    if not math.isfinite(expected) or not math.isfinite(actual) or abs(delta) > absolute_tolerance:
        raise ValueError(f"Initial loss mismatch: reference={expected}, actual={actual}, delta={delta}")
    return {
        "passed": True,
        "reference_loss": expected,
        "initial_loss": actual,
        "delta": delta,
        "absolute_tolerance": absolute_tolerance,
        "input_audit": audit,
    }


def write_evaluation_result(path: str, metrics: dict, input_audit: dict) -> None:
    if jax.process_index() == 0:
        values = {name: float(value) for name, value in metrics.items()}
        StoragePath(path).write_text(json.dumps({"metrics": values, "input_audit": input_audit}, indent=2))
