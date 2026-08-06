# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light helpers for the fixed Grug replay benchmark."""

import numpy as np


def build_loss_weight(loss_mask: np.ndarray, sequence_length: int) -> np.ndarray:
    """Align SkyRL action-token weights with native next-token loss positions."""

    if loss_mask.ndim != 2:
        raise ValueError(f"loss_mask must be rank 2, got {loss_mask.shape}")
    action_length = int(loss_mask.shape[1])
    start = sequence_length - action_length - 1
    end = sequence_length - 1
    if start < 0:
        raise ValueError(f"action length {action_length} does not fit sequence length {sequence_length}")
    result = np.zeros((loss_mask.shape[0], sequence_length), dtype=np.float32)
    result[:, start:end] = loss_mask.astype(np.float32, copy=False)
    return result


def repacked_operational_micro_loss(
    cross_entropy_sum,
    router_aux_loss,
    *,
    global_loss_tokens: int,
    microbatch_count: int,
):
    """Scale one repacked microbatch like one logical production batch.

    Token losses are additive, so each microbatch contributes its CE sum
    divided by the logical batch's total loss-token count. Router loss is
    already a mean statistic, so the logical update uses its arithmetic mean
    across the repacked microbatches.
    """

    if global_loss_tokens <= 0:
        raise ValueError("global_loss_tokens must be positive")
    if microbatch_count <= 0:
        raise ValueError("microbatch_count must be positive")
    return cross_entropy_sum / global_loss_tokens + router_aux_loss / microbatch_count
