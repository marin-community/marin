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
