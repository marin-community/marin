# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit source loss metadata and prediction-indexed cache weights."""

from enum import StrEnum

import numpy as np


class LossMode(StrEnum):
    CHAT_ASSISTANT_ONLY = "chat_assistant_only"
    FULL_TOKEN = "full_token"


def prediction_weights(
    tokens: np.ndarray,
    *,
    loss_mode: LossMode,
    message_start_id: int,
    assistant_prefix: tuple[int, ...],
    trained_assistant_turns: tuple[bool, ...] | None,
) -> np.ndarray:
    """Weight predictions of assistant bodies, or every next token for non-chat.

    A supplied teacher map has exactly one flag per rendered assistant turn.
    Weights index the prediction at the preceding token; the document's final
    position is zero so packing never trains across a document boundary.
    """
    if loss_mode == LossMode.FULL_TOKEN:
        if trained_assistant_turns is not None:
            raise ValueError("Teacher masks require chat loss mode")
        weights = np.ones(len(tokens), dtype=np.float32)
        weights[-1] = 0
        return weights
    starts = np.flatnonzero(tokens == message_start_id)
    target = np.zeros(len(tokens), dtype=np.float32)
    assistant_index = 0
    prefix = np.asarray(assistant_prefix, dtype=np.int32)
    if len(prefix) == 0 or prefix[0] != message_start_id:
        raise ValueError("Assistant prefix must begin at a message boundary")
    for index, start in enumerate(starts):
        if not np.array_equal(tokens[start : start + len(prefix)], prefix):
            continue
        trained = True
        if trained_assistant_turns is not None:
            if assistant_index >= len(trained_assistant_turns):
                raise ValueError("Too few teacher flags for rendered assistant turns")
            trained = trained_assistant_turns[assistant_index]
        assistant_index += 1
        if trained:
            end = int(starts[index + 1]) if index + 1 < len(starts) else len(tokens)
            target[start + len(prefix) : end] = 1
    if trained_assistant_turns is not None and assistant_index != len(trained_assistant_turns):
        raise ValueError("Teacher flags do not match rendered assistant turn count")
    if not assistant_index:
        raise ValueError("Chat source has no matching assistant headers")
    weights = np.roll(target, -1)
    weights[-1] = 0
    return weights
