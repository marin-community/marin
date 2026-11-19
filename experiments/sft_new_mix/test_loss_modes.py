# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.sft_new_mix.loss_modes import LossMode, prediction_weights


def test_chat_masks_tool_and_student_turns_at_prediction_positions():
    # [header, role, end-header] with literal 8/9 tokens as message bodies.
    tokens = np.array([1, 2, 3, 8, 1, 4, 3, 8, 9, 1, 5, 3, 8, 1, 4, 3, 9], dtype=np.int32)
    weights = prediction_weights(
        tokens,
        loss_mode=LossMode.CHAT_ASSISTANT_ONLY,
        message_start_id=1,
        assistant_prefix=(1, 4, 3),
        trained_assistant_turns=(False, True),
    )
    np.testing.assert_array_equal(np.flatnonzero(weights), [15])
    ordinary = prediction_weights(
        tokens,
        loss_mode=LossMode.CHAT_ASSISTANT_ONLY,
        message_start_id=1,
        assistant_prefix=(1, 4, 3),
        trained_assistant_turns=None,
    )
    np.testing.assert_array_equal(np.flatnonzero(ordinary), [6, 7, 15])


def test_nonchat_full_loss_stops_at_the_document_boundary():
    tokens = np.array([9, 8, 7, 6], dtype=np.int32)
    weights = prediction_weights(
        tokens,
        loss_mode=LossMode.FULL_TOKEN,
        message_start_id=1,
        assistant_prefix=(1, 4, 3),
        trained_assistant_turns=None,
    )
    np.testing.assert_array_equal(weights, [1, 1, 1, 0])
