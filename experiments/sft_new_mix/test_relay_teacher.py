# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.sft_new_mix.loss_modes import LossMode, prediction_weights
from experiments.sft_new_mix.relay_teacher import render_teacher_transcript


def test_teacher_reasoning_and_commands_share_the_source_flag():
    transcript = render_teacher_transcript(
        [
            {"role": "system", "content": "Reasoning: /think", "trained": False},
            {"role": "user", "content": "Solve the task.", "trained": False},
            {"role": "assistant", "content": '{"commands": [{"keystrokes": "pwd"}]}', "trained": False},
            {"role": "user", "content": "/workspace", "trained": False},
            {
                "role": "assistant",
                "content": '<|start_think|>teacher reasoning<|end_think|>{"commands": [], "task_complete": true}',
                "trained": True,
            },
        ]
    )
    assert transcript.trained_assistant_turns == (False, True)
    # Character-level IDs make the actual rendered boundaries and shift visible.
    vocabulary = {character: index for index, character in enumerate(sorted(set(transcript.text)))}
    # The production masker uses a reserved one-token header marker; collapse
    # that marker while retaining every other actual rendered character.
    header = "<|start_header_id|>"
    chunks = transcript.text.split(header)
    start_id = len(vocabulary)
    arrays = []
    for index, chunk in enumerate(chunks):
        if index:
            arrays.append(start_id)
        arrays.extend(vocabulary[character] for character in chunk)
    ids = np.asarray(arrays, dtype=np.int32)
    prefix = (start_id, *(vocabulary[c] for c in "assistant<|end_header_id|>\n"))
    weights = prediction_weights(
        ids,
        loss_mode=LossMode.CHAT_ASSISTANT_ONLY,
        message_start_id=start_id,
        assistant_prefix=prefix,
        trained_assistant_turns=transcript.trained_assistant_turns,
    )
    trained_target_indices = np.flatnonzero(np.roll(weights, 1))
    teacher_start = int(np.flatnonzero(ids == start_id)[-1]) + len(prefix)
    np.testing.assert_array_equal(trained_target_indices, np.arange(teacher_start, len(ids)))
