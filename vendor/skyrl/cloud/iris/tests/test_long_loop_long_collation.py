# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise long-loop collation at the requested window, without GPU qualification claims."""

from types import SimpleNamespace

import torch

from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl_train.long_context_qualification import qualify_learner_batch


def test_full_window_preserves_both_assistant_turns_and_masks_tool_output():
    prompt = [11] * 1024
    response = [21] * 1024 + [31] * 99328 + [41] * 1024
    mask = [1] * 1024 + [0] * 99328 + [1] * 1024
    behavior = [-0.25] * 1024 + [0.0] * 99328 + [-0.75] * 1024
    result = convert_prompts_responses_to_batch_tensors(
        SimpleNamespace(pad_token_id=0),
        [prompt, [12]],
        [response, [22, 42]],
        [[0.0] * (len(response) - 1) + [1.0], [0.0, 1.0]],
        [mask, [1, 1]],
        [behavior, [-0.5, -0.5]],
    )
    sequences, attention, response_mask, _, loss, logprobs, _, _ = result
    assert sequences[0].tolist() == prompt + response
    assert loss[0].tolist() == mask
    assert logprobs[0].tolist() == behavior
    assert loss[1, :2].tolist() == [1, 1]
    assert not torch.any(loss[1, 2:])
    evidence = qualify_learner_batch(
        {"attention_mask": attention, "response_mask": response_mask,
         "loss_mask": loss, "rollout_logprobs": logprobs},
        minimum_tokens=65537, maximum_tokens=102400,
    )
    assert evidence["longest_unpadded_tokens"] == 102400
    assert evidence["action_tokens"] == 2050
