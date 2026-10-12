# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay grading text for this campaign's Qwen and Marin tokenizer transports."""

from skyrl_gym.envs.nemotron_ultra.answer_extraction import REASONING_DELIMITERS, final_answer_text


def grading_text(
    decoded: str,
    messages: list[dict],
    preserve_full_response: bool,
    *,
    eos_token_text: str = "<|im_end|>",
) -> str:
    # Match DirectModelClient: remove one configured tokenizer EOS before decoding.
    semantic = decoded.removesuffix(eos_token_text)
    if preserve_full_response:
        return semantic
    if any(marker in semantic for pair in REASONING_DELIMITERS for marker in pair):
        return final_answer_text(semantic)
    return next(message["content"] for message in reversed(messages) if message["role"] == "assistant")
