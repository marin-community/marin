# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.datakit.download.glm_kernelgym_rollouts import (
    TruncationFilter,
    chat_conversation_messages,
    row_to_chat_doc,
)


def test_chat_conversation_restores_reasoning_and_merges_user_context() -> None:
    first_response = "inspect the kernel</think>Use shared memory."
    final_response = "verify the result</think>The optimized kernel passes."
    messages = [
        {"role": "user", "content": "Optimize this kernel."},
        {"role": "assistant", "content": first_response},
        {"role": "user", "content": "The benchmark improved."},
        {"role": "user", "content": "Should another attempt be made?"},
        {"role": "assistant", "content": "Yes, continue."},
        {"role": "user", "content": "Try one more optimization."},
        {"role": "assistant", "content": final_response},
        {"role": "user", "content": "KERNELGYM_FINAL"},
        {"role": "assistant", "content": "KERNELGYM_FINAL"},
    ]
    turns = [{"response": first_response}, {"response": final_response}]

    assert chat_conversation_messages(messages, turns) == [
        {"role": "user", "content": "Optimize this kernel."},
        {
            "role": "assistant",
            "content": "<|start_think|>inspect the kernel<|end_think|>Use shared memory.",
        },
        {
            "role": "user",
            "content": "The benchmark improved.\n\nTry one more optimization.",
        },
        {
            "role": "assistant",
            "content": "<|start_think|>verify the result<|end_think|>The optimized kernel passes.",
        },
    ]


def test_chat_row_drops_trajectory_with_any_truncated_generation() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Optimize this kernel."},
            {"role": "assistant", "content": "unfinished reasoning"},
            {"role": "user", "content": "Try again."},
            {"role": "assistant", "content": "done</think>Final kernel."},
        ],
        "turns": [
            {"response": "unfinished reasoning", "usage": {"completion_tokens": 16}},
            {"response": "done</think>Final kernel.", "usage": {"completion_tokens": 4}},
        ],
        "max_tokens": 16,
    }

    assert row_to_chat_doc(row, TruncationFilter.ANY_TURN) == []


def test_chat_row_drops_multiple_reasoning_boundaries() -> None:
    response = "first</think>answer text containing another </think> marker"
    row = {
        "messages": [
            {"role": "user", "content": "Optimize this kernel."},
            {"role": "assistant", "content": response},
        ],
        "turns": [{"response": response, "usage": {"completion_tokens": 8}}],
        "max_tokens": 16,
    }

    assert row_to_chat_doc(row, TruncationFilter.ANY_TURN) == []
