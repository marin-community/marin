# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render GLM demonstrations with the frozen Snowball reasoning format and loss mask."""

from transformers import BatchEncoding, PreTrainedTokenizerBase

START_THINK = "<|start_think|>"
END_THINK = "<|end_think|>"
START_THINK_ID = 128002
END_THINK_ID = 128003
END_TURN_ID = 128009


def training_row(row: dict) -> dict:
    """Move separate teacher reasoning into the student's assistant content exactly once."""
    messages = row["messages"]
    if [message["role"] for message in messages] != ["user", "assistant"]:
        raise ValueError(f"Expected one user and one assistant turn: {row['id']}")
    reasoning = messages[1]["reasoning_content"].strip()
    answer = messages[1]["content"].strip()
    if not reasoning or not answer:
        raise ValueError(f"Missing reasoning or final answer: {row['id']}")
    if any(marker in reasoning or marker in answer for marker in (START_THINK, END_THINK, "<|eot_id|>")):
        raise ValueError(f"Teacher text already contains student control tokens: {row['id']}")
    return {
        "id": row["id"],
        "problem_id": row["problem_id"],
        "messages": [
            {"role": "user", "content": messages[0]["content"]},
            {"role": "assistant", "content": f"{reasoning}\n{END_THINK}\n\n{answer}"},
        ],
        "chat_template_kwargs": {"enable_thinking": True},
    }


def token_counts(row: dict, tokenizer: PreTrainedTokenizerBase) -> tuple[int, int]:
    """Check the native mask covers reasoning, its closing delimiter, and the answer."""
    encoded = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
        **row["chat_template_kwargs"],
    )
    assert isinstance(encoded, BatchEncoding)
    tokens = encoded["input_ids"]
    mask = encoded["assistant_masks"]
    if START_THINK_ID in tokens:
        raise ValueError(f"Unexpected opening reasoning token: {row['id']}")
    if tokens.count(END_THINK_ID) != 1 or mask[tokens.index(END_THINK_ID)] != 1:
        raise ValueError(f"Reasoning delimiter is missing, repeated or unsupervised: {row['id']}")
    prompt_tokens = tokenizer.apply_chat_template(
        row["messages"][:-1], tokenize=True, return_dict=False, add_generation_prompt=True, **row["chat_template_kwargs"]
    )
    start = len(prompt_tokens)
    if tokens[:start] != prompt_tokens:
        raise ValueError(f"Training prefix differs from the generation prompt: {row['id']}")
    if any(mask[:start]) or not all(mask[start:]):
        raise ValueError(f"Assistant loss mask omits output or includes the prompt: {row['id']}")
    if tokens[-1] != END_TURN_ID:
        raise ValueError(f"Missing final end-of-turn token: {row['id']}")
    return len(tokens), sum(mask)
