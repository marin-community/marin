# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render pilot agentic traces into Grug chat-template tokens with an assistant-only loss mask."""

import json
import re

THINK = re.compile(r"\A\s*<think>(.*?)</think>(.*)\Z", re.DOTALL)


def tmax_messages(row: dict) -> tuple[list[dict], list[dict]]:
    """Convert one allenai/tmax-sft row to template messages and tool definitions.

    Rows come from pyarrow ``to_pylist``, so tool calls are plain dicts.
    """
    messages = []
    for message in row["messages"]:
        converted = {"role": message["role"], "content": message["content"] or ""}
        if message["role"] == "assistant":
            if message["reasoning_content"]:
                converted["reasoning_content"] = message["reasoning_content"]
            calls = message["tool_calls"] or []
            if calls:
                converted["tool_calls"] = calls
        elif message["role"] == "tool":
            ids = message["tool_call_ids"] or []
            if len(ids) != 1:
                raise ValueError(f"Expected one tool-call id per tool message, got {ids}")
            converted["tool_call_id"] = ids[0]
        messages.append(converted)
    return messages, json.loads(row["tools"])


def kimi_messages(row: dict) -> list[dict]:
    """Convert one Kimi SWE-smith terminus-2 row, moving inline <think> blocks into reasoning_content."""
    messages = []
    for message in row["conversations"]:
        converted = {"role": message["role"], "content": message["content"]}
        if message["role"] == "assistant":
            match = THINK.match(message["content"])
            if match:
                converted = {"role": "assistant", "reasoning_content": match[1].strip(), "content": match[2]}
            elif "<think>" in message["content"] or "</think>" in message["content"]:
                raise ValueError("Unpaired think delimiter in an assistant turn")
        messages.append(converted)
    return messages


def render(tokenizer, messages: list[dict], tools: list[dict] | None = None) -> tuple[list[int], list[int]]:
    """Return token ids and a per-token assistant mask (1 inside the template's generation blocks)."""
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        enable_thinking=True,
    )
    ids, mask = list(encoded["input_ids"]), list(encoded["assistant_masks"])
    if len(ids) != len(mask) or not any(mask):
        raise ValueError("Assistant mask is empty or misaligned")
    return ids, mask


def training_row(ids: list[int], mask: list[int], context: int, pad_token: int) -> dict:
    """Pad to the context and shift the mask so weight i scores the prediction of token i + 1."""
    if len(ids) > context:
        raise ValueError("Refusing to truncate a trajectory")
    padding = context - len(ids)
    return {"input_ids": ids + [pad_token] * padding, "loss_weights": mask[1:] + [0] * (padding + 1)}
