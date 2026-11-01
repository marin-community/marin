# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prompt files and tokenization shared by the torch harness and the vLLM reference.

A prompt file is JSONL (s3:// or local; repository .gitignore drops *.jsonl, so keep prompt sets in object
storage). Each row has an ``id`` and either ``messages`` (chat turns, rendered with the checkpoint's chat template
and a generation prompt, with any ``chat_template_kwargs`` such as ``{"enable_thinking": false}``) or ``text`` (raw
text, tokenized as is). Optional ``prefill`` is assistant text appended after the generation prompt. Optional
``targets`` lists continuations whose log-probabilities the harness reports. ``--prompts check`` selects the
built-in logit-check set below.

This module must not import torch: the vLLM reference job runs without it.
"""

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join
from transformers import AutoTokenizer

SNOWBALL_V01 = "hf://open-athena/Marin-Snowball-67B-A2B-v0.1@adc0078a19a4383bc998fcdb43867b3360baf5f3"
"""Snowball v0.1, pinned. The same 39 shards are in-region for cw-us-east-02a at
s3://marin-us-east-02a/marin/models/open-athena--Snowball-Smartie-RL-2026-10-15-32-step14--6783e9743478bd25a9ce03ab3460e84d6a9814b0;
pass that as ``--weights`` for a job there."""


def hf_repo_revision(weights: str) -> tuple[str, str | None] | None:
    """``(repo, revision)`` for an ``hf://repo[@revision]`` path, else None."""
    if not weights.startswith("hf://"):
        return None
    repo, _, revision = weights.removeprefix("hf://").partition("@")
    return repo, revision or None


TOKENIZER_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "added_tokens.json",
)

CHECK_PROMPTS = [
    {"id": "who-are-you", "messages": [{"role": "user", "content": "Who are you?"}]},
    {"id": "who-made-you", "messages": [{"role": "user", "content": "Who created you?"}]},
    {"id": "your-name", "messages": [{"role": "user", "content": "What is your name? Answer in one word."}]},
    {
        "id": "which-model",
        "messages": [{"role": "user", "content": "Which language model am I talking to, and which company trained it?"}],
    },
    {
        "id": "system-snowball",
        "messages": [
            {"role": "system", "content": "You are Marin Snowball, a language model made by Open Athena."},
            {"role": "user", "content": "Who are you?"},
        ],
    },
    {"id": "who-made-chatgpt", "messages": [{"role": "user", "content": "Which company made ChatGPT?"}]},
    {"id": "capital", "messages": [{"role": "user", "content": "What is the capital of France?"}]},
    {"id": "arithmetic", "messages": [{"role": "user", "content": "What is 17 times 23? Give only the number."}]},
    {"id": "code", "messages": [{"role": "user", "content": "Write a Python function that reverses a string."}]},
    {
        "id": "fiction",
        "messages": [{"role": "user", "content": "Write the opening line of a short story about a lighthouse keeper."}],
    },
    {"id": "raw-created-by", "text": "I am an AI assistant. I was created by Open"},
]
"""Logit-check set: identity questions, a system-prompt identity, controls, and a raw-text completion."""


@dataclass(frozen=True)
class Prompt:
    id: str
    tokens: tuple[int, ...]
    assistant_start: int
    """First position of the assistant turn (its header), or 0 for raw text."""
    targets: tuple[str, ...] = field(default=())


def load_tokenizer(weights: str):
    fs, path = filesystem_for(weights)
    with tempfile.TemporaryDirectory() as directory:
        for name in TOKENIZER_FILES:
            if fs.exists(prefix_join(path, name)):
                Path(directory, name).write_bytes(fs.cat_file(prefix_join(path, name)))
        return AutoTokenizer.from_pretrained(directory)


def _ids(encoded) -> list[int]:
    return list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)


def encode(tokenizer, row: dict) -> Prompt:
    if "messages" in row:
        if row["messages"][-1]["role"] == "assistant":
            raise ValueError(f"{row['id']}: the last message must not be an assistant turn")
        kwargs = row.get("chat_template_kwargs", {})
        tokens = _ids(
            tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True, tokenize=True, **kwargs)
        )
        prefix = _ids(
            tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=True, **kwargs)
        )
        if tokens[: len(prefix)] != prefix:
            raise ValueError(f"{row['id']}: the generation prompt does not extend the rendered chat")
        start = len(prefix)
    else:
        tokens = tokenizer.encode(row["text"], add_special_tokens=False)
        start = 0
    if row.get("prefill"):
        tokens = tokens + tokenizer(row["prefill"], add_special_tokens=False)["input_ids"]
    return Prompt(row["id"], tuple(tokens), start, tuple(row.get("targets", ())))


def read_prompts(path: str, tokenizer) -> list[Prompt]:
    if path == "check":
        rows = CHECK_PROMPTS
    else:
        fs, local = filesystem_for(path)
        rows = [json.loads(line) for line in fs.cat_file(local).decode().splitlines() if line.strip()]
    prompts = [encode(tokenizer, row) for row in rows]
    if len({prompt.id for prompt in prompts}) != len(prompts):
        raise ValueError(f"Duplicate prompt ids in {path}")
    return prompts
