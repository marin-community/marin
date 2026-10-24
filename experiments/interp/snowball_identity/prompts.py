# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prompt files and tokenization shared by the torch harness and the vLLM reference.

A prompt file is JSONL. Each row has an ``id`` and either ``messages`` (chat turns, rendered with the
checkpoint's chat template and a generation prompt) or ``text`` (raw text, tokenized as is). Optional
``targets`` lists continuations whose log-probabilities the harness reports.
"""

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from marin.merging.checkpoint import METADATA_NAMES
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
        for name in METADATA_NAMES:
            if fs.exists(prefix_join(path, name)):
                Path(directory, name).write_bytes(fs.cat_file(prefix_join(path, name)))
        return AutoTokenizer.from_pretrained(directory)


def _ids(encoded) -> list[int]:
    return list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)


def encode(tokenizer, row: dict) -> Prompt:
    if "messages" in row:
        if row["messages"][-1]["role"] == "assistant":
            raise ValueError(f"{row['id']}: the last message must not be an assistant turn")
        tokens = _ids(tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True, tokenize=True))
        prefix = _ids(tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=True))
        if tokens[: len(prefix)] != prefix:
            raise ValueError(f"{row['id']}: the generation prompt does not extend the rendered chat")
        start = len(prefix)
    else:
        tokens = tokenizer.encode(row["text"], add_special_tokens=False)
        start = 0
    return Prompt(row["id"], tuple(tokens), start, tuple(row.get("targets", ())))


def read_prompts(path: str, tokenizer) -> list[Prompt]:
    fs, local = filesystem_for(path)
    rows = [json.loads(line) for line in fs.cat_file(local).decode().splitlines() if line.strip()]
    prompts = [encode(tokenizer, row) for row in rows]
    if len({prompt.id for prompt in prompts}) != len(prompts):
        raise ValueError(f"Duplicate prompt ids in {path}")
    return prompts
