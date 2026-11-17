# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restore withheld LMSYS and WildChat prompts in Nemotron chat exports."""

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType

from datasets import load_dataset
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.factory import open_url
from rigging.filesystem.storage_path import prefix_join
from zephyr.readers import load_jsonl

SEED_DATASET_REVISIONS: Mapping[str, str] = MappingProxyType(
    {
        "allenai/WildChat-1M": "7d6490e462285cf85d91eabea0f9a954fbddcd1f",
        "lmsys/lmsys-chat-1m": "200748d9d3cddcc9d782887541057aca0b18c5da",
    }
)


def _source_seed_prompts(row: dict) -> tuple[str | None, str | None]:
    system = None
    first_user = None
    conversation = row.get("conversation") or []
    for message in conversation:
        if message.get("role") == "system" and system is None:
            system = message.get("content")
        if message.get("role") == "user":
            first_user = message.get("content")
            break
    if first_user is None and conversation:
        first_user = conversation[0].get("content")
    return system, first_user


def _protected_dataset(metadata: dict) -> str | None:
    dataset = metadata.get("seed_dataset") or metadata.get("seed_source")
    if not isinstance(dataset, str):
        return None
    normalized = dataset.lower()
    if normalized.startswith("wildchat") or normalized == "allenai/wildchat-1m":
        return "allenai/WildChat-1M"
    if normalized.startswith("lmsys"):
        return "lmsys/lmsys-chat-1m"
    return None


def _needed_hashes(path: str) -> dict[str, set[str]]:
    needed = {dataset: set() for dataset in SEED_DATASET_REVISIONS}
    for row in load_jsonl(path):
        metadata = row.get("metadata") or {}
        dataset = _protected_dataset(metadata)
        if dataset not in needed:
            continue
        first_user = next((message for message in row["messages"] if message.get("role") == "user"), None)
        if first_user is not None and first_user.get("content") is not None:
            continue
        digest = metadata.get("seed_prompt_sha256")
        if not isinstance(digest, str):
            raise ValueError(f"Missing seed_prompt_sha256 for Nemotron row {row.get('uuid')}")
        needed[dataset].add(digest)
    return needed


def _replacement_prompts(needed: Mapping[str, set[str]]) -> dict[tuple[str, str], tuple[str | None, str]]:
    replacements: dict[tuple[str, str], tuple[str | None, str]] = {}
    for dataset, digests in needed.items():
        remaining = set(digests)
        if not remaining:
            continue
        source = load_dataset(
            dataset,
            revision=SEED_DATASET_REVISIONS[dataset],
            split="train",
            streaming=True,
            token=True if dataset.startswith("lmsys/") else None,
        )
        for row in source:
            system, first_user = _source_seed_prompts(row)
            if not isinstance(first_user, str):
                continue
            digest = hashlib.sha256(first_user.encode("utf-8")).hexdigest()
            if digest in remaining:
                replacements[(dataset, digest)] = system, first_user
                remaining.remove(digest)
            if not remaining:
                break
        if remaining:
            raise ValueError(f"Could not restore {len(remaining)} prompts from {dataset}")
    return replacements


def restore_chat_row(row: dict, replacements: Mapping[tuple[str, str], tuple[str | None, str]]) -> dict:
    """Replace a withheld initial prompt using its source SHA-256 match."""
    metadata = row.get("metadata") or {}
    dataset = _protected_dataset(metadata)
    if dataset not in SEED_DATASET_REVISIONS:
        return row
    messages = row["messages"]
    first_user = next((message for message in messages if message.get("role") == "user"), None)
    if first_user is not None and first_user.get("content") is not None:
        return row
    digest = metadata.get("seed_prompt_sha256")
    if not isinstance(digest, str) or (dataset, digest) not in replacements:
        raise ValueError(f"No replacement prompt for Nemotron row {row.get('uuid')}")
    if first_user is None:
        raise ValueError(f"No user turn in Nemotron row {row.get('uuid')}")
    system, user = replacements[(dataset, digest)]
    restored = {**row, "messages": [dict(message) for message in messages]}
    if restored["messages"] and restored["messages"][0].get("role") == "system":
        if system is None:
            restored["messages"].pop(0)
            train_turns = metadata.get("train_turns")
            if isinstance(train_turns, list):
                restored["metadata"] = {**metadata, "train_turns": train_turns[1:]}
        else:
            restored["messages"][0]["content"] = system
    next(message for message in restored["messages"] if message.get("role") == "user")["content"] = user
    return restored


def restore_chat_prompts(input_path: str, output_path: str) -> None:
    """Write a prompt-complete chat JSONL from the pinned source datasets."""
    source = prefix_join(input_path, "data/chat.jsonl")
    destination = prefix_join(output_path, "data/chat.jsonl")
    replacements = _replacement_prompts(_needed_hashes(source))
    with atomic_rename(destination) as temporary_path, open_url(temporary_path, "wt") as output:
        for row in load_jsonl(source):
            restored = restore_chat_row(row, replacements)
            output.write(json.dumps(restored, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")
