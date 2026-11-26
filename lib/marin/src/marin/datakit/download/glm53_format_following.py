# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 format-following answers paired with their original WildChat prompts."""

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterator
from functools import partial

import fsspec
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from openai_harmony import Message, Role
from rigging.filesystem.storage_path import prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from marin.datakit.chat_normalize import CHAT_SCHEMA, ChatChannel, normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import (
    CHAT_CONTROL_TOKEN,
    REASONING_TOKEN,
    chat_document,
    load_parquet_batched,
)
from marin.execution.step_spec import StepSpec

WILDCHAT_NAME = "wildchat-glm53-format-completions"
WILDCHAT_REPO = "open-athena/" + WILDCHAT_NAME
WILDCHAT_REVISION = "e042ff3eed9c6f5fd0d2b4773c1a35fe49619769"
NUM_SHARDS = 32
SOURCE_REPO = "allenai/WildChat-4.8M"
SOURCE_REVISION = "c827c6df8fcf008219ffaffa4d1dd77491099367"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def wildchat_document(row: dict, source: dict) -> dict:
    """Recover the exact single user turn and verify its pinned WildChat identity."""
    index = row["source_turn_index"]
    if index < 0:
        raise ValueError("Negative WildChat turn index")
    message = source["conversation"][index]
    prompt = message["content"].strip()
    if (
        source["conversation_hash"] != row["conversation_hash"]
        or message["role"] != "user"
        or str(message.get("turn_identifier")) != row["turn_identifier"]
        or _sha256(prompt) != row["prompt_sha256"]
        or _sha256(re.sub(r"\s+", " ", prompt).strip().casefold()) != row["source_id"]
    ):
        raise ValueError("WildChat source reference mismatch")
    prompt += "\n\n" + row["format_instruction"]
    answer = row["answer"]
    if not prompt.strip() or not answer.strip():
        raise ValueError("A completion must have a nonempty prompt and answer")
    if any(pattern.search(text) for pattern in (CHAT_CONTROL_TOKEN, REASONING_TOKEN) for text in (prompt, answer)):
        raise ValueError("Completion data contains native chat control tokens")
    return chat_document(
        [
            Message.from_role_and_content(Role.USER, prompt),
            Message.from_role_and_content(Role.ASSISTANT, answer).with_channel(ChatChannel.FINAL),
        ],
        WILDCHAT_REPO,
        source_id=row["pair_id"],
    )


def resolve_reference_file(filename: str, rows: Iterator[dict], *, source_root: str) -> Iterator[dict]:
    """Yield chat documents after validating their source references."""
    if re.fullmatch(r"data/train-[0-9]{5}-of-[0-9]{5}\.parquet", filename) is None:
        raise ValueError("Invalid source filename")
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if (
            row["source_dataset"] != SOURCE_REPO
            or row["dataset_revision"] != SOURCE_REVISION
            or row["source_file"] != filename
            or row["source_split"] != "train"
        ):
            raise ValueError("Unexpected source dataset, revision, file, or split")
        if row["source_row_group"] < 0 or row["source_row_in_group"] < 0:
            raise ValueError("Negative source row coordinate")
        groups[row["source_row_group"]].append(row)
    with fsspec.open(prefix_join(source_root, filename), "rb", block_size=1024 * 1024) as stream:
        parquet = pq.ParquetFile(stream)
        for group, references in sorted(groups.items()):
            sources = parquet.read_row_group(group, columns=["conversation_hash", "conversation"]).to_pylist()
            for row in references:
                yield wildchat_document(row, sources[row["source_row_in_group"]])


def transform_chat(input_path: str, output_path: str) -> None:
    source_root = f"hf://datasets/{SOURCE_REPO}@{SOURCE_REVISION}"
    pipeline = (
        Dataset.from_files(f"{input_path}/data/train-*.parquet")
        .flat_map(load_parquet_batched)
        .group_by(
            key=lambda row: row["source_file"],
            reducer=partial(resolve_reference_file, source_root=source_root),
            num_output_shards=NUM_SHARDS,
        )
        .write_parquet(
            f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", schema=CHAT_SCHEMA, skip_existing=True
        )
    )
    ZephyrContext(name=WILDCHAT_NAME, resources=ResourceConfig(cpu=1, ram="8g"), max_workers=NUM_SHARDS).execute(
        pipeline
    )


def glm53_format_following_chat_normalize_steps() -> tuple[StepSpec, ...]:
    name = WILDCHAT_NAME
    download = download_hf_step(
        f"raw/{name}",
        hf_dataset_id=WILDCHAT_REPO,
        revision=WILDCHAT_REVISION,
        hf_urls_glob=["data/train-*.parquet"],
    )
    processed = StepSpec(
        name=f"processed-chat/{name}",
        deps=[download],
        fn=lambda output_path: transform_chat(download.output_path, output_path),
        hash_attrs={
            "version": "2026.09.13",
            "source_revision": SOURCE_REVISION,
            "split": "train",
        },
    )
    return processed, normalize_chat_step(name=f"normalized-chat/{name}", download=processed)
