# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Join completion-only releases to their referenced source rows."""

import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial

import fsspec
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from openai_harmony import Message, Role
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


def completion_document(prompt: str, answer: str, source: str, source_id: str) -> dict:
    """Keep quoted history and literal output formats as text, without interpreting tool tags."""
    if not prompt.strip() or not answer.strip():
        raise ValueError("A completion must have a nonempty prompt and answer")
    if any(pattern.search(text) for pattern in (CHAT_CONTROL_TOKEN, REASONING_TOKEN) for text in (prompt, answer)):
        raise ValueError("Completion data contains native chat control tokens")
    return chat_document(
        [
            Message.from_role_and_content(Role.USER, prompt),
            Message.from_role_and_content(Role.ASSISTANT, answer).with_channel(ChatChannel.FINAL),
        ],
        source,
        source_id=source_id,
    )


@dataclass(frozen=True)
class ReferencedCompletion:
    name: str
    revision: str
    repo: str
    source_repo: str
    source_revision: str
    columns: tuple[str, ...]
    document: Callable[[dict, dict], dict]
    splits: tuple[str, ...] = ("train",)


def resolve_reference_file(
    filename: str, rows: Iterator[dict], *, config: ReferencedCompletion, source_root: str
) -> Iterator[dict]:
    """Read each referenced Parquet row group once, projecting only source prompt columns."""
    if re.fullmatch(r"data/train-[0-9]{5}-of-[0-9]{5}\.parquet", filename) is None:
        raise ValueError("Invalid source filename")
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if (
            row["source_dataset"] != config.source_repo
            or row["dataset_revision"] != config.source_revision
            or row["source_file"] != filename
            or row["source_split"] != "train"
        ):
            raise ValueError("Unexpected source dataset, revision, file, or split")
        if row["source_row_group"] < 0 or row["source_row_in_group"] < 0:
            raise ValueError("Negative source row coordinate")
        groups[row["source_row_group"]].append(row)
    with fsspec.open(f"{source_root}/{filename}", "rb", block_size=1024 * 1024) as stream:
        parquet = pq.ParquetFile(stream)
        for group, references in sorted(groups.items()):
            sources = parquet.read_row_group(group, columns=list(config.columns)).to_pylist()
            for row in references:
                yield config.document(row, sources[row["source_row_in_group"]])


def transform_chat(input_path: str, output_path: str, config: ReferencedCompletion, split: str) -> None:
    source_root = f"hf://datasets/{config.source_repo}@{config.source_revision}"
    pipeline = (
        Dataset.from_files(f"{input_path}/data/{split}-*.parquet")
        .flat_map(load_parquet_batched)
        .group_by(
            key=lambda row: row["source_file"],
            reducer=partial(resolve_reference_file, config=config, source_root=source_root),
            num_output_shards=32,
        )
        .write_parquet(
            f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", schema=CHAT_SCHEMA, skip_existing=True
        )
    )
    ZephyrContext(name=config.name, resources=ResourceConfig(cpu=1, ram="8g"), max_workers=32).execute(pipeline)


def chat_normalize_steps(
    config: ReferencedCompletion, split: str, *, prompt_hash_attrs: dict[str, str]
) -> tuple[StepSpec, ...]:
    if split not in config.splits:
        raise ValueError(f"Unsupported split {split} for {config.name}")
    name = config.name if split == "train" else f"{config.name}/{split}"
    download = download_hf_step(
        f"raw/{name}",
        hf_dataset_id=config.repo,
        revision=config.revision,
        hf_urls_glob=[f"data/{split}-*.parquet"],
    )
    processed = StepSpec(
        name=f"processed-chat/{name}",
        deps=[download],
        fn=lambda output_path: transform_chat(download.output_path, output_path, config, split),
        hash_attrs={
            "version": "2026.09.13",
            "source_revision": config.source_revision,
            **prompt_hash_attrs,
            "split": split,
        },
    )
    return processed, normalize_chat_step(name=f"normalized-chat/{name}", download=processed)
