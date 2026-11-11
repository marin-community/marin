# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Terminus-2 relay SFT traces from laion/calibforge-relay-traces.

In a relay episode a student model (Grug Datakit 09-21, or an SFT checkpoint trained from it) works a CalibForge
terminal task until a hand-over trigger fires, and Qwen3.8-27B plays every later turn; only the teacher's turns are
trained. The ``terminus2_relay_sft`` config is the full training set of the best Terminus-2 recipe: 5,916 relay
episodes plus 4,392 Kimi-2.5 SWE-smith Terminus-2 traces, every assistant turn of the Kimi traces trained.

After dropping the template-only system message, ``source_train_turns`` preserves one ``trained`` flag per source
message. These flags are provenance, not an output-message loss mask: reasoning and command text can become separate
messages, and neither the renderer nor the trainer applies the flags yet.

CalibForge tasks that are variants of Terminal-Bench 2.1 or TB-Lite test tasks are excluded using the pinned
``ricdomolm/CalibForge-decontaminated`` list. At the pinned source and list revisions, this drops 229 of the 5,916
CalibForge relay episodes. The transform also counts dropped episodes in
``calibforge_relay/chat/benchmark_task_filtered``.
"""

import json
from functools import partial

import pyarrow as pa
from fray.types import ResourceConfig
from rigging.filesystem.factory import open_url
from rigging.filesystem.storage_path import prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from marin.datakit.chat_normalize import CHAT_SCHEMA, normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import (
    TRAJECTORY_FAILED_TAG,
    TRAJECTORY_SOLVED_TAG,
    checked_openai_chat_document,
    load_parquet_batched,
)
from marin.datakit.download.terminus import TerminusParseMode, ThinkTokens, terminus_protocol_messages
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "laion/calibforge-relay-traces"
HF_REVISION = "66feab4d24bfc37af4c38d95a6a26187b5648308"
EXCLUSION_DATASET_ID = "ricdomolm/CalibForge-decontaminated"
EXCLUSION_REVISION = "a33f23c4f489d7a543a28fd2038f78cd1eb2c38a"
EXCLUSION_LIST_SIZE = 405
EXCLUDED_SOURCE_ROWS = 229
SOURCE_NAME = "calibforge-relay/terminus2"
TRANSFORM_VERSION = "2026.11.02.1"
THINK_TOKENS = ThinkTokens("<|start_think|>", "<|end_think|>")
COUNTER_PREFIX = "calibforge_relay/chat"
SOURCE_CHAT_SCHEMA = pa.schema(
    [
        *CHAT_SCHEMA,
        pa.field("source_train_turns", pa.list_(pa.bool_())),
        pa.field("teacher", pa.string()),
        pa.field("task_source", pa.string()),
        pa.field("outcome", pa.string()),
    ]
)


def excluded_task_ids(exclusion_path: str) -> frozenset[str]:
    """Load the pinned CalibForge task exclusions from the downloaded JSONL file."""
    with open_url(prefix_join(exclusion_path, "excluded_tasks.jsonl"), "rt") as source:
        task_ids = frozenset(json.loads(line)["task_id"] for line in source)
    if len(task_ids) != EXCLUSION_LIST_SIZE:
        raise ValueError(f"Expected {EXCLUSION_LIST_SIZE} excluded CalibForge tasks, found {len(task_ids)}")
    return task_ids


def row_to_chat_doc(row: dict, excluded_task_ids: frozenset[str]) -> list[dict]:
    task = row["task"]
    if task.startswith("calibforge__") and task.split("__", 2)[2] in excluded_task_ids:
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/benchmark_task_filtered", 1)
        return []
    source_messages = row.get("messages")
    if not source_messages:
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/empty_filtered", 1)
        return []
    conversation: list[dict] = []
    source_train_turns: list[bool] = []
    for message in source_messages:
        # The only system message is the student template's "Reasoning: /think" line, not part of the task.
        if message["role"] == "system":
            continue
        conversation.append({"role": message["role"], "content": message["content"]})
        source_train_turns.append(bool(message["trained"]))
    messages = terminus_protocol_messages(conversation, THINK_TOKENS, mode=TerminusParseMode.REPAIR)
    if messages is None:
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/terminus_protocol_filtered", 1)
        return []
    return checked_openai_chat_document(
        messages,
        HF_DATASET_ID,
        counter_prefix=COUNTER_PREFIX,
        source_id=row["sid"],
        source_train_turns=source_train_turns,
        teacher=row["teacher"],
        task_source=row["source"],
        outcome=TRAJECTORY_SOLVED_TAG if row["passed"] else TRAJECTORY_FAILED_TAG,
    )


def transform_chat(input_path: str, exclusion_path: str, output_path: str) -> None:
    task_ids = excluded_task_ids(exclusion_path)
    pipeline = (
        Dataset.from_files(prefix_join(input_path, "**/*.parquet"))
        .flat_map(load_parquet_batched)
        .flat_map(partial(row_to_chat_doc, excluded_task_ids=task_ids))
        .write_parquet(
            prefix_join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"),
            schema=SOURCE_CHAT_SCHEMA,
            skip_existing=True,
        )
    )
    ZephyrContext(name="calibforge-relay-chat-transform", resources=ResourceConfig(cpu=1, ram="16g")).execute(pipeline)


def calibforge_relay_chat_normalize_steps() -> tuple[StepSpec, ...]:
    """Download the Terminus-2 relay SFT set and normalize its structured chat."""
    raw = download_hf_step(
        f"raw/{SOURCE_NAME}",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=["data/terminus2_relay_sft/*.parquet"],
    )
    exclusions = download_hf_step(
        "raw/calibforge-benchmark-exclusions",
        hf_dataset_id=EXCLUSION_DATASET_ID,
        revision=EXCLUSION_REVISION,
        hf_urls_glob=["excluded_tasks.jsonl"],
    )
    processed = StepSpec(
        name=f"processed-chat/{SOURCE_NAME}",
        deps=[raw, exclusions],
        fn=lambda output_path: transform_chat(raw.output_path, exclusions.output_path, output_path),
        hash_attrs={
            "version": TRANSFORM_VERSION,
            "think_tokens": THINK_TOKENS,
            "exclusion_revision": EXCLUSION_REVISION,
            "excluded_source_rows": EXCLUDED_SOURCE_ROWS,
        },
    )
    return processed, normalize_chat_step(
        name=f"normalized-chat/{SOURCE_NAME}", download=processed, output_schema=SOURCE_CHAT_SCHEMA
    )
