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
"""

import pyarrow as pa
from fray.types import ResourceConfig
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
SOURCE_NAME = "calibforge-relay/terminus2"
TRANSFORM_VERSION = "2026.10.30.1"
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


def row_to_chat_doc(row: dict) -> list[dict]:
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


def transform_chat(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(prefix_join(input_path, "**/*.parquet"))
        .flat_map(load_parquet_batched)
        .flat_map(row_to_chat_doc)
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
    processed = StepSpec(
        name=f"processed-chat/{SOURCE_NAME}",
        deps=[raw],
        fn=lambda output_path: transform_chat(raw.output_path, output_path),
        hash_attrs={"version": TRANSFORM_VERSION, "think_tokens": THINK_TOKENS},
    )
    return processed, normalize_chat_step(
        name=f"normalized-chat/{SOURCE_NAME}", download=processed, output_schema=SOURCE_CHAT_SCHEMA
    )
