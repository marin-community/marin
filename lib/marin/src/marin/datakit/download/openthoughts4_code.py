# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.2 code reasoning responses from the OpenThoughts4 prompt set."""

import re

import pyarrow as pa
from fray.types import ResourceConfig
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from marin.datakit.chat_normalize import CHAT_SCHEMA, normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import (
    ReasoningFormatError,
    checked_openai_chat_document,
    load_parquet_batched,
    render_role_message,
    text_document,
)
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

SOURCE_CHAT_SCHEMA = pa.schema(
    [
        *CHAT_SCHEMA,
        pa.field("upstream_id", pa.string()),
        pa.field("prompt_index", pa.int64()),
        pa.field("response_index", pa.int64()),
    ]
)

HF_DATASET_ID = "marin-community/openthoughts4-code-9168-prompts-glm-5.2-n4"
HF_REVISION = "91f275562e041d798254122a7d50632e9d27badb"
TRAIN_PARQUET_GLOB = "data/train-*.parquet"


def _normalize_reasoning_tokens(text: str) -> str:
    """Normalize balanced reasoning tags to the tokenizer's atomic delimiters."""
    text = re.sub(r"<think>", "<|start_think|>", text, flags=re.IGNORECASE)
    text = re.sub(r"</think>", "<|end_think|>", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|start_think\|>\s*<\|end_think\|>\s*", "", text)
    depth = 0
    for match in re.finditer(r"<\|(start|end)_think\|>", text):
        if match.group(1) == "start":
            depth += 1
        else:
            depth -= 1
        if depth not in (0, 1):
            raise ReasoningFormatError("Assistant reasoning delimiters must be balanced and cannot nest")
    if depth != 0:
        raise ReasoningFormatError("Assistant reasoning delimiters must be balanced and cannot nest")
    return text.strip()


def _repair_reasoning_delimiters(message: dict) -> dict:
    """Repair GLM responses that use a lone ``</think>`` as the final-answer boundary."""
    repaired = dict(message)
    if repaired.get("role") != "assistant" or not isinstance(repaired.get("content"), str):
        return repaired

    content = repaired["content"]
    if "<think>" not in content and content.count("</think>") == 1:
        repaired["content"] = f"<think>{content}"
    return repaired


def row_to_chat_doc(row: dict) -> list[dict]:
    """Preserve one native message transcript in the canonical chat schema."""
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return []
    return checked_openai_chat_document(
        [_repair_reasoning_delimiters(message) for message in messages],
        HF_DATASET_ID,
        counter_prefix="openthoughts4_code/chat",
        prompt_index=row.get("prompt_index"),
        response_index=row.get("response_index"),
        upstream_id=str(row["source_id"]) if row.get("source_id") is not None else None,
    )


def row_to_doc(row: dict) -> list[dict]:
    """Render one native message transcript for ordinary Datakit consumers."""
    documents = row_to_chat_doc(row)
    if not documents:
        return []
    messages = [_repair_reasoning_delimiters(message) for message in row["messages"]]
    for message in messages:
        if message["role"] == "assistant":
            message["content"] = _normalize_reasoning_tokens(message["content"])
    text = "\n\n".join(render_role_message(message) for message in messages)
    return [text_document(text, HF_DATASET_ID)]


def _transform(input_path: str, output_path: str, *, chat: bool) -> None:
    transform_row = row_to_chat_doc if chat else row_to_doc
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet_batched)
        .flat_map(transform_row)
        .write_parquet(
            f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet",
            schema=SOURCE_CHAT_SCHEMA if chat else None,
            skip_existing=True,
        )
    )
    ZephyrContext(
        name="openthoughts4-code-chat" if chat else "openthoughts4-code", resources=ResourceConfig(cpu=1, ram="32g")
    ).execute(pipeline)


def _download_step() -> StepSpec:
    return download_hf_step(
        "raw/openthoughts4-code-glm-5.2-n4",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[TRAIN_PARQUET_GLOB],
    )


def openthoughts4_code_normalize_steps() -> tuple[StepSpec, ...]:
    download = _download_step()
    processed = StepSpec(
        name="processed/openthoughts4-code-glm-5.2-n4",
        deps=[download],
        fn=lambda output_path: _transform(download.output_path, output_path, chat=False),
        hash_attrs={"version": "2026.09.05.2"},
    )
    return processed, normalize_step(name="normalized/openthoughts4-code-glm-5.2-n4", download=processed)


def openthoughts4_code_chat_normalize_steps() -> tuple[StepSpec, ...]:
    download = _download_step()
    processed = StepSpec(
        name="processed-chat/openthoughts4-code-glm-5.2-n4",
        deps=[download],
        fn=lambda output_path: _transform(download.output_path, output_path, chat=True),
        hash_attrs={"version": "2026.09.09.upstream-id"},
    )
    return processed, normalize_chat_step(
        output_schema=SOURCE_CHAT_SCHEMA, name="normalized-chat/openthoughts4-code-glm-5.2-n4", download=processed
    )
