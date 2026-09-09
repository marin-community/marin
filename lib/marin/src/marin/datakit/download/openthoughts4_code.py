# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.2 code reasoning responses from the OpenThoughts4 prompt set."""

from fray.types import ResourceConfig
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from marin.datakit.chat_normalize import normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import (
    checked_chat_document,
    load_parquet_batched,
    render_role_message,
    text_document,
)
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "marin-community/openthoughts4-code-9168-prompts-glm-5.2-n4"
HF_REVISION = "91f275562e041d798254122a7d50632e9d27badb"
TRAIN_PARQUET_GLOB = "data/train-*.parquet"


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
    return checked_chat_document(
        [_repair_reasoning_delimiters(message) for message in messages],
        HF_DATASET_ID,
        counter_prefix="openthoughts4_code/chat",
        prompt_index=row.get("prompt_index"),
        response_index=row.get("response_index"),
        source_id=row.get("source_id"),
    )


def row_to_doc(row: dict) -> list[dict]:
    """Render one native message transcript for ordinary Datakit consumers."""
    documents = row_to_chat_doc(row)
    if not documents:
        return []
    text = "\n\n".join(render_role_message(message) for message in documents[0]["messages"])
    return [text_document(text, HF_DATASET_ID)]


def _transform(input_path: str, output_path: str, *, chat: bool) -> None:
    transform_row = row_to_chat_doc if chat else row_to_doc
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet_batched)
        .flat_map(transform_row)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
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
        hash_attrs={"version": "2026.09.05.2.harmony"},
    )
    return processed, normalize_chat_step(name="normalized-chat/openthoughts4-code-glm-5.2-n4", download=processed)
