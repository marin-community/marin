# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Science conversations with Shellsim tool calls and observations."""

from fray.types import ResourceConfig
from rigging.filesystem.storage_path import prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from marin.datakit.chat_normalize import CHAT_SCHEMA, normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import checked_openai_chat_document, load_parquet_batched
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "open-athena/science-tool-use-conversations"
HF_REVISION = "332a9f0779717fcc07707ec42d0fc8cfb41fa64b"
DATA_FILE = "train.parquet"
CHAT_NAME = "science-tool-use-conversations"


def row_to_chat_doc(row: dict) -> list[dict]:
    """Keep the tool transcript and remove the unanswered closing user prompt."""
    full_log = row["full_log"]
    if not full_log or full_log[0]["role"] != "system":
        raise ValueError("Science tool-use conversations must start with a system prompt")
    # The source prompt asks for fenced bash, while full_log records function calls.
    # Marin's tool header supplies the matching function-call instructions.
    messages = full_log[1:]
    if messages and messages[-1]["role"] == "user":
        messages = messages[:-1]
    return checked_openai_chat_document(
        messages,
        HF_DATASET_ID,
        counter_prefix="science_tool_use/chat",
        source_id=str(row["idx"]),
        chat_template_kwargs={"tools": row["tools"]},
    )


def transform_chat(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(prefix_join(input_path, DATA_FILE))
        .flat_map(load_parquet_batched)
        .flat_map(row_to_chat_doc)
        .write_parquet(
            prefix_join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"),
            schema=CHAT_SCHEMA,
            skip_existing=True,
        )
    )
    ZephyrContext(name="science-tool-use-chat-transform", resources=ResourceConfig(cpu=1, ram="8g")).execute(pipeline)


def science_tool_use_chat_normalize_steps() -> tuple[StepSpec, ...]:
    download = download_hf_step(
        f"raw/{CHAT_NAME}",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[DATA_FILE],
    )
    processed = StepSpec(
        name=f"processed-chat/{CHAT_NAME}",
        deps=[download],
        fn=lambda output_path: transform_chat(download.output_path, output_path),
        hash_attrs={"version": "2026.09.28"},
    )
    return processed, normalize_chat_step(name=f"normalized-chat/{CHAT_NAME}", download=processed)
