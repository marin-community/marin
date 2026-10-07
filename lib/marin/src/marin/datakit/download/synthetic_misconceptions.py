# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic conversations that correct common misconceptions."""

from fray.types import ResourceConfig
from rigging.filesystem.storage_path import prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_parquet

from marin.datakit.chat_normalize import CHAT_SCHEMA, normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import checked_openai_chat_document
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "open-athena/synthetic-misconceptions-conversations"
HF_REVISION = "aac8a4cb74bd999ffce39944058a136c27460cbb"
DATA_FILE = "train.parquet"


def row_to_chat_doc(row: dict) -> list[dict]:
    """Remove the source's closing user turn so the SFT record ends on an assistant target."""
    messages = row["messages"]
    if messages[-1]["role"] != "user":
        raise ValueError("Synthetic misconception conversations must end with a user turn")
    return checked_openai_chat_document(
        messages[:-1],
        HF_DATASET_ID,
        counter_prefix="synthetic_misconceptions/chat",
        source_id=f"{row['id']}:{row['opener_index']}",
    )


def transform_chat(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(prefix_join(input_path, DATA_FILE))
        .flat_map(load_parquet)
        .flat_map(row_to_chat_doc)
        .write_parquet(
            prefix_join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"),
            schema=CHAT_SCHEMA,
            skip_existing=True,
        )
    )
    ZephyrContext(
        name="synthetic-misconceptions-chat-transform",
        resources=ResourceConfig(cpu=1, ram="4g"),
    ).execute(pipeline)


def synthetic_misconceptions_chat_normalize_steps() -> tuple[StepSpec, ...]:
    download = download_hf_step(
        "raw/synthetic-misconceptions-conversations",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[DATA_FILE],
    )
    processed = StepSpec(
        name="processed-chat/synthetic-misconceptions-conversations",
        deps=[download],
        fn=lambda output_path: transform_chat(download.output_path, output_path),
        hash_attrs={"version": "2026.09.18"},
    )
    return processed, normalize_chat_step(
        name="normalized-chat/synthetic-misconceptions-conversations",
        download=processed,
    )
