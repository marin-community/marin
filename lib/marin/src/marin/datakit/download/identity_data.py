# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""marin-community/identity-data download and normalization."""

from fray.types import ResourceConfig
from rigging.filesystem.storage_path import prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_parquet

from marin.datakit.chat_normalize import CHAT_SCHEMA, normalize_chat_step
from marin.datakit.download.hf_simple_util import hf_normalize_steps
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import checked_openai_chat_document
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "marin-community/identity-data"
HF_REVISION = "665ac6e"
MARIN_NAME = "identity-data/content"
CHAT_NAME = "identity-data"
DATA_GLOB = "data/train-*.parquet"


def identity_data_content_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the ``(download, normalize)`` chain for rendered identity conversations."""
    return hf_normalize_steps(
        marin_name=MARIN_NAME,
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=("data/train-*.parquet",),
        id_field="seed_id",
        text_field="content",
    )


def row_to_chat_doc(row: dict) -> list[dict]:
    """Use structured turns so identity answers are assistant targets."""
    return checked_openai_chat_document(
        row["messages"],
        HF_DATASET_ID,
        counter_prefix="identity_data/chat",
        source_id=row["seed_id"],
    )


def transform_chat(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(prefix_join(input_path, DATA_GLOB))
        .flat_map(load_parquet)
        .flat_map(row_to_chat_doc)
        .write_parquet(
            prefix_join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"),
            schema=CHAT_SCHEMA,
            skip_existing=True,
        )
    )
    ZephyrContext(name="identity-data-chat-transform", resources=ResourceConfig(cpu=1, ram="4g")).execute(pipeline)


def identity_data_chat_normalize_steps() -> tuple[StepSpec, ...]:
    download = download_hf_step(
        "raw/marin-community__identity-data",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[DATA_GLOB],
    )
    processed = StepSpec(
        name=f"processed-chat/{CHAT_NAME}",
        deps=[download],
        fn=lambda output_path: transform_chat(download.output_path, output_path),
        hash_attrs={"version": "2026.09.24"},
    )
    return processed, normalize_chat_step(
        name=f"normalized-chat/{CHAT_NAME}",
        download=processed,
    )
