# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenThoughts agent trajectories in Datakit's canonical chat format."""

from fray.types import ResourceConfig
from rigging.filesystem.storage_path import prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from marin.datakit.chat_normalize import CHAT_SCHEMA, normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import checked_openai_chat_document, load_parquet_batched
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "open-thoughts/OpenThoughts-Agent-SFT-100K"
HF_REVISION = "45fb28fcc38d352133cb28a1c8a43a2f14fea97b"
TRAIN_PARQUET_GLOB = "data/train-*.parquet"


def row_to_chat_doc(row: dict) -> list[dict]:
    """Convert one OpenThoughts trajectory into canonical Harmony messages."""
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return []
    source_id = row.get("run_id")
    return checked_openai_chat_document(
        conversations,
        HF_DATASET_ID,
        counter_prefix="openthoughts_agent/chat",
        source_id=source_id if isinstance(source_id, str) else None,
    )


def transform_chat(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(prefix_join(input_path, "**/*.parquet"))
        .flat_map(load_parquet_batched)
        .flat_map(row_to_chat_doc)
        .write_parquet(
            prefix_join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"),
            schema=CHAT_SCHEMA,
            skip_existing=True,
        )
    )
    ZephyrContext(
        name="openthoughts-agent-chat-transform",
        resources=ResourceConfig(cpu=1, ram="16g"),
    ).execute(pipeline)


def openthoughts_agent_chat_normalize_steps() -> tuple[StepSpec, ...]:
    download = download_hf_step(
        "raw/openthoughts-agent-sft-100k",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[TRAIN_PARQUET_GLOB],
    )
    processed = StepSpec(
        name="processed-chat/openthoughts-agent-sft-100k",
        deps=[download],
        fn=lambda output_path: transform_chat(download.output_path, output_path),
        hash_attrs={"version": "2026.09.22"},
    )
    return processed, normalize_chat_step(
        name="normalized-chat/openthoughts-agent-sft-100k",
        download=processed,
    )
