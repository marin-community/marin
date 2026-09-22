# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""UltraChat conversations extended through persona self-play."""

from collections.abc import Mapping
from functools import cache, partial

import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem.factory import open_url
from rigging.filesystem.storage_path import prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from marin.datakit.chat_normalize import CHAT_SCHEMA, normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import checked_openai_chat_document, load_parquet_batched
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "open-athena/ultrachat-persona-conversations"
HF_REVISION = "576ad1516d49b9837605a70908367aa076c5874a"
DATA_FILE = "train.parquet"
SOURCE_DATASET_ID = "HuggingFaceH4/ultrachat_200k"
SOURCE_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
SOURCE_FILES = (
    "data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
    "data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
    "data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet",
)


@cache
def _source_prompts(source_root: str) -> Mapping[str, str]:
    prompts: dict[str, str] = {}
    for filename in SOURCE_FILES:
        with open_url(prefix_join(source_root, filename), "rb") as stream:
            parquet = pq.ParquetFile(stream)
            for batch in parquet.iter_batches(batch_size=4096, columns=["prompt_id", "prompt"]):
                for row in batch.to_pylist():
                    prompt_id = row["prompt_id"]
                    prompt = row["prompt"]
                    if not isinstance(prompt_id, str) or not isinstance(prompt, str) or not prompt.strip():
                        counters.pipeline.update_counter("ultrachat_persona/invalid_source_prompt", 1)
                        continue
                    previous = prompts.setdefault(prompt_id, prompt)
                    if previous != prompt:
                        raise ValueError(f"UltraChat prompt ID maps to conflicting text: {prompt_id!r}")
    return prompts


def row_to_chat_doc(row: dict, prompts: Mapping[str, str]) -> list[dict]:
    """Restore the opening prompt and keep a conversation ending on an assistant target."""
    prompt_id = row["prompt_id"]
    if not isinstance(prompt_id, str) or prompt_id not in prompts:
        counters.pipeline.update_counter("ultrachat_persona/missing_source_prompt", 1)
        return []

    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) < 2 or row["turns"] != len(messages):
        raise ValueError("UltraChat persona turns must match a conversation with at least two messages")
    if messages[0] != {"role": "user", "content": None}:
        raise ValueError("UltraChat persona conversations must redact only their opening user prompt")

    restored = [{"role": "user", "content": prompts[prompt_id]}, *messages[1:]]
    if restored[-1].get("role") == "user":
        counters.pipeline.update_counter("ultrachat_persona/trailing_user_removed", 1)
        restored.pop()
    if len(restored) < 2 or restored[-1].get("role") != "assistant":
        raise ValueError("UltraChat persona conversations must contain an assistant target")

    return checked_openai_chat_document(
        restored,
        HF_DATASET_ID,
        counter_prefix="ultrachat_persona/chat",
        source_id=f"{prompt_id}:{row['persona_uuid']}",
    )


def _restore_row(row: dict, *, source_root: str) -> list[dict]:
    return row_to_chat_doc(row, _source_prompts(source_root))


def transform_chat(input_path: str, source_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(prefix_join(input_path, DATA_FILE))
        .flat_map(load_parquet_batched)
        .flat_map(partial(_restore_row, source_root=source_path))
        .write_parquet(
            prefix_join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"),
            schema=CHAT_SCHEMA,
            skip_existing=True,
        )
    )
    ZephyrContext(
        name="ultrachat-persona-chat-transform",
        resources=ResourceConfig(cpu=1, ram="8g"),
    ).execute(pipeline)


def ultrachat_persona_chat_normalize_steps() -> tuple[StepSpec, ...]:
    conversations = download_hf_step(
        "raw/ultrachat-persona-conversations",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[DATA_FILE],
    )
    source = download_hf_step(
        "raw/ultrachat-200k-train-sft",
        hf_dataset_id=SOURCE_DATASET_ID,
        revision=SOURCE_REVISION,
        hf_urls_glob=list(SOURCE_FILES),
    )
    processed = StepSpec(
        name="processed-chat/ultrachat-persona-conversations",
        deps=[conversations, source],
        fn=lambda output_path: transform_chat(conversations.output_path, source.output_path, output_path),
        hash_attrs={"version": "2026.09.19"},
    )
    return (
        conversations,
        source,
        processed,
        normalize_chat_step(
            name="normalized-chat/ultrachat-persona-conversations",
            download=processed,
        ),
    )
