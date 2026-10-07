# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b dataset download and transform.

GPT-OSS-120B reasoning rollouts with chain-of-thought in <think> tags.
Each row has a math prompt and a model response with reasoning traces.
"""

from fray.types import ResourceConfig
from rigging.filesystem.storage_path import prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_jsonl

from marin.datakit.chat_normalize import CHAT_SCHEMA, normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import checked_openai_chat_document, text_document
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b"
HF_REVISION = "21b55a6"

STAGES = [
    "Superior-Reasoning-SFT-gpt-oss-120b-stage1-train-data.jsonl",
    "Superior-Reasoning-SFT-gpt-oss-120b-stage2-train-data.jsonl",
]


def row_to_doc(row: dict) -> list[dict]:
    prompt = row.get("input") or ""
    response = row.get("output") or ""
    if not prompt or not response:
        counters.pipeline.update_counter("superior_reasoning/dropped", 1)
        return []

    response = response.replace("<think>", "").replace("</think>", "").strip()
    if not response:
        counters.pipeline.update_counter("superior_reasoning/dropped", 1)
        return []

    text = f"<user>\n{prompt}\n</user>\n\n<assistant>\n{response}\n</assistant>"

    counters.pipeline.update_counter("superior_reasoning/kept", 1)
    return [text_document(text, "Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b")]


def row_to_chat_doc(row: dict) -> list[dict]:
    prompt = row.get("input") or ""
    response = row.get("output") or ""
    if not prompt or not response:
        return []
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    return checked_openai_chat_document(messages, HF_DATASET_ID, counter_prefix="superior_reasoning/chat")


def transform(input_path: str, output_path: str) -> None:
    input_files = [f"{input_path}/{stage}" for stage in STAGES]
    pipeline = (
        Dataset.from_list(input_files)
        .flat_map(load_jsonl)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="superior-reasoning-transform", resources=ResourceConfig(cpu=1, ram="8g"))
    ctx.execute(pipeline)


def transform_chat(input_path: str, output_path: str) -> None:
    input_files = [f"{input_path}/{stage}" for stage in STAGES]
    pipeline = (
        Dataset.from_list(input_files)
        .flat_map(load_jsonl)
        .flat_map(row_to_chat_doc)
        .write_parquet(
            prefix_join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"), schema=CHAT_SCHEMA, skip_existing=True
        )
    )
    ZephyrContext(name="superior-reasoning-chat-transform", resources=ResourceConfig(cpu=1, ram="8g")).execute(pipeline)


def download_superior_reasoning_step() -> StepSpec:
    """Download and transform Superior-Reasoning-SFT-gpt-oss-120b into JSONL documents."""
    dl = download_hf_step(
        "raw/superior-reasoning-sft",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=STAGES,
    )

    return StepSpec(
        name="processed/superior-reasoning-sft",
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v1"},
    )


def superior_reasoning_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for superior-reasoning."""
    processed = download_superior_reasoning_step()
    return (
        processed,
        normalize_step(name="normalized/superior-reasoning", download=processed),
    )


def superior_reasoning_chat_normalize_steps() -> tuple[StepSpec, ...]:
    download = download_hf_step(
        "raw/superior-reasoning-sft", hf_dataset_id=HF_DATASET_ID, revision=HF_REVISION, hf_urls_glob=STAGES
    )
    processed = StepSpec(
        name="processed-chat/superior-reasoning-sft",
        deps=[download],
        fn=lambda output_path: transform_chat(download.output_path, output_path),
        hash_attrs={"version": "2026.09.05.2.harmony-arrow"},
    )
    return processed, normalize_chat_step(
        output_schema=CHAT_SCHEMA, name="normalized-chat/superior-reasoning", download=processed
    )
