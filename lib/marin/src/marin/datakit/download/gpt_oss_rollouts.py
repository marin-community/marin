# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""andyrdt/gpt-oss-20b-rollouts dataset download and transform.

GPT-OSS-20B rollouts with parsed reasoning chains. We include only the
non-benchmark subsets: NuminaMath-CoT, WildChat-1M, and ultrachat_200k.

Each row has a user prompt, the model's internal thinking, and the final
assistant response. We render these into a single document.
"""

from fray.types import ResourceConfig
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_jsonl

from marin.datakit.chat_normalize import normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.opencode import INLINE_TOOL_CALL
from marin.datakit.download.rollout_transforms import (
    CHAT_CONTROL_TOKEN,
    ReasoningFormatError,
    openai_chat_document,
    text_document,
)
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "andyrdt/gpt-oss-20b-rollouts"
HF_REVISION = "f47b4a2"

SUBSETS = {
    "NuminaMath-CoT": "train",
    "WildChat-1M": "train",
    "ultrachat_200k": "train_sft",
}


def row_to_doc(row: dict) -> list[dict]:
    user = row.get("user_content") or ""
    thinking = row.get("assistant_thinking") or ""
    response = row.get("assistant_content") or ""
    if not user or not response:
        counters.pipeline.update_counter("gpt_oss_rollouts/dropped", 1)
        return []

    parts = [f"<user>\n{user}\n</user>"]
    if thinking:
        parts.append(f"<thinking>\n{thinking}\n</thinking>")
    parts.append(f"<assistant>\n{response}\n</assistant>")

    text = "\n\n".join(parts)

    counters.pipeline.update_counter("gpt_oss_rollouts/kept", 1)
    return [text_document(text, "andyrdt/gpt-oss-20b-rollouts")]


def row_to_chat_doc(row: dict) -> list[dict]:
    user = row.get("user_content") or ""
    response = row.get("assistant_content") or ""
    if not user or not response:
        return []
    thinking = row.get("assistant_thinking") or ""
    if any(CHAT_CONTROL_TOKEN.search(text) for text in (user, thinking, response)):
        counters.pipeline.update_counter("gpt_oss_rollouts/chat_control_token_filtered", 1)
        return []
    if thinking and any(token in thinking for token in ("<think>", "<|start_think|>")):
        assistant = f"{thinking}\n\n{response}"
    elif thinking:
        assistant = f"<think>\n{thinking}\n</think>\n\n{response}"
    else:
        assistant = response
    if INLINE_TOOL_CALL.search(assistant):
        counters.pipeline.update_counter("gpt_oss_rollouts/chat_inline_tool_syntax_filtered", 1)
        return []
    messages = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    try:
        return [openai_chat_document(messages, HF_DATASET_ID)]
    except ReasoningFormatError:
        counters.pipeline.update_counter("gpt_oss_rollouts/chat_malformed_reasoning_filtered", 1)
        return []


def transform(input_path: str, output_path: str) -> None:
    input_files = [f"{input_path}/{subset}/{split}.jsonl" for subset, split in SUBSETS.items()]
    pipeline = (
        Dataset.from_list(input_files)
        .flat_map(load_jsonl)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="gpt-oss-rollouts-transform", resources=ResourceConfig(cpu=1, ram="8g"))
    ctx.execute(pipeline)


def transform_chat(input_path: str, output_path: str) -> None:
    input_files = [f"{input_path}/{subset}/{split}.jsonl" for subset, split in SUBSETS.items()]
    pipeline = (
        Dataset.from_list(input_files)
        .flat_map(load_jsonl)
        .flat_map(row_to_chat_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ZephyrContext(name="gpt-oss-rollouts-chat-transform", resources=ResourceConfig(cpu=1, ram="8g")).execute(pipeline)


def download_gpt_oss_rollouts_step() -> StepSpec:
    """Download and transform non-benchmark GPT-OSS-20B rollouts into JSONL documents."""
    hf_urls_glob = []
    for subset, split in SUBSETS.items():
        hf_urls_glob.append(f"{subset}/{split}.jsonl")

    dl = download_hf_step(
        "raw/gpt-oss-20b-rollouts",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=hf_urls_glob,
    )

    return StepSpec(
        name="processed/gpt-oss-20b-rollouts",
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v2"},
    )


def gpt_oss_rollouts_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for gpt-oss-rollouts."""
    processed = download_gpt_oss_rollouts_step()
    return (
        processed,
        normalize_step(name="normalized/gpt-oss-rollouts", download=processed),
    )


def gpt_oss_rollouts_chat_normalize_steps() -> tuple[StepSpec, ...]:
    paths = [f"{subset}/{split}.jsonl" for subset, split in SUBSETS.items()]
    download = download_hf_step(
        "raw/gpt-oss-20b-rollouts", hf_dataset_id=HF_DATASET_ID, revision=HF_REVISION, hf_urls_glob=paths
    )
    processed = StepSpec(
        name="processed-chat/gpt-oss-20b-rollouts",
        deps=[download],
        fn=lambda output_path: transform_chat(download.output_path, output_path),
        hash_attrs={"version": "2026.09.04.3.harmony-direct"},
    )
    return processed, normalize_chat_step(name="normalized-chat/gpt-oss-rollouts", download=processed)
