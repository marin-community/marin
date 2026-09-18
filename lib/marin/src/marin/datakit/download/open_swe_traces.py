# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA Open-SWE-Traces as structured agent conversations."""

import json
from collections import deque
from types import MappingProxyType

import pyarrow as pa
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from marin.datakit.chat_normalize import CHAT_SCHEMA, normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import checked_openai_chat_document, load_parquet_batched
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "nvidia/Open-SWE-Traces"
HF_REVISION = "f967cba3312573981a47fd7a7b80029b53909b5f"
TRANSFORM_VERSION = "2026.09.17.chat-v1"
SOURCE_CHAT_SCHEMA = pa.schema([*CHAT_SCHEMA, pa.field("resolved", pa.int32())])

# Rough token weights use each release's uncompressed/compressed size ratio
# and four bytes per token; token-store preparation measures rendered sizes.
OPEN_SWE_TRACES_PARTITIONS = MappingProxyType(
    {
        "open_swe_traces/v1_0/openhands/minimax_m25/swe_rebench_v2": (
            "data/openhands/minimax_m25/swe-rebench-v2/*.parquet",
            3.043,
        ),
        "open_swe_traces/v1_0/openhands/qwen35_122b/swe_rebench_v2": (
            "data/openhands/qwen35_122b/swe-rebench-v2/*.parquet",
            3.101,
        ),
        "open_swe_traces/v1_0/sweagent/minimax_m25/swe_rebench_v2": (
            "data/sweagent/minimax_m25/swe-rebench-v2/*.parquet",
            2.906,
        ),
        "open_swe_traces/v1_0/sweagent/qwen35_122b/swe_rebench_v2": (
            "data/sweagent/qwen35_122b/swe-rebench-v2/*.parquet",
            1.579,
        ),
        "open_swe_traces/v1_1/openhands/deepseek_v4_flash/scale_swe": (
            "data/openhands/deepseek_v4_flash/scale-swe/*.parquet",
            2.098,
        ),
        "open_swe_traces/v1_1/openhands/qwen36_27b/scale_swe": ("data/openhands/qwen36_27b/scale-swe/*.parquet", 2.821),
        "open_swe_traces/v1_1/openhands/qwen36_27b/swe_rebench_v2": (
            "data/openhands/qwen36_27b/swe-rebench-v2/*.parquet",
            2.260,
        ),
        "open_swe_traces/v1_1/sweagent/qwen36_27b/scale_swe": ("data/sweagent/qwen36_27b/scale-swe/*.parquet", 2.765),
        "open_swe_traces/v1_1/sweagent/qwen36_27b/swe_rebench_v2": (
            "data/sweagent/qwen36_27b/swe-rebench-v2/*.parquet",
            2.472,
        ),
        "open_swe_traces/v1_1/minisweagent/qwen36_27b/scale_swe": (
            "data/minisweagent/qwen36_27b/scale-swe/*.parquet",
            1.555,
        ),
        "open_swe_traces/v1_1/minisweagent/qwen36_27b/swe_rebench_v2": (
            "data/minisweagent/qwen36_27b/swe-rebench-v2/*.parquet",
            1.763,
        ),
        "open_swe_traces/v1_2/minisweagent/qwen38_27b/scale_swe": (
            "data/minisweagent/qwen38_27b/scale-swe/*.parquet",
            3.397,
        ),
        "open_swe_traces/v1_2/minisweagent/qwen38_27b/swe_rebench_v2": (
            "data/minisweagent/qwen38_27b/swe-rebench-v2/*.parquet",
            3.571,
        ),
    }
)


def row_to_chat_doc(row: dict) -> list[dict]:
    messages = row.get("messages")
    if not messages:
        counters.pipeline.update_counter("open_swe_traces/empty_messages_filtered", 1)
        return []
    tools = [json.loads(tool) for tool in row["tools"]]
    # The export omits tool_call_id on observations, including parallel calls.
    # Observations follow their calls in order, so restore IDs before conversion.
    pending: deque[str] = deque()
    linked_messages: list[dict] = []
    for index, message in enumerate(messages):
        linked = dict(message)
        if message.get("role") == "assistant" and message.get("tool_calls"):
            calls = []
            for call_index, call in enumerate(message["tool_calls"]):
                call_id = f"call_{index}_{call_index}"
                calls.append({**call, "id": call_id})
                pending.append(call_id)
            linked["tool_calls"] = calls
        elif message.get("role") == "tool" and pending:
            linked["tool_call_id"] = pending.popleft()
        linked_messages.append(linked)
    return checked_openai_chat_document(
        linked_messages,
        HF_DATASET_ID,
        counter_prefix="open_swe_traces/chat",
        source_id=row["trajectory_id"],
        resolved=row["resolved"],
        chat_template_kwargs={"tools": tools},
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
    ZephyrContext(name="open-swe-traces-chat-transform", resources=ResourceConfig(cpu=1, ram="32g")).execute(pipeline)


def open_swe_traces_chat_normalize_steps(source_name: str) -> tuple[StepSpec, ...]:
    """Download one Open-SWE-Traces partition and normalize its structured chat."""
    glob, _ = OPEN_SWE_TRACES_PARTITIONS[source_name]
    raw = download_hf_step(f"raw/{source_name}", hf_dataset_id=HF_DATASET_ID, revision=HF_REVISION, hf_urls_glob=[glob])
    processed = StepSpec(
        name=f"processed-chat/{source_name}",
        deps=[raw],
        fn=lambda output_path: transform_chat(raw.output_path, output_path),
        hash_attrs={"version": TRANSFORM_VERSION},
    )
    return processed, normalize_chat_step(
        name=f"normalized-chat/{source_name}", download=processed, output_schema=SOURCE_CHAT_SCHEMA
    )
