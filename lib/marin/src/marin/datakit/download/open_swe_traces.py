# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA Open-SWE-Traces as structured agent conversations."""

import json
from collections import deque
from collections.abc import Callable
from types import MappingProxyType

import pyarrow as pa
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

# The exclusion list holds the resolved Qwen3.8 traces with 8 or more assistant turns that reason about the
# hidden tests, the grader or the upstream fix, and 4 traces that read the upstream fix from the environment.
QWEN38_SOURCES = frozenset(
    {
        "open_swe_traces/v1_2/minisweagent/qwen38_27b/scale_swe",
        "open_swe_traces/v1_2/minisweagent/qwen38_27b/swe_rebench_v2",
    }
)
EXCLUSION_DATASET_ID = "Min-Li/marin-sft-data-audit"
EXCLUSION_REVISION = "fb186cd8902777e24cb4c433398aaec9af0ecd6b"
EXCLUSION_FILE = "open_swe/pinned_f967cba3/exclusions.parquet"

# The Qwen3.8-27B mini-swe-agent teacher was prompted with the NeMo-Skills template
# nemo_skills/prompt/config/eval/swe-bench/mini-swe-agent/swebench.yaml at commit
# 592ebaec5a80e17a3535623d0f5704d29e13a64f (Apache-2.0), which ends the task message with this
# section between </CRITICAL> and </instructions>. The export drops it, and every row of both
# pinned partitions ends with the anchor below.
TASK_END_TAG = "</instructions>"
SOLUTION_ORIGINALITY_ANCHOR = "</CRITICAL>\n" + TASK_END_TAG
SOLUTION_ORIGINALITY_SECTION = "\n".join(
    (
        "## Solution Originality",
        "",
        "Implement your own original solution to the task.",
        "As an expert software engineer, your task is to create a novel independent solution based only on the "
        "current repository state.",
        "Your approach must be ENTIRELY UNBIASED, making no attempt to access external solutions, hints or "
        "information in any way.",
        "Remember that your own implementation is strongly preferred over a copy or modification of an existing "
        "solution.",
        "Because you are working independently, you **MUST NOT** attempt to:",
        "- clone, download files from, install as a package, read pull requests or otherwise access the upstream "
        "repository,",
        "- use git commands to access any commits, tags or references outside of the current branch,",
        "- read previous trajectories, golden patches, test patches or any other hidden information from the "
        "filesystem,",
        "- search locally installed packages or caches for upstream code,",
        "- find loopholes to access external solutions or hidden tests in any other way,",
        'for ANY reason at all, even if it\'s just "for reference" or to verify your own solution.',
        "WARNING: If you seek any form of external solutions, hints or information while solving the task,",
        "your solution will be DISCARDED regardless of its correctness!",
    )
)


def restore_solution_originality(row: dict) -> dict:
    """Return the row with the dropped Solution Originality section back in its task message."""
    messages = row["messages"]
    task = messages[1]["content"]
    if task.count(SOLUTION_ORIGINALITY_ANCHOR) != 1 or not task.endswith(SOLUTION_ORIGINALITY_ANCHOR):
        raise ValueError(f"Task message of {row['trajectory_id']} does not end with the template anchor")
    restored_task = task.removesuffix(TASK_END_TAG) + "\n" + SOLUTION_ORIGINALITY_SECTION + "\n" + TASK_END_TAG
    return {**row, "messages": [messages[0], {**messages[1], "content": restored_task}, *messages[2:]]}


def excluded_trajectory_ids(exclusion_path: str) -> frozenset[str]:
    """Load the pinned Qwen3.8 trace exclusions from the downloaded Parquet file."""
    with open_url(prefix_join(exclusion_path, EXCLUSION_FILE), "rb") as source:
        return frozenset(pq.read_table(source, columns=["trajectory_id"]).column("trajectory_id").to_pylist())


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


def transform_chat(
    input_path: str,
    output_path: str,
    *,
    excluded_ids: frozenset[str],
    repair_row: Callable[[dict], dict],
) -> None:
    pipeline = (
        Dataset.from_files(prefix_join(input_path, "**/*.parquet"))
        .flat_map(load_parquet_batched)
        .filter(lambda row: row["trajectory_id"] not in excluded_ids)
        .map(repair_row)
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
    if source_name in QWEN38_SOURCES:
        exclusions = download_hf_step(
            "raw/open-swe-traces-audit-exclusions",
            hf_dataset_id=EXCLUSION_DATASET_ID,
            revision=EXCLUSION_REVISION,
            hf_urls_glob=[EXCLUSION_FILE],
        )
        deps = [raw, exclusions]

        def transform(output_path: str) -> None:
            transform_chat(
                raw.output_path,
                output_path,
                excluded_ids=excluded_trajectory_ids(exclusions.output_path),
                repair_row=restore_solution_originality,
            )

    else:
        deps = [raw]

        def transform(output_path: str) -> None:
            transform_chat(raw.output_path, output_path, excluded_ids=frozenset(), repair_row=lambda row: row)

    processed = StepSpec(
        name=f"processed-chat/{source_name}", deps=deps, fn=transform, hash_attrs={"version": TRANSFORM_VERSION}
    )
    return processed, normalize_chat_step(
        name=f"normalized-chat/{source_name}", download=processed, output_schema=SOURCE_CHAT_SCHEMA
    )
