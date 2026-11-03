# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA Open-SWE-Traces as structured agent conversations.

The two Qwen3.8-27B mini-swe-agent partitions get two fixes from the audit in marin-community/marin#10865. Their
task messages regain the Solution Originality section that the export dropped, and the traces on the pinned
``Min-Li/marin-sft-data-audit`` exclusion list are dropped. At the pinned source and list revisions, the exclusions
drop 3,562 of the 53,553 scale-swe rows and 5,052 of the 53,714 swe-rebench-v2 rows, all of them resolved. The
transform counts dropped rows in ``open_swe_traces/audit_exclusion_filtered``.
"""

import json
from collections import deque
from functools import partial
from types import MappingProxyType

import pyarrow as pa
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from marin.datakit.chat_normalize import CHAT_SCHEMA, normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import (
    checked_openai_chat_document,
    load_parquet_batched,
    pinned_jsonl_ids,
)
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "nvidia/Open-SWE-Traces"
HF_REVISION = "f967cba3312573981a47fd7a7b80029b53909b5f"
TRANSFORM_VERSION = "2026.11.03.chat-v2"
SOURCE_CHAT_SCHEMA = pa.schema([*CHAT_SCHEMA, pa.field("resolved", pa.int32())])
QWEN38_SCALE_SWE = "open_swe_traces/v1_2/minisweagent/qwen38_27b/scale_swe"
QWEN38_SWE_REBENCH_V2 = "open_swe_traces/v1_2/minisweagent/qwen38_27b/swe_rebench_v2"

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
        QWEN38_SCALE_SWE: ("data/minisweagent/qwen38_27b/scale-swe/*.parquet", 3.397),
        QWEN38_SWE_REBENCH_V2: ("data/minisweagent/qwen38_27b/swe-rebench-v2/*.parquet", 3.571),
    }
)

# The exclusion list holds the resolved Qwen3.8 traces with 8 or more assistant turns that reason about the
# hidden tests, the grader or the upstream fix, and 4 traces that read the upstream fix from the environment.
QWEN38_SOURCES = frozenset({QWEN38_SCALE_SWE, QWEN38_SWE_REBENCH_V2})
EXCLUSION_DATASET_ID = "Min-Li/marin-sft-data-audit"
EXCLUSION_REVISION = "5cc076cc822550f96287c861cf70c00c7ee8b4a6"
EXCLUSION_FILE = "open_swe/pinned_f967cba3/exclusions.jsonl"
EXCLUSION_LIST_SIZE = 8614

# The Qwen3.8-27B mini-swe-agent teacher was prompted with the NeMo-Skills template
# nemo_skills/prompt/config/eval/swe-bench/mini-swe-agent/swebench.yaml at commit
# 592ebaec5a80e17a3535623d0f5704d29e13a64f (Apache-2.0), which ends the task message with this
# section between </CRITICAL> and </instructions>. The export drops the section, although 32% of
# the successful traces quote it. Every row of both pinned partitions ends with the anchor below.
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


def repaired_qwen38_rows(row: dict, excluded_trajectory_ids: frozenset[str]) -> list[dict]:
    """Drop an excluded Qwen3.8 trace, or restore the Solution Originality section of a kept one."""
    if row["trajectory_id"] in excluded_trajectory_ids:
        counters.pipeline.update_counter("open_swe_traces/audit_exclusion_filtered", 1)
        return []
    return [restore_solution_originality(row)]


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


def source_rows(input_path: str) -> Dataset:
    return Dataset.from_files(prefix_join(input_path, "**/*.parquet")).flat_map(load_parquet_batched)


def write_chat(rows: Dataset, output_path: str) -> None:
    pipeline = rows.flat_map(row_to_chat_doc).write_parquet(
        prefix_join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"),
        schema=SOURCE_CHAT_SCHEMA,
        skip_existing=True,
    )
    ZephyrContext(name="open-swe-traces-chat-transform", resources=ResourceConfig(cpu=1, ram="32g")).execute(pipeline)


def transform_chat(input_path: str, output_path: str) -> None:
    write_chat(source_rows(input_path), output_path)


def transform_qwen38_chat(input_path: str, exclusion_path: str, output_path: str) -> None:
    trajectory_ids = pinned_jsonl_ids(prefix_join(exclusion_path, EXCLUSION_FILE), "trajectory_id", EXCLUSION_LIST_SIZE)
    rows = source_rows(input_path).flat_map(partial(repaired_qwen38_rows, excluded_trajectory_ids=trajectory_ids))
    write_chat(rows, output_path)


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
        processed = StepSpec(
            name=f"processed-chat/{source_name}",
            deps=[raw, exclusions],
            fn=lambda output_path: transform_qwen38_chat(raw.output_path, exclusions.output_path, output_path),
            hash_attrs={"version": TRANSFORM_VERSION, "exclusion_revision": EXCLUSION_REVISION},
        )
    else:
        processed = StepSpec(
            name=f"processed-chat/{source_name}",
            deps=[raw],
            fn=lambda output_path: transform_chat(raw.output_path, output_path),
            hash_attrs={"version": TRANSFORM_VERSION},
        )
    return processed, normalize_chat_step(
        name=f"normalized-chat/{source_name}", download=processed, output_schema=SOURCE_CHAT_SCHEMA
    )
