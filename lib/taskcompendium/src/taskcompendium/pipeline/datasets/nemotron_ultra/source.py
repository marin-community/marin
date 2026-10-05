# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable Ultra row readers and family policy construction."""

from functools import lru_cache
from typing import Any

from rigging.filesystem.storage_path import StoragePath
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_parquet

from taskcompendium.pipeline.datasets.nemotron_ultra.normalization import normalize
from taskcompendium.pipeline.models import ImportRejection, NormalizedTask, RawRow, ReviewRubric, TaskPipeline


@lru_cache(maxsize=1)
def swe_gym_ids(path: StoragePath) -> frozenset[str]:
    """Read membership IDs from a supplied SWE-Gym parquet file."""
    return frozenset(row["instance_id"] for row in load_parquet(str(path)))


@lru_cache(maxsize=1024)
def placeholder_record(path: StoragePath, index: int) -> dict[str, Any]:
    """Read one placeholder record from a supplied pinned parquet file."""
    records = list(load_parquet(InputFileSpec(path=str(path), row_start=index, row_end=index + 1)))
    if len(records) != 1:
        raise ValueError(f"Placeholder file {path} omitted row {index}")
    return records[0]


ACTION_COMPARISON_CRITERION = (
    "Only when agent_ref.name is single_step_tool_use_with_argument_comparison_agent, "
    "swe_pivot_single_step_tool_use_with_argument_comparison_agent, or "
    "toolcall_schema_single_step_tool_use_with_argument_comparison_agent at verifier revision "
    "d8b6e8c163def3660e9d3072c1c174226a1709fa, expected_action.type=message accepts any nonempty "
    "assistant text with no tool calls; the stored message is not a literal answer key. For "
    "expected_action.type=function_call, the scorer requires one call with the expected name and "
    "recursively matching argument keys and values, allowing floating-point tolerance 1e-6. These "
    "documented rules do not certify a bound runtime."
)


def quality_pipeline(family: str, selector: str, rubric_id: str, criteria: tuple[str, ...]) -> TaskPipeline:
    """Build an Ultra normalization and review policy from family criteria."""

    def normalize_row(row: RawRow) -> NormalizedTask | ImportRejection:
        return normalize(row, selector, family)

    return TaskPipeline(normalize=normalize_row, rubric=ReviewRubric(id=rubric_id, version="1", criteria=criteria))


def preference_pipeline(selector: str, rubric_id: str, criteria: tuple[str, ...]) -> TaskPipeline:
    """Keep generation-based GenRM prompts distinct from stored preference pairs."""
    return quality_pipeline("preference", selector, rubric_id, criteria)
