# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize pre-staged Nemotron Code v2 file contents."""

from rigging.filesystem import StoragePath, prefix_join

from marin.datakit.normalize import DedupMode, normalize_step
from marin.execution.step_spec import StepSpec


def _validate_nemotron_code_v2_content(output_path: str) -> None:
    shards = StoragePath(prefix_join(output_path, "*.parquet")).glob()
    if next(iter(shards), None) is None:
        raise FileNotFoundError(f"No Parquet shards found under {output_path}")


def nemotron_code_v2_content_step() -> StepSpec:
    source_path = "raw/nemotron-code-v2-content"
    return StepSpec(
        name=source_path,
        override_output_path=source_path,
        fn=_validate_nemotron_code_v2_content,
        hash_attrs={
            "version": "2026.07.14",
            "format": "parquet",
            "columns": ["sha1_git", "sha1", "content", "present"],
            "rows": 132_903_245,
            "present_rows": 132_666_330,
            "shards": 133,
        },
    )


def nemotron_code_v2_content_normalize_steps() -> tuple[StepSpec, ...]:
    source = nemotron_code_v2_content_step()
    normalized = normalize_step(
        name="normalized/nemotron_code_v2_content",
        download=source,
        text_field="content",
        id_field="sha1_git",
        file_extensions=(".parquet",),
        dedup_mode=DedupMode.NONE,
        bare=True,
    )
    return source, normalized
