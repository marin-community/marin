# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HuggingFaceCode/stack-v3-train download and repository-context transform."""

from pathlib import PurePosixPath
from typing import TypedDict

import pyarrow as pa
from fray.types import ResourceConfig
from rigging.filesystem import prefix_join
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.readers import load_parquet

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "HuggingFaceCode/stack-v3-train"
HF_REVISION = "bb2fa95033c00931906761bed7bc37b525155db6"
TRAIN_PARQUET_GLOB = "data/*.parquet"

REPOSITORY_HEADER = "Repository:"
FILE_HEADER = "File:"
SERIALIZATION_FORMAT = "natural_headers_directory_block_dfs_v1"

OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("text", pa.string()),
        pa.field("source", pa.string()),
        pa.field("repo_path", pa.string()),
        pa.field("repo_id", pa.int64()),
        pa.field("commit_id", pa.string()),
        pa.field(
            "github_metadata",
            pa.struct(
                [
                    pa.field("branch", pa.string()),
                    pa.field("commit_count", pa.int64()),
                    pa.field("forked_from", pa.string()),
                    pa.field("forks", pa.int64()),
                    pa.field("is_fork", pa.bool_()),
                    pa.field("is_org_owned", pa.bool_()),
                    pa.field("issues", pa.int64()),
                    pa.field("pull_requests", pa.int64()),
                    pa.field("repo_created_at", pa.string()),
                    pa.field("stars", pa.int64()),
                ]
            ),
        ),
        pa.field("num_files", pa.int64()),
        pa.field(
            "file_metadata",
            pa.list_(
                pa.struct(
                    [
                        pa.field("content_id", pa.string()),
                        pa.field("detected_licenses", pa.list_(pa.string())),
                        pa.field("file_path", pa.string()),
                        pa.field("file_timestamp", pa.int64()),
                        pa.field("is_vendor", pa.bool_()),
                        pa.field("language", pa.string()),
                        pa.field("license_type", pa.string()),
                        pa.field("size_bytes", pa.int64()),
                    ]
                )
            ),
        ),
        pa.field("serialization_format", pa.string()),
    ]
)


class StackV3File(TypedDict):
    content_id: str
    content: str
    detected_licenses: list[str]
    file_path: str
    file_timestamp: int
    is_vendor: bool
    language: str
    license_type: str
    size_bytes: int


def _directory_block_dfs(files: list[StackV3File]) -> list[StackV3File]:
    files_by_directory: dict[tuple[str, ...], list[StackV3File]] = {}
    child_directories: dict[tuple[str, ...], set[tuple[str, ...]]] = {}
    for file in files:
        *directory_parts, _ = PurePosixPath(file["file_path"]).parts
        directory = tuple(directory_parts)
        files_by_directory.setdefault(directory, []).append(file)
        for depth in range(len(directory)):
            parent = directory[:depth]
            child_directories.setdefault(parent, set()).add(directory[: depth + 1])

    ordered_files = []
    pending_directories = [()]
    while pending_directories:
        directory = pending_directories.pop()
        ordered_files.extend(
            sorted(files_by_directory.get(directory, []), key=lambda file: PurePosixPath(file["file_path"]).name)
        )
        pending_directories.extend(sorted(child_directories.get(directory, ()), reverse=True))
    return ordered_files


def row_to_doc(row: dict) -> list[dict]:
    files = _directory_block_dfs(row["files"])
    sections = [f"{REPOSITORY_HEADER} {row['repo_path']}"]
    sections.extend(f"{FILE_HEADER} {file['file_path']}\n{file['content']}" for file in files)

    return [
        {
            "text": "\n\n".join(sections),
            "source": HF_DATASET_ID,
            "repo_path": row["repo_path"],
            "repo_id": row["repo_id"],
            "commit_id": row["commit_id"],
            "github_metadata": row["github_metadata"],
            "num_files": row["num_files"],
            "file_metadata": [{key: value for key, value in file.items() if key != "content"} for file in files],
            "serialization_format": SERIALIZATION_FORMAT,
        }
    ]


def transform(input_path: str, output_path: str) -> None:
    """Serialize repositories with each directory's files kept together."""
    pipeline = (
        Dataset.from_files(prefix_join(input_path, TRAIN_PARQUET_GLOB))
        .flat_map(load_parquet)
        .flat_map(row_to_doc)
        .write_parquet(
            prefix_join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"),
            schema=OUTPUT_SCHEMA,
            skip_existing=True,
        )
    )
    ctx = ZephyrContext(name="stack-v3-transform", resources=ResourceConfig(cpu=2, ram="32g", disk="10g"))
    ctx.execute(pipeline)


def processed_stack_v3_step() -> StepSpec:
    """Download and serialize Stack v3 repositories."""
    download = download_hf_step(
        "raw/stack-v3",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[TRAIN_PARQUET_GLOB],
    )
    return StepSpec(
        name="processed/stack-v3",
        deps=[download],
        fn=lambda output_path: transform(input_path=download.output_path, output_path=output_path),
        hash_attrs={
            "serialization_format": SERIALIZATION_FORMAT,
            "repository_header": REPOSITORY_HEADER,
            "file_header": FILE_HEADER,
            "output_schema": str(OUTPUT_SCHEMA),
        },
    )


def stack_v3_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the Stack v3 processed-repository and normalize chain."""
    processed = processed_stack_v3_step()
    return (
        processed,
        normalize_step(name="normalized/stack-v3", download=processed, id_field="repo_id"),
    )
