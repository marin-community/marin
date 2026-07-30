# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.download.stack_v3 import (
    HF_DATASET_ID,
    NORMALIZED_OUTPUT_SCHEMA,
    SERIALIZATION_FORMAT,
    row_to_doc,
    transform,
)
from marin.datakit.normalize import normalize_to_parquet


def _file(path: str, content: str, content_id: str) -> dict:
    return {
        "content_id": content_id,
        "content": content,
        "size_bytes": len(content.encode()),
        "file_path": path,
        "file_timestamp": 1_700_000_000,
        "language": "Python",
        "is_vendor": False,
        "license_type": "permissive",
        "detected_licenses": ["MIT"],
    }


def test_row_to_doc_keeps_same_directory_files_together_in_depth_first_order():
    row = {
        "repo_path": "marin-community/marin",
        "repo_id": 123,
        "commit_id": "abc123",
        "github_metadata": {"branch": "main", "stars": 42},
        "num_files": 4,
        "files": [
            _file("a.py", "root", "root-id"),
            _file("a/z.py", "z", "z-id"),
            _file("b.py", "b", "b-id"),
            _file("a/nested/x.py", "x", "x-id"),
        ],
    }

    doc = row_to_doc(row)

    assert doc["text"] == (
        "Repository: marin-community/marin\n\n"
        "File: a.py\nroot\n\n"
        "File: b.py\nb\n\n"
        "File: a/z.py\nz\n\n"
        "File: a/nested/x.py\nx"
    )
    assert doc["source"] == HF_DATASET_ID
    assert doc["repo_path"] == row["repo_path"]
    assert doc["repo_id"] == row["repo_id"]
    assert doc["commit_id"] == row["commit_id"]
    assert doc["github_metadata"] == row["github_metadata"]
    assert doc["num_files"] == row["num_files"]
    assert doc["serialization_format"] == SERIALIZATION_FORMAT
    assert [metadata["file_path"] for metadata in doc["file_metadata"]] == [
        "a.py",
        "b.py",
        "a/z.py",
        "a/nested/x.py",
    ]
    assert [metadata["content_id"] for metadata in doc["file_metadata"]] == [
        "root-id",
        "b-id",
        "z-id",
        "x-id",
    ]
    assert all("content" not in metadata for metadata in doc["file_metadata"])


def test_row_to_doc_serializes_empty_file_path():
    row = {
        "repo_path": "marin-community/marin",
        "repo_id": 123,
        "commit_id": "abc123",
        "github_metadata": {"branch": "main", "stars": 42},
        "num_files": 1,
        "files": [_file("", "orphaned content", "orphaned-id")],
    }

    doc = row_to_doc(row)

    assert doc["text"] == "Repository: marin-community/marin\n\nFile: (unknown path)\norphaned content"
    assert doc["file_metadata"][0]["file_path"] == ""


def test_transform_writes_nullable_metadata_with_stable_schema(tmp_path: Path):
    raw_dir = tmp_path / "raw" / "data"
    raw_dir.mkdir(parents=True)
    row = {
        "repo_path": "marin-community/marin",
        "repo_id": 123,
        "commit_id": "abc123",
        "github_metadata": {
            "branch": "main",
            "commit_count": 1,
            "forked_from": None,
            "forks": 2,
            "is_fork": False,
            "is_org_owned": True,
            "issues": 3,
            "pull_requests": 4,
            "repo_created_at": "2025-01-01T00:00:00Z",
            "stars": 42,
        },
        "num_files": 1,
        "files": [_file("src/main.py", "print('hello')", "main-id")],
    }
    pq.write_table(pa.Table.from_pylist([row]), raw_dir / "data.parquet")

    output_dir = tmp_path / "processed"
    transform(str(tmp_path / "raw"), str(output_dir))

    [output_path] = output_dir.glob("*.parquet")
    output_schema = pq.read_schema(output_path)
    github_metadata = output_schema.field("github_metadata").type
    assert github_metadata.field("forked_from").type == pa.string()
    assert pq.read_table(output_path).to_pylist()[0]["github_metadata"]["forked_from"] is None

    normalized_dir = tmp_path / "normalized"
    normalize_to_parquet(
        input_path=str(output_dir),
        output_path=str(normalized_dir),
        id_field="repo_id",
        output_schema=NORMALIZED_OUTPUT_SCHEMA,
    )

    [normalized_path] = (normalized_dir / "outputs" / "main").glob("*.parquet")
    normalized_schema = pq.read_schema(normalized_path)
    normalized_github_metadata = normalized_schema.field("github_metadata").type
    assert normalized_github_metadata.field("forked_from").type == pa.string()
    assert pq.read_table(normalized_path).to_pylist()[0]["github_metadata"]["forked_from"] is None


def test_duplicate_file_entries_are_serialized_once():
    """A Stack v3 row lists each file many times; the serialized document carries it once."""
    row = {
        "repo_path": "org/repo",
        "repo_id": 1,
        "commit_id": "c",
        "github_metadata": {"branch": "main", "stars": 1},
        "num_files": 20,
        "files": (
            [_file("a.py", "alpha", "id-a") for _ in range(10)] + [_file("b.py", "beta", "id-b") for _ in range(10)]
        ),
    }

    doc = row_to_doc(row)

    assert doc["text"] == ("Repository: org/repo\n\nFile: a.py\nalpha\n\nFile: b.py\nbeta")
    assert doc["num_files"] == 2
    assert [entry["file_path"] for entry in doc["file_metadata"]] == ["a.py", "b.py"]


def test_same_content_at_two_paths_is_kept():
    """Two paths sharing one content_id are distinct files and both survive."""
    row = {
        "repo_path": "org/repo",
        "repo_id": 1,
        "commit_id": "c",
        "github_metadata": {"branch": "main", "stars": 1},
        "num_files": 2,
        "files": [_file("pkg/__init__.py", "", "empty-id"), _file("pkg/sub/__init__.py", "", "empty-id")],
    }

    doc = row_to_doc(row)

    assert doc["num_files"] == 2
    assert [entry["file_path"] for entry in doc["file_metadata"]] == ["pkg/__init__.py", "pkg/sub/__init__.py"]
