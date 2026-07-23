# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.datakit.download.stack_v3 import (
    HF_DATASET_ID,
    SERIALIZATION_FORMAT,
    row_to_doc,
)


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


def test_row_to_doc_serializes_all_files_in_depth_first_order_with_provenance():
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

    [doc] = row_to_doc(row)

    assert doc["text"] == (
        "<|repo_name|>marin-community/marin\n"
        "<|file_sep|>a/nested/x.py\nx\n"
        "<|file_sep|>a/z.py\nz\n"
        "<|file_sep|>a.py\nroot\n"
        "<|file_sep|>b.py\nb"
    )
    assert doc["source"] == HF_DATASET_ID
    assert doc["repo_path"] == row["repo_path"]
    assert doc["repo_id"] == row["repo_id"]
    assert doc["commit_id"] == row["commit_id"]
    assert doc["github_metadata"] == row["github_metadata"]
    assert doc["num_files"] == row["num_files"]
    assert doc["serialization_format"] == SERIALIZATION_FORMAT
    assert [metadata["file_path"] for metadata in doc["file_metadata"]] == [
        "a/nested/x.py",
        "a/z.py",
        "a.py",
        "b.py",
    ]
    assert [metadata["content_id"] for metadata in doc["file_metadata"]] == [
        "x-id",
        "z-id",
        "root-id",
        "b-id",
    ]
    assert all("content" not in metadata for metadata in doc["file_metadata"])
