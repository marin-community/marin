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

    [doc] = row_to_doc(row)

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
