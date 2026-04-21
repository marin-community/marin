# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for HuggingFace download scripts."""

import json

import pytest

from marin.datakit.download.huggingface import (
    DownloadConfig,
    _relative_path_in_source,
    download_hf,
    stream_file_to_fsspec,
)


def _write(root, relative_path: str, content: bytes) -> None:
    full = root / relative_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(content)


def test_download_hf_basic(tmp_path):
    """End-to-end download against a local fsspec source (no mocks)."""
    source_root = tmp_path / "src"
    _write(source_root, "data/file1.txt", b"Content 1")
    _write(source_root, "data/file2.txt", b"Content 2")
    _write(source_root, "README.md", b"# Test Dataset")

    output_path = tmp_path / "output"
    output_path.mkdir()

    cfg = DownloadConfig(
        hf_dataset_id="test-org/test-dataset",
        revision="abc1234",
        gcs_output_path=str(output_path),
        source_url_override=str(source_root),
    )

    download_hf(cfg)

    assert (output_path / "data" / "file1.txt").read_bytes() == b"Content 1"
    assert (output_path / "data" / "file2.txt").read_bytes() == b"Content 2"
    assert (output_path / "README.md").read_bytes() == b"# Test Dataset"

    provenance = json.loads((output_path / "provenance.json").read_text())
    assert provenance["dataset"] == "test-org/test-dataset"
    assert provenance["version"] == "abc1234"
    assert "access_time" in provenance
    assert len(provenance["links"]) == 3


def test_download_hf_appends_sha_when_configured(tmp_path):
    """Outputs are written under a revision subdirectory when requested."""
    source_root = tmp_path / "src"
    _write(source_root, "data/file1.txt", b"Content 1")

    base_output_path = tmp_path / "output"
    revision = "abc1234"

    cfg = DownloadConfig(
        hf_dataset_id="test-org/test-dataset",
        revision=revision,
        gcs_output_path=str(base_output_path),
        append_sha_to_path=True,
        source_url_override=str(source_root),
    )

    download_hf(cfg)

    target_output = base_output_path / revision
    assert (target_output / "data" / "file1.txt").read_bytes() == b"Content 1"
    assert (target_output / "provenance.json").exists()


def test_relative_path_in_source_supports_bucket_paths():
    file_path = "hf://buckets/demo-user/demo-bucket/data/train/file1.txt"
    source_path = "hf://buckets/demo-user/demo-bucket/data"
    assert _relative_path_in_source(file_path, source_path) == "train/file1.txt"


def test_relative_path_in_source_supports_revision_qualified_bucket_paths():
    file_path = "hf://buckets/demo-user/demo-bucket/data@main/train/file1.txt"
    source_path = "hf://buckets/demo-user/demo-bucket/data"
    assert _relative_path_in_source(file_path, source_path) == "train/file1.txt"


def test_download_hf_bucket_requires_newer_huggingface_hub(tmp_path):
    cfg = DownloadConfig(
        hf_dataset_id="buckets/demo-user/demo-bucket/data",
        revision="main",
        gcs_output_path=str(tmp_path),
        hf_repo_type_prefix="",
    )

    with pytest.raises(RuntimeError, match=r"huggingface_hub>=1\.6\.0"):
        download_hf(cfg)


def test_stream_file_to_fsspec_reads_local_source(tmp_path):
    """stream_file_to_fsspec should copy an arbitrary fsspec URL to the destination."""
    source_file = tmp_path / "src" / "file1.txt"
    source_file.parent.mkdir(parents=True)
    content = b"hello world"
    source_file.write_bytes(content)

    output_path = tmp_path / "output"
    output_path.mkdir()
    destination = output_path / "data" / "file1.txt"

    result = stream_file_to_fsspec(
        str(output_path),
        str(source_file),
        str(destination),
        expected_size=len(content),
        read_timeout_seconds=1.0,
        progress_log_interval_seconds=0.0,
    )

    assert result["status"] == "success"
    assert destination.read_bytes() == content
