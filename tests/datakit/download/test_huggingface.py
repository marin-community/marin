# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for HuggingFace download scripts."""

import io
import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from marin.datakit.download.huggingface_utils import (
    DownloadConfig,
    _relative_path_in_source,
    download_hf,
    stream_file_to_fsspec,
)


@pytest.fixture
def mock_hf_fs():
    """Create a mock HuggingFace filesystem with test files."""

    def _create(files: dict[str, bytes] | None = None) -> MagicMock:
        """Create mock HfFileSystem.

        Args:
            files: Dict mapping file paths to file contents

        Returns:
            Mock HfFileSystem
        """
        if files is None:
            files = {}

        fs = MagicMock()

        def mock_open(path, mode="rb", **_kwargs):
            if path in files:
                return io.BytesIO(files[path])
            raise FileNotFoundError(f"File not found: {path}")

        fs.open.side_effect = mock_open
        fs.exists = Mock(side_effect=lambda p: p in files)
        fs.find = Mock(return_value=list(files.keys()))
        fs.glob = Mock(
            side_effect=lambda pattern, revision=None: [
                f for f in files.keys() if "*" not in pattern or pattern.split("*")[0] in f
            ]
        )
        fs.ls = Mock(return_value=list(files.keys()))
        fs.info = Mock(side_effect=lambda path, revision=None: {"size": len(files.get(path, b""))})

        return fs

    return _create


def test_download_hf_basic(mock_hf_fs, tmp_path):
    """Test basic HF download functionality."""
    test_files = {
        "datasets/test-org/test-dataset/data/file1.txt": b"Content 1",
        "datasets/test-org/test-dataset/data/file2.txt": b"Content 2",
        "datasets/test-org/test-dataset/README.md": b"# Test Dataset",
    }

    hf_fs = mock_hf_fs(test_files)

    output_path = tmp_path / "output"
    output_path.mkdir()

    cfg = DownloadConfig(
        hf_dataset_id="test-org/test-dataset",
        revision="abc1234",
        gcs_output_path=str(output_path),
    )

    # Mock HfFileSystem creation
    with patch("marin.datakit.download.huggingface_utils.HfFileSystem", return_value=hf_fs):
        download_hf(cfg)

    # Verify files were downloaded
    assert (output_path / "data" / "file1.txt").exists()
    assert (output_path / "data" / "file2.txt").exists()
    assert (output_path / "README.md").exists()

    # Verify content
    assert (output_path / "data" / "file1.txt").read_bytes() == b"Content 1"
    assert (output_path / "data" / "file2.txt").read_bytes() == b"Content 2"

    # Verify provenance file was created
    provenance_file = output_path / "provenance.json"
    assert provenance_file.exists()

    provenance = json.loads(provenance_file.read_text())
    assert provenance["dataset"] == "test-org/test-dataset"
    assert provenance["version"] == "abc1234"
    assert "access_time" in provenance
    assert len(provenance["links"]) == 3


def test_download_hf_appends_sha_when_configured(mock_hf_fs, tmp_path):
    """Ensure outputs are written under a revision subdirectory when requested."""

    test_files = {
        "datasets/test-org/test-dataset/data/file1.txt": b"Content 1",
    }

    hf_fs = mock_hf_fs(test_files)

    base_output_path = tmp_path / "output"
    revision = "abc1234"

    cfg = DownloadConfig(
        hf_dataset_id="test-org/test-dataset",
        revision=revision,
        gcs_output_path=str(base_output_path),
        append_sha_to_path=True,
    )

    with patch("marin.datakit.download.huggingface_utils.HfFileSystem", return_value=hf_fs):
        download_hf(cfg)

    target_output = base_output_path / revision
    assert (target_output / "data" / "file1.txt").exists()
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


def test_stream_file_to_fsspec_retries_on_timeout(tmp_path):
    """A socket timeout while reading should trigger retry and then succeed."""
    file_path = "datasets/test-org/test-dataset/data/file1.txt"
    output_path = tmp_path / "output"
    output_path.mkdir()
    destination = output_path / "data" / "file1.txt"

    content = b"retry me"

    hf_fs = MagicMock()
    read_attempts = {"count": 0}

    class FlakyReader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, chunk_size):
            read_attempts["count"] += 1
            if read_attempts["count"] == 1:
                raise TimeoutError("simulated timeout")
            if read_attempts["count"] == 2:
                return content
            return b""

    hf_fs.open.side_effect = lambda path, mode="rb", **_kwargs: FlakyReader()

    with (
        patch("marin.datakit.download.huggingface_utils.HfFileSystem", return_value=hf_fs),
        patch("marin.datakit.download.huggingface_utils.time.sleep", return_value=None),
    ):
        result = stream_file_to_fsspec(
            str(output_path),
            file_path,
            str(destination),
            expected_size=len(content),
            read_timeout_seconds=1.0,
            progress_log_interval_seconds=0.0,
        )

    assert result["status"] == "success"
    assert destination.read_bytes() == content
    assert read_attempts["count"] >= 3
