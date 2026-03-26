# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for HuggingFace download scripts."""

import io
import json
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from marin.download.huggingface.download_hf import (
    DownloadConfig,
    _get_expected_file_count,
    _relative_path_in_source,
    download_hf,
    stream_file_to_fsspec,
)
from marin.download.huggingface.stream_remove_columns import (
    DatasetConfig,
    prune_hf_dataset,
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


def test_get_expected_file_count_returns_count():
    """_get_expected_file_count returns the number of files from HfApi."""
    cfg = DownloadConfig(
        hf_dataset_id="test-org/test-dataset",
        revision="abc1234",
    )
    repo_files = ["data/file1.txt", "data/file2.txt", "README.md"]
    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = repo_files

    with patch("marin.download.huggingface.download_hf.HfApi", return_value=mock_api):
        result = _get_expected_file_count(cfg)

    mock_api.list_repo_files.assert_called_once_with("test-org/test-dataset", repo_type="dataset", revision="abc1234")
    assert result == 3


def test_download_hf_cross_references_find_with_list_repo_files(mock_hf_fs, tmp_path):
    """download_hf uses hf_fs.find() but cross-references count with list_repo_files()."""
    test_files = {
        "datasets/test-org/test-dataset/data/file1.txt": b"Content 1",
    }
    hf_fs = mock_hf_fs(test_files)
    # list_repo_files returns the same count as find — no truncation
    repo_files = ["data/file1.txt"]
    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = repo_files

    output_path = tmp_path / "output"
    output_path.mkdir()
    cfg = DownloadConfig(
        hf_dataset_id="test-org/test-dataset",
        revision="abc1234",
        gcs_output_path=str(output_path),
    )

    with (
        patch("marin.download.huggingface.download_hf.HfFileSystem", return_value=hf_fs),
        patch("marin.download.huggingface.download_hf.HfApi", return_value=mock_api),
    ):
        download_hf(cfg)

    # find() SHOULD be called — it's the primary listing method
    hf_fs.find.assert_called_once()
    # list_repo_files is called for cross-reference
    mock_api.list_repo_files.assert_called_once()
    assert (output_path / "data" / "file1.txt").exists()


def test_download_hf_raises_on_truncated_find(mock_hf_fs, tmp_path):
    """download_hf raises RuntimeError when find() returns fewer files than list_repo_files()."""
    test_files = {
        "datasets/test-org/test-dataset/data/file1.txt": b"Content 1",
    }
    hf_fs = mock_hf_fs(test_files)
    # list_repo_files reports more files than find() returned — truncation detected
    repo_files = ["data/file1.txt", "data/file2.txt", "data/file3.txt"]
    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = repo_files

    output_path = tmp_path / "output"
    output_path.mkdir()
    cfg = DownloadConfig(
        hf_dataset_id="test-org/test-dataset",
        revision="abc1234",
        gcs_output_path=str(output_path),
    )

    with (
        patch("marin.download.huggingface.download_hf.HfFileSystem", return_value=hf_fs),
        patch("marin.download.huggingface.download_hf.HfApi", return_value=mock_api),
    ):
        with pytest.raises(RuntimeError, match="pagination bug"):
            download_hf(cfg)


def test_download_hf_basic(mock_hf_fs, tmp_path):
    """Test basic HF download functionality."""
    test_files = {
        "datasets/test-org/test-dataset/data/file1.txt": b"Content 1",
        "datasets/test-org/test-dataset/data/file2.txt": b"Content 2",
        "datasets/test-org/test-dataset/README.md": b"# Test Dataset",
    }

    hf_fs = mock_hf_fs(test_files)
    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = ["data/file1.txt", "data/file2.txt", "README.md"]

    output_path = tmp_path / "output"
    output_path.mkdir()

    cfg = DownloadConfig(
        hf_dataset_id="test-org/test-dataset",
        revision="abc1234",
        gcs_output_path=str(output_path),
    )

    with (
        patch("marin.download.huggingface.download_hf.HfFileSystem", return_value=hf_fs),
        patch("marin.download.huggingface.download_hf.HfApi", return_value=mock_api),
    ):
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
    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = ["data/file1.txt"]

    base_output_path = tmp_path / "output"
    revision = "abc1234"

    cfg = DownloadConfig(
        hf_dataset_id="test-org/test-dataset",
        revision=revision,
        gcs_output_path=str(base_output_path),
        append_sha_to_path=True,
    )

    with (
        patch("marin.download.huggingface.download_hf.HfFileSystem", return_value=hf_fs),
        patch("marin.download.huggingface.download_hf.HfApi", return_value=mock_api),
    ):
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


def test_prune_hf_dataset(tmp_path):
    """Test full dataset pruning pipeline."""
    # Create test parquet data
    test_data = pd.DataFrame(
        {
            "id": [1, 2],
            "text": ["hello", "world"],
            "unwanted": ["a", "b"],
        }
    )

    # Create multiple buffers since each call needs a fresh one
    def create_buffer():
        buffer = io.BytesIO()
        test_data.to_parquet(buffer, index=False)
        buffer.seek(0)
        return buffer

    cfg = DatasetConfig(
        hf_repo_id="test-org/test-dataset",
        hf_revision="main",
        hf_paths=["data"],
        output_path=str(tmp_path / "output"),
        keep_columns=["id", "text"],
    )

    # Create output directory structure
    output_dir = tmp_path / "output" / "data"
    output_dir.mkdir(parents=True)

    mock_fs = MagicMock()
    mock_fs.glob = Mock(return_value=["hf://datasets/test-org/test-dataset@main/data/file.parquet"])
    mock_fs.open = Mock(side_effect=lambda path, mode="rb": create_buffer())

    with patch("marin.download.huggingface.stream_remove_columns.hf_fs", mock_fs):
        prune_hf_dataset(cfg)

    # Verify output
    output_file = tmp_path / "output" / "data" / "file.parquet"
    assert output_file.exists()
    result_df = pd.read_parquet(output_file)
    assert list(result_df.columns) == ["id", "text"]


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
        patch("marin.download.huggingface.download_hf.HfFileSystem", return_value=hf_fs),
        patch("marin.download.huggingface.download_hf.time.sleep", return_value=None),
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
