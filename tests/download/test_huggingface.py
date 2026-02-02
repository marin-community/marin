# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for HuggingFace download scripts."""

import io
import json
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from marin.download.huggingface.download_hf import DownloadConfig, download_hf
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

        def mock_open(path, mode="rb"):
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
    with patch("marin.download.huggingface.download_hf.HfFileSystem", return_value=hf_fs):
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

    with patch("marin.download.huggingface.download_hf.HfFileSystem", return_value=hf_fs):
        download_hf(cfg)

    target_output = base_output_path / revision
    assert (target_output / "data" / "file1.txt").exists()
    assert (target_output / "provenance.json").exists()


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
