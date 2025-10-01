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

import os
import tempfile
from unittest.mock import Mock, patch

import pytest

try:
    from marin.rl.weight_transfer import (
        GCSCheckpointClient,
        WeightTransferConfig,
        WeightTransferMode,
    )
except ImportError:
    pytest.skip("Post training imports unavailable", allow_module_level=True)


pytestmark = pytest.mark.skipif(os.environ.get("CI") is not None, reason="GCS tests are skipped on CI")


@pytest.fixture
def gcs_config():
    """Create weight transfer config for GCS checkpoint mode."""
    return WeightTransferConfig(
        mode=WeightTransferMode.GCS_CHECKPOINT,
        checkpoint_dir="gs://marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints",
    )


@pytest.fixture
def local_config():
    """Create weight transfer config for local filesystem."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = WeightTransferConfig(
            mode=WeightTransferMode.GCS_CHECKPOINT,
            checkpoint_dir=temp_dir,
        )
        yield config


def test_find_latest_checkpoint_with_trailing_slashes(gcs_config):
    """Test handling of trailing slashes in directory names from fs.ls()."""
    client = GCSCheckpointClient(gcs_config)

    # Mock fsspec to return dirs with trailing slashes (common for GCS)
    mock_fs = Mock()
    mock_fs.exists.return_value = True
    mock_fs.ls.return_value = [
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_36/",
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_37/",
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_38/",
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_39/",
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_40/",
    ]

    with patch(
        "fsspec.core.url_to_fs",
        return_value=(
            mock_fs,
            "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints",
        ),
    ):
        result = client._find_latest_checkpoint()

    # Should find step_40 (highest number) and reconstruct full GCS URL
    # Note: preserves trailing slash from original fs.ls() result
    expected_path = "gs://marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_40/"
    expected_step = 40
    assert result == (expected_path, expected_step)


def test_find_latest_checkpoint_without_trailing_slashes(gcs_config):
    """Test handling when fs.ls() returns paths without trailing slashes."""
    client = GCSCheckpointClient(gcs_config)

    mock_fs = Mock()
    mock_fs.exists.return_value = True
    mock_fs.ls.return_value = [
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_36",
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_40",
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_38",
    ]

    with patch(
        "fsspec.core.url_to_fs",
        return_value=(
            mock_fs,
            "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints",
        ),
    ):
        result = client._find_latest_checkpoint()

    expected_path = "gs://marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_40"
    expected_step = 40
    assert result == (expected_path, expected_step)


def test_find_latest_checkpoint_mixed_formats(gcs_config):
    """Test handling of mixed directory formats."""
    client = GCSCheckpointClient(gcs_config)

    mock_fs = Mock()
    mock_fs.exists.return_value = True
    mock_fs.ls.return_value = [
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_36/",
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_40",
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/other_file.txt",
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_38/",
    ]

    with patch(
        "fsspec.core.url_to_fs",
        return_value=(
            mock_fs,
            "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints",
        ),
    ):
        result = client._find_latest_checkpoint()

    # Should still find step_40 as the latest
    expected_path = "gs://marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_40"
    expected_step = 40
    assert result == (expected_path, expected_step)


def test_find_latest_checkpoint_step_number_extraction(gcs_config):
    """Test correct extraction of step numbers from various formats."""
    client = GCSCheckpointClient(gcs_config)

    mock_fs = Mock()
    mock_fs.exists.return_value = True
    mock_fs.ls.return_value = [
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_9/",
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_100/",
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_39/",
    ]

    with patch(
        "fsspec.core.url_to_fs",
        return_value=(
            mock_fs,
            "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints",
        ),
    ):
        result = client._find_latest_checkpoint()

    # Should find step_100 (highest number), not step_9
    # Note: preserves trailing slash from original fs.ls() result
    expected_path = "gs://marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_100/"
    expected_step = 100
    assert result == (expected_path, expected_step)


def test_find_latest_checkpoint_local_filesystem(local_config):
    """Test checkpoint discovery with local filesystem paths."""
    client = GCSCheckpointClient(local_config)

    mock_fs = Mock()
    mock_fs.exists.return_value = True
    mock_fs.ls.return_value = [
        f"{local_config.checkpoint_dir}/step_5",
        f"{local_config.checkpoint_dir}/step_10",
        f"{local_config.checkpoint_dir}/step_3",
    ]

    with patch("fsspec.core.url_to_fs", return_value=(mock_fs, local_config.checkpoint_dir)):
        result = client._find_latest_checkpoint()

    # For local paths, no scheme reconstruction needed
    expected_path = f"{local_config.checkpoint_dir}/step_10"
    expected_step = 10
    assert result == (expected_path, expected_step)


def test_find_latest_checkpoint_empty_directory(gcs_config):
    """Test behavior when checkpoint directory is empty."""
    client = GCSCheckpointClient(gcs_config)

    mock_fs = Mock()
    mock_fs.exists.return_value = True
    mock_fs.ls.return_value = []

    with patch(
        "fsspec.core.url_to_fs",
        return_value=(
            mock_fs,
            "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints",
        ),
    ):
        result = client._find_latest_checkpoint()

    assert result is None


def test_find_latest_checkpoint_nonexistent_directory(gcs_config):
    """Test behavior when checkpoint directory doesn't exist."""
    client = GCSCheckpointClient(gcs_config)

    mock_fs = Mock()
    mock_fs.exists.return_value = False

    with patch(
        "fsspec.core.url_to_fs",
        return_value=(
            mock_fs,
            "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints",
        ),
    ):
        result = client._find_latest_checkpoint()

    assert result is None


def test_find_latest_checkpoint_no_step_directories(gcs_config):
    """Test behavior when directory has files but no step_ directories."""
    client = GCSCheckpointClient(gcs_config)

    mock_fs = Mock()
    mock_fs.exists.return_value = True
    mock_fs.ls.return_value = [
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/config.json",
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/log.txt",
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/other_dir/",
    ]

    with patch(
        "fsspec.core.url_to_fs",
        return_value=(
            mock_fs,
            "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints",
        ),
    ):
        result = client._find_latest_checkpoint()

    assert result is None


def test_url_reconstruction_with_scheme(gcs_config):
    """Test that URL reconstruction preserves the original scheme."""
    client = GCSCheckpointClient(gcs_config)

    mock_fs = Mock()
    mock_fs.exists.return_value = True
    mock_fs.ls.return_value = [
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_42",
    ]

    with patch(
        "fsspec.core.url_to_fs",
        return_value=(
            mock_fs,
            "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints",
        ),
    ):
        result = client._find_latest_checkpoint()

    # Should reconstruct with gs:// scheme
    assert result is not None
    checkpoint_path, step_num = result
    assert checkpoint_path.startswith("gs://")
    assert "step_42" in checkpoint_path
    assert step_num == 42


def test_url_reconstruction_already_has_scheme(gcs_config):
    """Test that URL reconstruction doesn't double-add scheme."""
    client = GCSCheckpointClient(gcs_config)

    mock_fs = Mock()
    mock_fs.exists.return_value = True
    # Simulate fs.ls() returning full URLs (edge case)
    mock_fs.ls.return_value = [
        "gs://marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_42",
    ]

    with patch(
        "fsspec.core.url_to_fs",
        return_value=(
            mock_fs,
            "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints",
        ),
    ):
        result = client._find_latest_checkpoint()

    # Should not have double gs:// prefix
    assert result is not None
    checkpoint_path, step_num = result
    assert checkpoint_path == "gs://marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_42"
    assert not checkpoint_path.startswith("gs://gs://")
    assert step_num == 42


def test_step_parsing_with_very_large_numbers(gcs_config):
    """Test parsing of very large step numbers."""
    client = GCSCheckpointClient(gcs_config)

    mock_fs = Mock()
    mock_fs.exists.return_value = True
    mock_fs.ls.return_value = [
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_999999/",
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_1000000/",
        "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints/step_1000001/",
    ]

    with patch(
        "fsspec.core.url_to_fs",
        return_value=(
            mock_fs,
            "marin-eu-west4/rl_testing/llama-1b-math-rl-test-0-35d621/policy_checkpoints",
        ),
    ):
        result = client._find_latest_checkpoint()

    assert result is not None
    checkpoint_path, step_num = result
    assert "step_1000001" in checkpoint_path
    assert step_num == 1000001
