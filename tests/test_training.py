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

import dataclasses
from pathlib import Path
from unittest.mock import patch

import pytest
from fray.v2 import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.distributed import RayConfig
from levanter.main import train_lm
from levanter.trainer import TrainerConfig

from marin.training.training import (
    TrainLmOnPodConfig,
    _doublecheck_paths,
)


@pytest.fixture
def trainer_config():
    """Create a basic trainer config for tests."""
    return TrainerConfig(
        id="test-run",
        checkpointer=CheckpointerConfig(),
        ray=RayConfig(),
    )


@dataclasses.dataclass
class MockDataConfig:
    """Mock data config for testing."""

    cache_dir: str


@dataclasses.dataclass
class MockNestedDataConfig:
    """Mock nested data config for testing."""

    cache_dir: str
    subdir: dict


@dataclasses.dataclass
class MockNestedConfig:
    """Mock nested config for testing."""

    path: str


@pytest.mark.parametrize("tpu_type", [None, "v3-8"])
def test_lm_config_with_local_paths(trainer_config, tpu_type):
    """Test that local paths are allowed when running locally."""
    with (
        patch("marin.training.training.get_vm_region") as mock_get_vm_region,
        patch("marin.training.training.get_bucket_location") as mock_get_bucket_location,
    ):

        # Set up mocks
        mock_get_vm_region.return_value = "us-central1"
        mock_get_bucket_location.return_value = "us-central1"

        # Create a config with local paths
        resources = ResourceConfig.with_tpu(tpu_type) if tpu_type else ResourceConfig.with_cpu()
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=MockDataConfig(cache_dir="local/path"),
                trainer=trainer_config,
            ),
            resources=resources,
        )

        # This should not raise an exception
        _doublecheck_paths(config)


def test_lm_config_with_gcs_paths_same_region(trainer_config):
    """Test that GCS paths in the same region are allowed."""
    with (
        patch("marin.training.training.get_vm_region") as mock_get_vm_region,
        patch("marin.training.training.get_bucket_location") as mock_get_bucket_location,
    ):

        # Set up mocks
        mock_get_vm_region.return_value = "us-central1"
        mock_get_bucket_location.return_value = "us-central1"

        # Create a config with GCS paths in the same region
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=MockDataConfig(cache_dir="gs://bucket/path"),
                trainer=trainer_config,
            ),
            resources=ResourceConfig.with_tpu("v3-8"),  # TPU mode
        )

        # This should not raise an exception
        _doublecheck_paths(config)


def test_lm_config_with_gcs_paths_different_region(trainer_config):
    """Test that GCS paths in a different region raise an exception."""
    with (
        patch("marin.training.training.get_vm_region") as mock_get_vm_region,
        patch("marin.training.training.get_bucket_location") as mock_get_bucket_location,
    ):

        # Set up mocks
        mock_get_vm_region.return_value = "us-central1"
        mock_get_bucket_location.return_value = "us-east1"

        # Create a config with GCS paths in a different region
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=MockDataConfig(cache_dir="gs://bucket/path"),
                trainer=trainer_config,
            ),
            resources=ResourceConfig.with_tpu("v3-8"),  # TPU mode
        )

        # This should raise an exception
        with pytest.raises(ValueError) as excinfo:
            _doublecheck_paths(config)

        assert "not in the same region" in str(excinfo.value)


def test_lm_config_with_allowed_out_of_region_paths(trainer_config):
    """Test that paths in allow_out_of_region are allowed to be in different regions."""
    with (
        patch("marin.training.training.get_vm_region") as mock_get_vm_region,
        patch("marin.training.training.get_bucket_location") as mock_get_bucket_location,
    ):

        # Set up mocks
        mock_get_vm_region.return_value = "us-central1"
        mock_get_bucket_location.return_value = "us-east1"

        # Create a config with GCS paths in a different region but allowed
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=MockDataConfig(cache_dir="gs://bucket/path"),
                trainer=trainer_config,
            ),
            resources=ResourceConfig.with_tpu("v3-8"),  # TPU mode
            allow_out_of_region=("data.cache_dir",),
        )

        # This should not raise an exception
        _doublecheck_paths(config)


def test_recursive_path_checking(trainer_config):
    """Test that paths are checked recursively in nested structures."""
    with (
        patch("marin.training.training.get_vm_region") as mock_get_vm_region,
        patch("marin.training.training.get_bucket_location") as mock_get_bucket_location,
    ):

        # Set up mocks
        mock_get_vm_region.return_value = "us-central1"
        mock_get_bucket_location.return_value = "us-east1"

        # Create a config with nested GCS paths in a different region
        nested_data = MockNestedDataConfig(
            cache_dir="gs://bucket/path", subdir={"file": "gs://bucket/other/path", "list": ["gs://bucket/another/path"]}
        )

        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=nested_data,
                trainer=trainer_config,
            ),
            resources=ResourceConfig.with_tpu("v3-8"),  # TPU mode
        )

        # This should raise an exception
        with pytest.raises(ValueError) as excinfo:
            _doublecheck_paths(config)

        assert "not in the same region" in str(excinfo.value)


def test_dataclass_recursive_checking(trainer_config):
    """Test that paths are checked recursively in dataclass objects."""
    with (
        patch("marin.training.training.get_vm_region") as mock_get_vm_region,
        patch("marin.training.training.get_bucket_location") as mock_get_bucket_location,
    ):

        # Set up mocks
        mock_get_vm_region.return_value = "us-central1"
        mock_get_bucket_location.return_value = "us-east1"

        # Create a config with a dataclass containing a GCS path
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=MockDataConfig(cache_dir=MockNestedConfig(path="gs://bucket/path")),  # type: ignore
                trainer=trainer_config,
            ),
            resources=ResourceConfig.with_tpu("v3-8"),  # TPU mode
        )

        # This should raise an exception
        with pytest.raises(ValueError) as excinfo:
            _doublecheck_paths(config)

        assert "not in the same region" in str(excinfo.value)


def test_pathlib_path_handling(trainer_config):
    """Test that pathlib.Path objects are handled correctly."""

    with (
        patch("marin.training.training.get_vm_region") as mock_get_vm_region,
        patch("marin.training.training.get_bucket_location") as mock_get_bucket_location,
    ):

        mock_get_vm_region.return_value = "us-central1"
        mock_get_bucket_location.return_value = "us-east1"

        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=MockDataConfig(cache_dir=Path("gs://bucket/path")),
                trainer=trainer_config,
            ),
            resources=ResourceConfig.with_tpu("v3-8"),
        )

        with pytest.raises(ValueError) as excinfo:
            _doublecheck_paths(config)

        assert "not in the same region" in str(excinfo.value)
