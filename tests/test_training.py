# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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


def test_lm_config_with_train_urls_allowed_out_of_region(trainer_config):
    """train/validation source URLs are exempt from region checks."""
    with (
        patch("iris.marin_fs.marin_region", return_value="us-central1"),
        patch("iris.marin_fs.get_bucket_location", return_value="us-east1"),
    ):
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data={"train_urls": ["gs://bucket/path"]},  # type: ignore[arg-type]
                trainer=trainer_config,
            ),
            resources=ResourceConfig.with_tpu("v4-8"),
        )
        _doublecheck_paths(config)


def test_recursive_path_checking(trainer_config):
    """Paths are checked recursively in nested structures."""
    with (
        patch("iris.marin_fs.marin_region", return_value="us-central1"),
        patch("iris.marin_fs.get_bucket_location", return_value="us-east1"),
    ):
        nested_data = MockNestedDataConfig(
            cache_dir="gs://bucket/path", subdir={"file": "gs://bucket/other/path", "list": ["gs://bucket/another/path"]}
        )
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=nested_data,
                trainer=trainer_config,
            ),
            resources=ResourceConfig.with_tpu("v4-8"),
        )
        with pytest.raises(ValueError, match="not in the same region"):
            _doublecheck_paths(config)


def test_dataclass_recursive_checking(trainer_config):
    """Paths are checked recursively in dataclass objects."""
    with (
        patch("iris.marin_fs.marin_region", return_value="us-central1"),
        patch("iris.marin_fs.get_bucket_location", return_value="us-east1"),
    ):
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=MockDataConfig(cache_dir=MockNestedConfig(path="gs://bucket/path")),  # type: ignore
                trainer=trainer_config,
            ),
            resources=ResourceConfig.with_tpu("v4-8"),
        )
        with pytest.raises(ValueError, match="not in the same region"):
            _doublecheck_paths(config)


def test_pathlib_path_handling(trainer_config):
    """pathlib.Path objects that represent GCS URIs are handled correctly."""
    with (
        patch("iris.marin_fs.marin_region", return_value="us-central1"),
        patch("iris.marin_fs.get_bucket_location", return_value="us-east1"),
    ):
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=MockDataConfig(cache_dir=Path("gs://bucket/path")),
                trainer=trainer_config,
            ),
            resources=ResourceConfig.with_tpu("v4-8"),
        )
        with pytest.raises(ValueError, match="not in the same region"):
            _doublecheck_paths(config)
