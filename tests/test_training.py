# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from pathlib import Path
from unittest.mock import patch

import pytest
from fray import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.main import train_lm
from levanter.trainer import TrainerConfig

from marin.training.training import (
    TrainLmOnPodConfig,
    _doublecheck_paths,
    _enforce_run_id,
)


@pytest.fixture
def trainer_config():
    """Create a basic trainer config for tests."""
    return TrainerConfig(
        id="test-run",
        checkpointer=CheckpointerConfig(),
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
        patch("rigging.filesystem.marin_region", return_value="us-central1"),
        patch("rigging.filesystem.get_bucket_location", return_value="us-east1"),
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
        patch("rigging.filesystem.marin_region", return_value="us-central1"),
        patch("rigging.filesystem.get_bucket_location", return_value="us-east1"),
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
        patch("rigging.filesystem.marin_region", return_value="us-central1"),
        patch("rigging.filesystem.get_bucket_location", return_value="us-east1"),
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
        patch("rigging.filesystem.marin_region", return_value="us-central1"),
        patch("rigging.filesystem.get_bucket_location", return_value="us-east1"),
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


# ---------------------------------------------------------------------------
# Cross-region temp checkpoint search-path wiring (mirrortmp://)
# ---------------------------------------------------------------------------


def _make_train_config(*, output_path: str | None, run_id: str | None, impute: bool) -> TrainLmOnPodConfig:
    return TrainLmOnPodConfig(
        train_config=train_lm.TrainLmConfig(
            data={"train_urls": ["gs://bucket/path"]},  # type: ignore[arg-type]
            trainer=TrainerConfig(id=run_id, checkpointer=CheckpointerConfig()),
        ),
        resources=ResourceConfig.with_tpu("v4-8"),
        output_path=output_path,
        impute_run_id_from_output_path=impute,
    )


def test_enforce_run_id_imputed_includes_run_id_literal_in_mirror_search_path():
    """impute_run_id_from_output_path=True ⇒ search path contains the imputed run-id literal,
    not the bare ``checkpoints-temp/`` root (which would glob across all runs)."""
    config = _make_train_config(
        output_path="gs://marin-us-central1/runs/foo-abc123",
        run_id=None,
        impute=True,
    )
    enforced = _enforce_run_id(config)
    search_paths = enforced.train_config.trainer.checkpointer.temporary_search_paths
    assert "mirrortmp://ttl=14d/checkpoints-temp/foo-abc123" in search_paths
    # Bare prefix without run-id must never appear.
    assert "mirrortmp://ttl=14d/checkpoints-temp" not in search_paths
    assert "mirrortmp://ttl=14d/checkpoints-temp/" not in search_paths


def test_enforce_run_id_explicit_includes_run_id_literal_in_mirror_search_path():
    config = _make_train_config(
        output_path="gs://marin-us-central1/runs/whatever",
        run_id="my-explicit-run",
        impute=False,
    )
    enforced = _enforce_run_id(config)
    search_paths = enforced.train_config.trainer.checkpointer.temporary_search_paths
    assert "mirrortmp://ttl=14d/checkpoints-temp/my-explicit-run" in search_paths


def test_enforce_run_id_appends_to_existing_search_paths():
    """Pre-existing entries on the user-supplied config are preserved; mirror entry is appended."""
    base_config = _make_train_config(
        output_path="gs://marin-us-central1/runs/foo-abc123",
        run_id=None,
        impute=True,
    )
    base_config = dataclasses.replace(
        base_config,
        train_config=dataclasses.replace(
            base_config.train_config,
            trainer=dataclasses.replace(
                base_config.train_config.trainer,
                checkpointer=dataclasses.replace(
                    base_config.train_config.trainer.checkpointer,
                    temporary_search_paths=["custom://prior/entry"],
                ),
            ),
        ),
    )
    enforced = _enforce_run_id(base_config)
    paths = enforced.train_config.trainer.checkpointer.temporary_search_paths
    assert paths == [
        "custom://prior/entry",
        "mirrortmp://ttl=14d/checkpoints-temp/foo-abc123",
    ]


def test_enforce_run_id_dedupes_existing_mirror_entry():
    """Idempotent: re-running _enforce_run_id (or a user pre-supplying the entry) doesn't duplicate."""
    config = _make_train_config(
        output_path="gs://marin-us-central1/runs/foo-abc123",
        run_id=None,
        impute=True,
    )
    config = dataclasses.replace(
        config,
        train_config=dataclasses.replace(
            config.train_config,
            trainer=dataclasses.replace(
                config.train_config.trainer,
                checkpointer=dataclasses.replace(
                    config.train_config.trainer.checkpointer,
                    temporary_search_paths=["mirrortmp://ttl=14d/checkpoints-temp/foo-abc123"],
                ),
            ),
        ),
    )
    enforced = _enforce_run_id(config)
    paths = enforced.train_config.trainer.checkpointer.temporary_search_paths
    assert paths.count("mirrortmp://ttl=14d/checkpoints-temp/foo-abc123") == 1
