# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fray import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.main import train_lm
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.trainer import TrainerConfig
from marin.training.training import (
    TrainLmOnPodConfig,
    _doublecheck_paths,
    _prepare_training_run,
    _update_config_to_use_out_path,
    run_levanter_train_lm,
    temporary_checkpoint_base_path,
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
    auto_build_caches: bool = False


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


def test_temporary_checkpoint_base_path_follows_output_path_region():
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-central1/scratch"}),
    ):
        assert temporary_checkpoint_base_path("gs://marin-us-east5/experiments/grug/base-trial") == (
            "gs://marin-us-east5/tmp/ttl=14d/" "checkpoints-temp/marin-us-east5/experiments/grug/base-trial/checkpoints"
        )


def test_update_config_to_use_out_path_sets_run_specific_temp_checkpoints(trainer_config):
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-central1/scratch"}),
    ):
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                trainer=trainer_config,
            ),
            resources=ResourceConfig.with_tpu("v4-8"),
            output_path="gs://marin-us-east5/experiments/grug/base-trial",
        )

        updated = _update_config_to_use_out_path(config)

        checkpointer = updated.train_config.trainer.checkpointer
        assert checkpointer.base_path == "gs://marin-us-east5/experiments/grug/base-trial/checkpoints"
        assert checkpointer.temporary_base_path == (
            "gs://marin-us-east5/tmp/ttl=14d/" "checkpoints-temp/marin-us-east5/experiments/grug/base-trial/checkpoints"
        )


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


def test_prepare_training_run_adds_eval_extra_for_lm_eval_harness(trainer_config, monkeypatch):
    """LM jobs with eval harness enabled include the eval dependency group."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("WANDB_API_KEY", "test-key")

    with (
        patch("levanter.infra.cli_helpers.load_config") as load_config,
        patch("marin.training.training._doublecheck_paths"),
    ):
        load_config.return_value.env_for_accel.return_value = {}
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=MockDataConfig(cache_dir="gs://bucket/path"),
                trainer=trainer_config,
                eval_harness=LmEvalHarnessConfig(task_spec=["mmlu"]),
            ),
            resources=ResourceConfig.with_tpu("v4-8"),
        )
        _, _, _, extras = _prepare_training_run(config)

    assert extras == ["tpu", "eval"]


def test_run_levanter_train_lm_uses_configured_child_job_name(trainer_config):
    config = TrainLmOnPodConfig(
        train_config=train_lm.TrainLmConfig(
            data=MockDataConfig(cache_dir="gs://bucket/path"),
            trainer=trainer_config,
        ),
        resources=ResourceConfig.with_tpu("v4-8"),
        job_name="train_lm_custom",
    )

    with (
        patch("marin.training.training._prepare_training_run", return_value=(config, config.train_config, {}, [])),
        patch("marin.training.training._submit_training_job") as submit_training_job,
    ):
        run_levanter_train_lm(config)

    assert submit_training_job.call_args.kwargs["job_name"] == "train_lm_custom"


def test_output_path_scopes_temporary_checkpoints(trainer_config):
    """Executor output paths namespace temporary checkpoints to avoid cross-run restores."""
    config = TrainLmOnPodConfig(
        train_config=train_lm.TrainLmConfig(
            data=MockDataConfig(cache_dir="gs://bucket/path"),
            trainer=trainer_config,
        ),
        resources=ResourceConfig.with_tpu("v4-8"),
        output_path="gs://marin-us-east5/experiments/checkpoints/foo/bar/output-hash/",
    )

    with patch(
        "marin.training.training.marin_temp_bucket", return_value="gs://marin-tmp-us-east5/ttl=14d/checkpoints-temp"
    ):
        updated = _update_config_to_use_out_path(config)

    checkpointer = updated.train_config.trainer.checkpointer
    assert checkpointer.base_path == "gs://marin-us-east5/experiments/checkpoints/foo/bar/output-hash/checkpoints"
    assert checkpointer.temporary_base_path == "gs://marin-tmp-us-east5/ttl=14d/checkpoints-temp/output-hash"
