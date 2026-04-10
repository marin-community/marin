# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fray.v2 import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import DatasetComponent, PreferenceChatLmDatasetFormat, PreferenceLmDataConfig
from levanter.distributed import RayConfig
from levanter.main import train_lm
from levanter.main.train_dpo import TrainDpoConfig
from levanter.trainer import TrainerConfig

from experiments.defaults import default_dpo, default_tokenize
from experiments.llama import llama_8b
from experiments.simple_dpo_config import SimpleDPOConfig
from marin.processing.tokenize import read_tokenized_cache_stats, tokenized_cache_stats_path
from marin.training.training import (
    TrainDpoOnPodConfig,
    TrainLmOnPodConfig,
    _doublecheck_paths,
    _maybe_auto_resolve_dpo_schedule,
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


def test_tokenized_cache_stats_path_handles_local_and_gcs_paths():
    assert tokenized_cache_stats_path("/tmp/cache", "train") == "/tmp/cache/train/.stats.json"
    assert (
        tokenized_cache_stats_path("gs://bucket/cache_root", "validation")
        == "gs://bucket/cache_root/validation/.stats.json"
    )


def test_read_tokenized_cache_stats(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)
    (train_dir / ".stats.json").write_text(json.dumps({"total_tokens": 123, "total_elements": 45}))

    stats = read_tokenized_cache_stats(str(tmp_path), "train")

    assert stats.total_tokens == 123
    assert stats.total_elements == 45


def test_auto_resolve_dpo_schedule_from_stats(trainer_config, tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)
    (train_dir / ".stats.json").write_text(json.dumps({"total_tokens": 0, "total_elements": 108765}))

    data = PreferenceLmDataConfig(
        components={
            "prefs": DatasetComponent(
                cache_dir=str(tmp_path),
                format=PreferenceChatLmDatasetFormat(),
            )
        },
        train_weights={"prefs": 1.0},
    )
    train_config = TrainDpoConfig(
        data=data,
        trainer=dataclasses.replace(trainer_config, train_batch_size=64, num_train_steps=1, steps_per_eval=1),
        validation_split_fraction=None,
    )
    config = TrainDpoOnPodConfig(
        train_config=train_config,
        resources=ResourceConfig.with_tpu("v4-8"),
        auto_num_epochs=1.0,
        auto_validation_runs=5,
    )

    resolved = _maybe_auto_resolve_dpo_schedule(config)

    assert resolved.train_config.trainer.num_train_steps == 1700
    assert resolved.train_config.run_initial_eval is True
    assert resolved.train_config.scheduled_eval_steps == [425, 850, 1275]


def test_auto_resolve_dpo_schedule_applies_validation_split(trainer_config, tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)
    (train_dir / ".stats.json").write_text(json.dumps({"total_tokens": 0, "total_elements": 250}))

    data = PreferenceLmDataConfig(
        components={
            "prefs": DatasetComponent(
                cache_dir=str(tmp_path),
                format=PreferenceChatLmDatasetFormat(),
            )
        },
        train_weights={"prefs": 1.0},
    )
    train_config = TrainDpoConfig(
        data=data,
        trainer=dataclasses.replace(trainer_config, train_batch_size=128, num_train_steps=1, steps_per_eval=1),
        validation_split_fraction=0.1,
    )
    config = TrainDpoOnPodConfig(
        train_config=train_config,
        resources=ResourceConfig.with_tpu("v4-8"),
        auto_num_epochs=1.0,
    )

    resolved = _maybe_auto_resolve_dpo_schedule(config)

    assert resolved.train_config.trainer.num_train_steps == 2


def test_default_dpo_attaches_lm_validation_sets():
    tokenized = default_tokenize(
        name="test_prefs",
        dataset="gs://example-bucket/preference/train.jsonl.gz",
        tokenizer="marin-community/marin-tokenizer",
        format=PreferenceChatLmDatasetFormat(),
    )
    lm_validation_steps = {
        "paloma/c4_en": default_tokenize(
            name="paloma/c4_en",
            dataset="gs://example-bucket/paloma/c4_en/val.jsonl.gz",
            tokenizer="marin-community/marin-tokenizer",
        ),
        "uncheatable_eval/github_python": default_tokenize(
            name="uncheatable_eval/github_python",
            dataset="gs://example-bucket/uncheatable/github_python.jsonl.gz",
            tokenizer="marin-community/marin-tokenizer",
        ),
    }
    with patch("experiments.defaults.default_validation_sets", return_value=lm_validation_steps):
        step = default_dpo(
            name="dpo/test",
            tokenized=tokenized,
            model_config=llama_8b,
            dpo_config=SimpleDPOConfig(
                resources=ResourceConfig.with_tpu("v4-8"),
                tokenizer="marin-community/marin-tokenizer",
                model_name_or_path="marin-community/marin-8b-instruct",
                reference_model_path="marin-community/marin-8b-instruct",
                reference_is_hf=True,
                validation_split_fraction=None,
            ),
        )

    lm_validation_data = step.config.train_config.lm_validation_data

    assert lm_validation_data is not None
    assert "paloma/c4_en" in lm_validation_data.components
    assert "uncheatable_eval/github_python" in lm_validation_data.components
    assert lm_validation_data.train_weights is not None
    assert all(weight == 0.0 for weight in lm_validation_data.train_weights.values())
    assert step.config.train_config.lm_validation_prefix == "lm_eval"
