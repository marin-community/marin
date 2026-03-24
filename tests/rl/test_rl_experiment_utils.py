# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from levanter.models.llama import LlamaConfig
from marin.rl.rl_experiment_utils import (
    ModelConfig,
    RLExperimentConfig,
    executor_main_config_for_rl_experiment,
    executor_step_resources_for_rl_experiment,
)
from marin.rl.rl_losses import RLOOLoss

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def _test_config(*, train_tpu_type: str, inference_tpu_type: str) -> RLExperimentConfig:
    return RLExperimentConfig(
        model_config=ModelConfig(
            name=MODEL_NAME,
            type="llama",
            tokenizer=MODEL_NAME,
            checkpoint=MODEL_NAME,
            config_class=LlamaConfig,
        ),
        rl_loss=RLOOLoss(
            kl_coef=0.0,
            clip_epsilon_low=0.2,
            clip_epsilon_high=0.28,
            synchronous=True,
            do_trainer_inference_mismatch_importance_sampling=True,
            tis_importance_sampling_ratio_max=2.0,
            do_overlong_filtering=True,
            vocab_tile_size=32064,
        ),
        experiment_name_suffix="test",
        train_tpu_type=train_tpu_type,
        inference_tpu_type=inference_tpu_type,
    )


def test_v5p_executor_step_regions_are_us_central1_or_us_east5():
    resources = executor_step_resources_for_rl_experiment(
        _test_config(train_tpu_type="v5p-8", inference_tpu_type="v5p-8")
    )

    assert resources.regions == ["us-central1", "us-east5"]


def test_non_v5p_executor_step_regions_are_unset():
    resources = executor_step_resources_for_rl_experiment(
        _test_config(train_tpu_type="v6e-4", inference_tpu_type="v6e-4")
    )

    assert resources.regions is None


def test_v5p_executor_main_config_uses_allowed_env_region(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-east5")

    executor_config = executor_main_config_for_rl_experiment(
        _test_config(train_tpu_type="v5p-8", inference_tpu_type="v5p-8")
    )

    assert executor_config.prefix == "gs://marin-us-east5"


def test_v5p_executor_main_config_defaults_to_us_central1_outside_allowed_regions(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-eu-west4")

    executor_config = executor_main_config_for_rl_experiment(
        _test_config(train_tpu_type="v5p-8", inference_tpu_type="v5p-8")
    )

    assert executor_config.prefix == "gs://marin-us-central1"
