# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from levanter.models.qwen import Qwen3Config
from marin.rl.decoding import DecodingConfig
from marin.rl.rl_experiment_utils import (
    config_class_path,
    default_train_decoding_for_experiment,
    executor_main_config_for_rl_experiment,
)

from experiments.exp_iris_rl_regression_executor_gcs_small_gpu import build_debug_config
from experiments.iris_rl_gpu_smoke import (
    CANONICAL_MODEL_NAME,
    DEFAULT_MODEL_ARTIFACT,
    gpu_smoke_curriculum,
    gpu_smoke_model_path,
    resolve_gpu_smoke_model_artifact,
)


def test_gpu_smoke_debug_config_builds_small_qwen_rloo_curriculum():
    config = build_debug_config(
        experiment_name_suffix="test",
        num_train_steps=1,
        region="us-east5",
        gpu_type="H100",
        gpu_count=1,
        model_path=None,
    )

    assert config.model_config.artifact is DEFAULT_MODEL_ARTIFACT
    assert config.model_config.name == CANONICAL_MODEL_NAME
    assert config.model_config.type == "qwen"
    assert config.model_config.config_class_path == config_class_path(Qwen3Config)
    assert config.train_batch_size == 1
    assert config.n_generations_per_prompt == 2
    assert config.inference_tensor_parallel_size == 1
    assert config.train_resources.device.kind == "gpu"
    assert config.rollout_resources.device.kind == "gpu"

    train_decoding = default_train_decoding_for_experiment(config)
    curriculum = gpu_smoke_curriculum(
        run_id="gpu-smoke-test",
        max_input_tokens=config.max_input_tokens,
        num_generations=config.n_generations_per_prompt,
        train_decoding=train_decoding,
    )
    lesson = curriculum.lessons["math_full"]

    assert lesson.sampling_params.n_prompts == 1
    assert lesson.sampling_params.n_generations_per_prompt == 2
    assert lesson.sampling_params.train_decoding.stop_strings == ["<|im_end|>"]
    assert lesson.sampling_params.max_output_tokens == config.max_output_tokens
    assert curriculum.max_seq_len == config.max_input_tokens + config.max_output_tokens


def test_gpu_smoke_curriculum_uses_decoding_config():
    train_decoding = DecodingConfig(
        temperature=1.0,
        max_output_tokens=128,
        top_k=2048,
        stop_strings=["<|im_end|>"],
    )

    curriculum = gpu_smoke_curriculum(
        run_id="gpu-smoke-test",
        max_input_tokens=64,
        num_generations=3,
        train_decoding=train_decoding,
    )
    lesson = curriculum.lessons["math_full"]

    assert lesson.sampling_params.n_prompts == 1
    assert lesson.sampling_params.n_generations_per_prompt == 3
    assert lesson.sampling_params.train_decoding.top_k == 2048
    assert lesson.sampling_params.train_decoding.stop_strings == ["<|im_end|>"]
    assert lesson.sampling_params.max_output_tokens == 128
    assert curriculum.max_seq_len == 192


def test_build_debug_config_allows_non_gcp_executor_prefix():
    config = build_debug_config(
        experiment_name_suffix="test",
        num_train_steps=1,
        region="ORD1",
        gpu_type="H100",
        gpu_count=1,
        model_path=None,
        executor_prefix="s3://marin-coreweave/rl-smoke",
    )

    assert config.model_config.artifact is DEFAULT_MODEL_ARTIFACT
    assert config.executor_prefix == "s3://marin-coreweave/rl-smoke"
    assert executor_main_config_for_rl_experiment(config).prefix == "s3://marin-coreweave/rl-smoke"


def test_gpu_smoke_model_path_rejects_cross_region_gcs_path():
    with pytest.raises(ValueError, match="launcher region"):
        resolve_gpu_smoke_model_artifact(region="us-east5", model_path=gpu_smoke_model_path("us-central1"))


@pytest.mark.parametrize(
    "model_path",
    [
        "Qwen/Qwen3-0.6B",
        "/mnt/models/qwen3-0.6b",
        "s3://external-bucket/models/qwen3-0.6b",
        "gs://external-bucket/models/qwen3-0.6b",
        "gs://marin-us-east5/models/custom-qwen",
    ],
)
def test_gpu_smoke_model_path_allows_non_cross_region_sources(model_path):
    assert resolve_gpu_smoke_model_artifact(region="us-east5", model_path=model_path) == model_path
