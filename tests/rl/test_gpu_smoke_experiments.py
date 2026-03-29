# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.exp_iris_rl_regression_direct_gcs_small_gpu import build_run_config
from experiments.exp_iris_rl_regression_executor_gcs_small_gpu import build_debug_config
from experiments.iris_rl_gpu_smoke import (
    DEFAULT_MODEL_PATH,
    gpu_smoke_curriculum,
    gpu_smoke_model_path,
    gpu_smoke_rollout_count,
    resolve_gpu_smoke_model_path,
)


def test_direct_gpu_smoke_run_config_uses_gpu_resources():
    run_config = build_run_config(region="us-central1", gpu_type="H100", gpu_count=4)

    assert run_config.train_resources.device.kind == "gpu"
    assert run_config.train_resources.device.count == 4
    assert run_config.train_resources.regions == ["us-central1"]
    assert run_config.rollout_resources.device.kind == "gpu"
    assert run_config.rollout_resources.device.count == 4
    assert run_config.rollout_resources.regions == ["us-central1"]
    assert run_config.num_rollout_workers == 1


def test_executor_gpu_smoke_config_uses_gpu_resources_and_sync_vllm():
    config = build_debug_config(
        experiment_name_suffix="exec-gcs-small-gpu",
        num_train_steps=5,
        region="us-central1",
        gpu_type="H100",
        gpu_count=4,
        model_path=None,
    )

    assert config.train_resources.device.kind == "gpu"
    assert config.train_resources.device.count == 4
    assert config.rollout_resources.device.kind == "gpu"
    assert config.rollout_resources.device.count == 4
    assert config.inference_tensor_parallel_size == 4
    assert config.inflight_weight_updates is False
    assert config.num_rollout_workers == 1
    assert config.train_batch_size == 4
    assert config.per_device_parallelism == 1
    assert config.n_prompts == 1
    assert config.n_generations_per_prompt == 4
    assert isinstance(config.model_config.artifact, str)
    assert config.model_config.artifact == gpu_smoke_model_path("us-central1")


def test_gpu_smoke_curriculum_aligns_with_rollout_count():
    curriculum = gpu_smoke_curriculum(run_id="gpu-smoke-test", num_generations=gpu_smoke_rollout_count(4))
    lesson = curriculum.lessons["math_full"]

    assert lesson.sampling_params.n_prompts == 1
    assert lesson.sampling_params.n_generations_per_prompt == 4
    assert curriculum.max_seq_len == 512


def test_gpu_smoke_model_path_defaults_to_requested_region():
    assert resolve_gpu_smoke_model_path(region="us-east5", model_path=None) == gpu_smoke_model_path("us-east5")


def test_gpu_smoke_model_path_rejects_cross_region_gcs_path():
    with pytest.raises(ValueError, match="launcher region"):
        resolve_gpu_smoke_model_path(region="us-east5", model_path=DEFAULT_MODEL_PATH)


def test_gpu_smoke_model_path_rejects_non_gcs_artifacts():
    with pytest.raises(ValueError, match="Marin GCS artifact"):
        resolve_gpu_smoke_model_path(region="us-central1", model_path="meta-llama/Llama-3.1-8B-Instruct")
