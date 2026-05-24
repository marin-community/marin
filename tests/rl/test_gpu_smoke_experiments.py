# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.exp_iris_rl_regression_executor_gcs_small_gpu import build_debug_config
from experiments.iris_rl_gpu_smoke import (
    DEFAULT_MODEL_ARTIFACT,
    gpu_smoke_model_path,
    resolve_gpu_smoke_model_artifact,
)


def test_gpu_smoke_model_artifact_defaults_to_executor_managed_model():
    assert resolve_gpu_smoke_model_artifact(region="us-east5", model_path=None) is DEFAULT_MODEL_ARTIFACT


def test_build_debug_config_uses_executor_managed_default_model():
    config = build_debug_config(
        experiment_name_suffix="test",
        num_train_steps=1,
        region="us-east5",
        gpu_type="H100",
        gpu_count=4,
        model_path=None,
    )

    assert config.model_config.artifact is DEFAULT_MODEL_ARTIFACT


def test_gpu_smoke_model_path_rejects_cross_region_gcs_path():
    with pytest.raises(ValueError, match="launcher region"):
        resolve_gpu_smoke_model_artifact(region="us-east5", model_path=gpu_smoke_model_path("us-central1"))


@pytest.mark.parametrize(
    "model_path",
    [
        "meta-llama/Llama-3.1-8B-Instruct",
        "/mnt/models/llama-3.1-8b",
        "s3://external-bucket/models/llama-3.1-8b",
        "gs://external-bucket/models/llama-3.1-8b",
        "gs://marin-us-east5/models/custom-llama",
    ],
)
def test_gpu_smoke_model_path_allows_non_cross_region_sources(model_path):
    assert resolve_gpu_smoke_model_artifact(region="us-east5", model_path=model_path) == model_path
