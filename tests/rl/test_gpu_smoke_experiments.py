# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.iris_rl_gpu_smoke import (
    DEFAULT_MODEL_PATH,
    gpu_smoke_model_path,
    resolve_gpu_smoke_model_path,
)


def test_gpu_smoke_model_path_defaults_to_requested_region():
    assert resolve_gpu_smoke_model_path(region="us-east5", model_path=None) == gpu_smoke_model_path("us-east5")


def test_gpu_smoke_model_path_rejects_cross_region_gcs_path():
    with pytest.raises(ValueError, match="launcher region"):
        resolve_gpu_smoke_model_path(region="us-east5", model_path=DEFAULT_MODEL_PATH)


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
    assert resolve_gpu_smoke_model_path(region="us-east5", model_path=model_path) == model_path
