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

"""Test whether alpaca eval works."""

import os

import pytest

from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate

MODEL_NAME = "test-alpaca-eval"
TEMPERATURE = 0.7
PRESENCE_PENALTY = 0.0
FREQUENCY_PENALTY = 0.0
REPETITION_PENALTY = 1.0
TOP_P = 1.0
TOP_K = -1


@pytest.mark.gcp
@pytest.mark.skipif(os.getenv("TPU_CI") != "true", reason="Skip this test if not running with a TPU in CI.")
def test_alpaca_eval(current_date_time, ray_tpu_cluster, model_config):
    config = EvaluationConfig(
        evaluator="alpaca",
        model_name=model_config.name,
        model_path=model_config.path,
        evaluation_path=f"gs://marin-us-east5/evaluation/alpaca_eval/{model_config.name}-{current_date_time}",
        resource_config=SINGLE_TPU_V6E_8,
        max_eval_instances=5,
        launch_with_ray=False,
        engine_kwargs={
            "temperature": TEMPERATURE,
            "presence_penalty": PRESENCE_PENALTY,
            "frequency_penalty": FREQUENCY_PENALTY,
            "repetition_penalty": REPETITION_PENALTY,
            "top_p": TOP_P,
            "top_k": TOP_K,
            **model_config.engine_kwargs,
        },
    )
    evaluate(config=config)
