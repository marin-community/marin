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


import time

import pytest
from fray.v2 import ResourceConfig
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.run import evaluate

from experiments.evals.task_configs import EvalTaskConfig


@pytest.fixture
def current_date_time():
    # Get the current local time and format as MM-DD-YYYY-HH-MM-SS
    formatted_time = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())

    return formatted_time


@pytest.fixture
def model_config():
    config = ModelConfig(
        name="test-llama-200m",
        path="gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m",
        engine_kwargs={"enforce_eager": True, "max_model_len": 1024},
        generation_params={"max_tokens": 16},
    )
    return config


@pytest.mark.tpu_ci
def test_lm_eval_harness_levanter(current_date_time, model_config):
    mmlu_config = EvalTaskConfig("mmlu", 0, task_alias="mmlu_0shot")
    config = EvaluationConfig(
        evaluator="levanter_lm_evaluation_harness",
        model_name=model_config.name,
        model_path=model_config.path,
        evaluation_path=f"gs://marin-us-east5/evaluation/lm_eval/{model_config.name}-{current_date_time}",
        evals=[mmlu_config],
        max_eval_instances=5,
        launch_with_ray=True,
        resource_config=ResourceConfig.with_cpu(cpu=1),
        engine_kwargs=model_config.engine_kwargs,
    )
    evaluate(config=config)


@pytest.mark.tpu_ci
def test_lm_eval_harness(current_date_time, model_config):
    gsm8k_config = EvalTaskConfig(name="gsm8k_cot", num_fewshot=8)
    config = EvaluationConfig(
        evaluator="lm_evaluation_harness",
        model_name=model_config.name,
        model_path=model_config.path,
        evaluation_path=f"gs://marin-us-east5/evaluation/lm_eval/{model_config.name}-{current_date_time}",
        evals=[gsm8k_config],
        max_eval_instances=1,
        launch_with_ray=True,
        engine_kwargs=model_config.engine_kwargs,
    )
    evaluate(config=config)


@pytest.mark.tpu_ci
def test_alpaca_eval(current_date_time, model_config):
    config = EvaluationConfig(
        evaluator="alpaca",
        model_name=model_config.name,
        model_path=model_config.path,
        evaluation_path=f"gs://marin-us-east5/evaluation/alpaca_eval/{model_config.name}-{current_date_time}",
        max_eval_instances=1,
        launch_with_ray=True,
        engine_kwargs={
            "temperature": 0.7,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            **model_config.engine_kwargs,
        },
    )
    evaluate(config=config)
