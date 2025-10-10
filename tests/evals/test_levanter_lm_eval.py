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

import os

import pytest

from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from experiments.evals.task_configs import EvalTaskConfig
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate


@pytest.mark.gcp
@pytest.mark.skipif(os.getenv("TPU_CI") != "true", reason="Skip this test if not running with a TPU in CI.")
def test_lm_eval_harness(current_date_time, ray_tpu_cluster, model_config):
    mmlu_config = EvalTaskConfig("mmlu", 0, task_alias="mmlu_0shot")
    config = EvaluationConfig(
        evaluator="levanter_lm_evaluation_harness",
        model_name=model_config.name,
        model_path=model_config.path,
        evaluation_path=f"gs://marin-us-east5/evaluation/lm_eval/{model_config.name}-{current_date_time}",
        evals=[mmlu_config],
        max_eval_instances=5,
        launch_with_ray=True,
        engine_kwargs=model_config.engine_kwargs,
        resource_config=SINGLE_TPU_V6E_8,
    )
    evaluate(config=config)
