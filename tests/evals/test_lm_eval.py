import os

import pytest

from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from experiments.evals.task_configs import EvalTaskConfig
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate
from tests.conftest import model_config


@pytest.mark.skipif(os.getenv("TPU_CI") != "true", reason="Skip this test if not running with a TPU in CI.")
def test_lm_eval_harness(current_date_time, ray_tpu_cluster):
    gsm8k_config = EvalTaskConfig(name="gsm8k_cot", num_fewshot=8)
    config = EvaluationConfig(
        evaluator="lm_evaluation_harness",
        model_name=model_config.name,
        model_path=model_config.path,
        evaluation_path=f"gs://marin-us-east5/evaluation/lm_eval/{model_config.name}-{current_date_time}",
        evals=[gsm8k_config],
        max_eval_instances=5,
        launch_with_ray=True,
        engine_kwargs=model_config.engine_kwargs,
        resource_config=SINGLE_TPU_V6E_8,
    )
    evaluate(config=config)
