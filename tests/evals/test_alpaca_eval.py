"""Test whether alpaca eval works."""

import os

import pytest

from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate
from tests.conftest import model_config

MODEL_NAME = "test-alpaca-eval"
TEMPERATURE = 0.7
PRESENCE_PENALTY = 0.0
FREQUENCY_PENALTY = 0.0
REPETITION_PENALTY = 1.0
TOP_P = 1.0
TOP_K = -1


@pytest.mark.skipif(os.getenv("TPU_CI") != "true", reason="Skip this test if not running with a TPU in CI.")
def test_alpaca_eval(current_date_time, ray_tpu_cluster):
    config = EvaluationConfig(
        evaluator="alpaca",
        model_name=model_config.name,
        model_path=model_config.path,
        evaluation_path=f"gs://marin-us-east5/evaluation/alpaca_eval/{model_config.name}-{current_date_time}",
        resource_config=SINGLE_TPU_V6E_8,
        max_eval_instances=5,
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
