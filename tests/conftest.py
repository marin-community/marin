import time

import pytest

from marin.evaluation.evaluators.evaluator import ModelConfig

model_config = ModelConfig(
    name="test-llama-200m",
    path="gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m",
    engine_kwargs={"enforce_eager": True, "max_model_len": 2048},
)


@pytest.fixture
def current_date_time():
    # Get the current local time and format as MM-DD-YYYY-HH-MM-SS
    formatted_time = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())

    return formatted_time
