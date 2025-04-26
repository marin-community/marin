import time

import pytest
import ray

from marin.evaluation.evaluators.evaluator import ModelConfig

model_config = ModelConfig(
    name="test-llama-200m",
    path="gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m",
    engine_kwargs={"enforce_eager": True, "max_model_len": 1024},
)


@pytest.fixture
def current_date_time():
    # Get the current local time and format as MM-DD-YYYY-HH-MM-SS
    formatted_time = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())

    return formatted_time


@pytest.fixture(scope="module")
def ray_tpu_cluster():
    ray.init(resources={"TPU": 8, "TPU-v6e-8-head": 1}, num_cpus=120, ignore_reinit_error=True)
    yield
    ray.shutdown()
