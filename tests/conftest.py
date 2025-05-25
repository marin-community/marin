import os
import time

import pytest
import ray

from marin.evaluation.evaluators.evaluator import ModelConfig

default_engine_kwargs = {"enforce_eager": True, "max_model_len": 1024}

large_model_engine_kwargs = {"max_model_len": 1024, "tensor_parallel_size": 8}

default_generation_params = {"max_tokens": 16}

DEFAULT_BUCKET_NAME = "marin-us-east5"
DEFAULT_DOCUMENT_PATH = "documents/test-document-path"


@pytest.fixture(scope="module")
def model_config():
    config = ModelConfig(
        name="test-llama-200m",
        path="gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m",
        engine_kwargs=default_engine_kwargs,
        generation_params=default_generation_params,
    )
    yield config
    config.destroy()


@pytest.fixture
def gcsfuse_mount_model_path():
    return "/opt/gcsfuse_mount/perplexity-models/llama-200m"


@pytest.fixture
def gcsfuse_mount_llama_70b_model_path():
    return "/opt/gcsfuse_mount/models/meta-llama--Llama-3-3-70B-Instruct"


@pytest.fixture
def gcsfuse_mount_llama_8b_model_path():
    return "/opt/gcsfuse_mount/models/meta-llama--Llama-3-1-8B-Instruct"


@pytest.fixture
def test_file_path():
    return "gs://marin-us-east5/documents/chris-test/test_50.jsonl.gz"


@pytest.fixture
def current_date_time():
    # Get the current local time and format as MM-DD-YYYY-HH-MM-SS
    formatted_time = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())

    return formatted_time


@pytest.fixture(scope="module")
def ray_cluster():
    if os.getenv("START_RAY_TPU_CLUSTER") == "true":
        ray.init(resources={"TPU": 8, "TPU-v6e-8-head": 1}, num_cpus=120, ignore_reinit_error=True)
    elif os.getenv("START_RAY_GPU_CLUSTER") == "true":
        ray.init(num_gpus=1, num_cpus=4, ignore_reinit_error=True)
    else:
        ray.init("auto", ignore_reinit_error=True)
    yield
    ray.shutdown()
