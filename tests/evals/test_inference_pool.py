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

"""Integration tests for the inference pool with real VLLM servers."""

import os
import tempfile
from logging import getLogger
from typing import Any

import pytest
from fray.cluster.base import ResourceConfig
from fray.cluster.local_cluster import LocalCluster
from fray.queue.file import FileQueue
from marin.evaluation.backends.inference_pool import InferencePool
from marin.evaluation.evaluation_config import EvalTaskConfig, InferencePoolConfig, ModelConfig
from marin.evaluation.evaluators.simple_evaluator import SimpleEvaluator
from openai import OpenAI

logger = getLogger(__name__)


@pytest.fixture(scope="module")
def test_model_config():
    """Create a model config for the tiny baby-llama-58m model."""
    return ModelConfig(
        name="timinar/baby-llama-58m",
        path="timinar/baby-llama-58m",
        engine_kwargs={
            "max_model_len": 128,
        },
        device="auto",
    )


@pytest.fixture(scope="module")
def cluster():
    """Create a local cluster for testing."""
    cluster = LocalCluster()
    yield cluster
    cluster.shutdown()


@pytest.fixture(scope="module")
def inference_pool(test_model_config, cluster):
    """Create an inference pool with 1 worker for testing."""
    queue_dir = tempfile.mkdtemp(prefix="test-pool-")

    request_queue = FileQueue[dict[str, Any]](path=os.path.join(queue_dir, "requests"))
    response_queue = FileQueue[dict[str, Any]](path=os.path.join(queue_dir, "responses"))

    pool_config = InferencePoolConfig(
        resource_config=ResourceConfig(cpu=2, ram="8g", replicas=1),
        model_config=test_model_config,
        proxy_host="127.0.0.1",
        proxy_port=8080,
    )

    with InferencePool(
        config=pool_config,
        cluster=cluster,
        request_queue=request_queue,
        response_queue=response_queue,
    ) as pool:
        pool.wait_for_healthy(timeout=20)
        yield pool


def test_pool_basic_request(inference_pool):
    """Test basic request/response through the inference pool."""
    base_url = inference_pool.base_url()
    client = OpenAI(base_url=base_url, api_key="unused")

    logger.info("Connecting to pool -- base URL: %s", base_url)

    response = client.chat.completions.create(
        model="timinar/baby-llama-58m",
        messages=[{"role": "user", "content": "Hello, my name is"}],
        temperature=0.0,
        max_tokens=10,
    )

    assert response is not None
    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None
    print(f"Response: {response.choices[0].message.content}")


def test_pool_multiple_requests(inference_pool):
    """Test multiple sequential requests through the pool."""
    base_url = inference_pool.base_url()
    client = OpenAI(base_url=base_url, api_key="unused")

    prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In a galaxy far, far away",
    ]

    for prompt in prompts:
        response = client.chat.completions.create(
            model="timinar/baby-llama-58m",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=16,
        )

        assert response is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        print(f"Prompt: {prompt}")
        print(f"Response: {response.choices[0].message.content}")
        print("-" * 80)


def test_simple_evaluator_with_pool(inference_pool, test_model_config):
    """Test SimpleEvaluator using the inference pool."""
    base_url = inference_pool.base_url()

    # Create evaluator
    evaluator = SimpleEvaluator()

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as output_dir:
        # Run evaluation with quick test plan
        evaluator.evaluate(
            model=test_model_config,
            evals=[EvalTaskConfig(name="quick", num_fewshot=1)],
            openai_base_url=base_url,
            output_path=output_dir,
        )

        # If we get here without exceptions, the test passed
        print("SimpleEvaluator completed successfully with pool!")


def test_proxy_health_endpoint(inference_pool):
    """Test that the proxy health endpoint works."""
    import requests

    proxy_url = f"http://{inference_pool.config.proxy_host}:{inference_pool.config.proxy_port}"
    response = requests.get(f"{proxy_url}/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_proxy_models_endpoint(inference_pool):
    """Test that the proxy models endpoint works."""
    import requests

    proxy_url = f"http://{inference_pool.config.proxy_host}:{inference_pool.config.proxy_port}"
    response = requests.get(f"{proxy_url}/v1/models")

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
