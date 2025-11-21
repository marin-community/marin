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


import jax
import pytest
from fray import ResourceConfig
from marin.evaluation.tasks.mock import MockTask
from marin.evaluation.types import EvaluationConfig, ModelConfig

# Use a tiny random model for testing to avoid large downloads and memory usage
TINY_MODEL = "HuggingFaceM4/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def mesh_context():
    # Ensure we are using CPU for testing
    jax.config.update("jax_platform_name", "cpu")
    from levanter.trainer import TrainerConfig

    trainer_config = TrainerConfig()
    mesh_ctx = trainer_config.use_device_mesh()
    mesh_ctx.__enter__()
    yield
    mesh_ctx.__exit__(None, None, None)


def _get_available_backends():
    """Get list of available backends for testing."""
    backends = ["levanter", "transformers"]  # These are always available

    try:
        import vllm  # noqa: F401

        backends.append("vllm")
    except ImportError:
        pass

    return backends


@pytest.fixture(params=_get_available_backends())
def backend(request):
    """Parameterized fixture for different evaluation backends."""
    return request.param


def test_mock_task_integration(mesh_context, backend):
    """Test mock task that generates simple one-word completions."""
    print(f"Testing mock task with model: {TINY_MODEL}, backend: {backend}")

    model_config = ModelConfig(
        name=TINY_MODEL,
        path=None,  # Will load from HF
        engine_kwargs={},
        generation_params={"max_tokens": 1},  # Just generate one token
    )

    class MockEvalConfig:
        name = "mock"
        prompts = ("Hello",)  # Single simple prompt

    config = EvaluationConfig(
        evaluator=backend,
        model_config=model_config,
        evals=[MockEvalConfig()],
        evaluation_path="/tmp/output",
        worker_resources=ResourceConfig(
            cpu=1,
            ram="1GB",
        ),
    )

    task = MockTask(config=config, prompts=["Hello"])
    results = task.run()
    print(f"Mock task results: {results}")
    assert len(results) == 1
    assert results[0].request_id == "mock_0"
    assert len(results[0].text) > 0
