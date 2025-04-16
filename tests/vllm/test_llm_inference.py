import os

import pytest
import ray

try:
    from vllm import LLM, SamplingParams
except ImportError:
    pytest.skip("vLLM is not installed", allow_module_level=True)

from contextlib import contextmanager

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.generation.ray_utils import scheduling_strategy_fn


@contextmanager
def test_model_config():
    model_config = ModelConfig(
        name="test-llama-200m",
        path="gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m",
        engine_kwargs={"enforce_eager": True, "max_model_len": 2048},
    )

    model_path = model_config.ensure_downloaded("/tmp/test-llama-eval")
    yield {
        "local_model_path": model_path,
        "engine_kwargs": model_config.engine_kwargs,
    }
    model_config.destroy()


@ray.remote(scheduling_strategy=scheduling_strategy_fn(tensor_parallel_size=1, strategy="STRICT_PACK"))
def _test_llm_func():
    with test_model_config() as model_config:
        llm = LLM(model=model_config["local_model_path"], **model_config["engine_kwargs"])

        sampling_params = SamplingParams(
            max_tokens=100,
            temperature=0.7,
        )

        generated_texts = llm.generate(
            "Hello, how are you?",
            sampling_params=sampling_params,
        )

        return generated_texts


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip this test in CI, since we run it as a separate worflow.")
def test_local_llm_inference():
    generated_texts = ray.get(_test_llm_func.remote())
    assert len(generated_texts) == 1


if __name__ == "__main__":
    # Manually create the ModelConfig and run the test logic
    # model_config = ModelConfig(
    #     name="test-llama-200m",
    #     path="gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m",
    #     engine_kwargs={"enforce_eager": True, "max_model_len": 2048}
    # )
    # local_model_path = model_config.ensure_downloaded("/tmp/test-llama-eval")
    # engine_kwargs = model_config.engine_kwargs
    generated_texts = ray.get(_test_llm_func.remote())
    print("Generated texts:", generated_texts)
