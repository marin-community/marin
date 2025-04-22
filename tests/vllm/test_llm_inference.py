"""Test whether vLLM can generate simple completions"""

import os

import pytest
import ray

try:
    from vllm import LLM, SamplingParams
except ImportError:
    pytest.skip("vLLM is not installed", allow_module_level=True)

from tests.conftest import model_config


@ray.remote(resources={"TPU-v6e-8-head": 1})
def _test_llm_func(model_config):
    model_path = model_config.ensure_downloaded("/tmp/test-llama-eval")

    llm = LLM(model=model_path, **model_config.engine_kwargs)

    sampling_params = SamplingParams(
        max_tokens=100,
        temperature=0.7,
    )

    generated_texts = llm.generate(
        "Hello, how are you?",
        sampling_params=sampling_params,
    )

    model_config.destroy()

    return generated_texts


@pytest.mark.skipif(os.getenv("TPU_CI") != "true", reason="Skip this test if not running with a TPU in CI.")
def test_local_llm_inference(ray_tpu_cluster):
    ray.get(_test_llm_func.remote(model_config))
