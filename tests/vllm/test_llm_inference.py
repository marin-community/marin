import os

import pytest
import ray

try:
    from vllm import LLM, SamplingParams
except ImportError:
    pytest.skip("vLLM is not installed", allow_module_level=True)

from marin.generation.ray_utils import scheduling_strategy_fn


@ray.remote(scheduling_strategy=scheduling_strategy_fn(tensor_parallel_size=8, strategy="STRICT_PACK"))
class LLMActor:
    def __init__(self):
        self.llm = LLM(
            # model="/opt/gcsfuse_mount/models/meta-llama--Llama-3-1-8B-Instruct",
            model="meta-llama/Llama-3.3-70B-Instruct",
            tensor_parallel_size=8,
            max_model_len=8192,
            enforce_eager=True,
        )

    def generate(self, prompt: str):
        sampling_params = SamplingParams(
            max_tokens=100,
            temperature=0.7,
        )

        generated_texts = self.llm.generate(prompt, sampling_params=sampling_params)
        return generated_texts


@ray.remote(scheduling_strategy=scheduling_strategy_fn(tensor_parallel_size=8, strategy="STRICT_PACK"))
def _test_llm_func():
    llm = LLM(
        model="/opt/gcsfuse_mount/models/meta-llama--Llama-3-3-70B-Instruct",
        tensor_parallel_size=8,
        max_model_len=8192,
        enforce_eager=True,
    )

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
def test_llm_inference():
    generated_texts = ray.get(_test_llm_func.remote())
    assert len(generated_texts) == 1

    llm_actor = LLMActor.remote()
    generated_texts = ray.get(llm_actor.generate.remote("Hello, how are you?"))
    assert len(generated_texts) == 1
