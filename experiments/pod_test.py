import time

import ray

from experiments.models import get_model_local_path
from marin.generation.llm_generation import LLMProvider


@ray.remote(resources={"TPU-v6e-8-head": 1})
def test():

    prompts = ["What is 2 + 2?", "What is the capital of France?"]

    sampling_params = {
        "temperature": 0.1,
        "top_p": 0.3,
        "n": 1,
        "max_tokens": 16,
    }

    engine_kwargs = {
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "max_model_len": 8192,
    }

    start = time.time()
    llm_provider = LLMProvider(
        model_name=get_model_local_path("meta-llama/Llama-3.1-8B-Instruct"),
        # model_name="gs://marin-us-east5/gcsfuse_mount/models/meta-llama--Llama-3-1-8B-Instruct",
        # model_name="meta-llama/Llama-3.1-8B-Instruct",
        generation_kwargs=sampling_params,
        engine_kwargs=engine_kwargs,
    )
    end = time.time()
    print(f"Time taken to load model: {end - start}")

    start = time.time()
    outputs = llm_provider.generate(prompts)
    end = time.time()
    print(f"Time taken to generate: {end - start}")
    print(outputs)


if __name__ == "__main__":
    bar = ray.get(test.remote())
