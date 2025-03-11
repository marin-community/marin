import ray
from vllm import LLM, SamplingParams

from marin.generation.ray_utils import scheduling_strategy_fn


@ray.remote(scheduling_strategy=scheduling_strategy_fn(tensor_parallel_size=8))
def test_llm():
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

    print(generated_texts)


ray.get(test_llm.remote())
