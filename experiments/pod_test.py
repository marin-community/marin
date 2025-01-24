import ray

# runtime_env = RuntimeEnv(
#     image_uri="docker.io/christopherchou/vllm-tpu-py11:latest",
#     env_vars={"HF_HOME": "/workspace/"}
#     # image_uri="docker.io/anyscale/ray:2.31.0-py39-cpu"
# )


# @ray.remote(runtime_env=runtime_env)
@ray.remote
def test():
    from vllm import LLM
    from vllm.sampling_params import SamplingParams

    print("Hello, world!")

    prompts = ["What is 2 + 2?", "What is the capital of France?"]

    sampling_params = SamplingParams(temperature=0.7, top_p=1.0, n=1, max_tokens=16)

    llm = LLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        enforce_eager=True,
        max_model_len=8192,
        tensor_parallel_size=8,
    )

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    bar = ray.get(test.remote())
