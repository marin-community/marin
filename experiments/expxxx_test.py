import ray
from vllm import LLM, SamplingParams


@ray.remote(resources={"TPU": 1})
def test_llm():
    llm = LLM(model="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-3B-Instruct--0cb88a4", max_model_len=8192)
    outputs = llm.generate(
        "Write a 1024 character long story about a cat driving into the forest.",
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1024),
    )
    return outputs


@ray.remote
def test_llms_on_same_node():
    node_id = ray.get_runtime_context().get_node_id()

    num_tpus = 8
    futures = []
    for _ in range(num_tpus):
        futures.append(
            test_llm.options(
                resources={"TPU": 1},
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id, soft=False),
            ).remote()
        )
    ray.get(futures)


if __name__ == "__main__":
    ray.get(test_llms_on_same_node.remote())
