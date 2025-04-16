import ray

from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate

MODEL_NAME = "test-alpaca-eval"
temperature: float = (0.7,)
presence_penalty: float = (0.0,)
frequency_penalty: float = (0.0,)
repetition_penalty: float = (1.0,)
top_p: float = (1.0,)
top_k: int = (-1,)

config = EvaluationConfig(
    evaluator="alpaca",
    model_name=MODEL_NAME,
    model_path="gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m",
    evaluation_path=f"gs://marin-us-east5/evaluation/alpaca_eval/{MODEL_NAME}",
    resource_config=SINGLE_TPU_V6E_8,
    engine_kwargs={
        "temperature": temperature,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "top_k": top_k,
    },
)


# Starts a head-node for the cluster.
# cluster = Cluster(
#     initialize_head=True,
#     head_node_args={
#         "num_cpus": 2,
#         "resources": {"TPU": 1}
#     })

# @ray.remote(num_cpus=2)
# def add(x):
#     return x + x


def run_alpaca_eval():
    ray.get(evaluate(config=config))


# @pytest.mark.timeout(2)
# def test_sleep():
#     import time
#     time.sleep(4)


def test_alpaca_eval():
    run_alpaca_eval()
    # x = ray.get(add.remote(2))
    # assert x == 4
