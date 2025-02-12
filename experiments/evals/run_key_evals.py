from experiments.evals.evals import default_key_evals
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from marin.execution.executor import executor_main

model_path = "gs://marin-us-central2/checkpoints/llama3.1_8b_tulu_3_seed0_alldocs-379cf0/hf/step-7334"

key_evals = default_key_evals(
    step=model_path,
    resource_config=SINGLE_TPU_V6E_8,
)

if __name__ == "__main__":
    executor_main(steps=key_evals)
