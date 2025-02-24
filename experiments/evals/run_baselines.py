import os

from experiments.evals.evals import default_key_evals
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from experiments.models import llama_3_1_8b, tulu_3_1_8b_sft
from marin.execution.executor import executor_main

# Insert your model path here

llama_3_1_8b_path = os.path.join(os.getenv("MARIN_PREFIX"), llama_3_1_8b.name)
tulu_3_1_8b_sft_path = os.path.join(os.getenv("MARIN_PREFIX"), tulu_3_1_8b_sft.name)

llama_3_8b_key_evals = default_key_evals(
    step=llama_3_1_8b_path,
    resource_config=SINGLE_TPU_V6E_8,
    model_name="meta-llama-3.1-8b",
)

tulu_3_1_8b_key_evals = default_key_evals(
    step=tulu_3_1_8b_sft_path,
    resource_config=SINGLE_TPU_V6E_8,
    model_name="tulu-3-8b",
)

if __name__ == "__main__":
    # Run this first to download the models, then run the key eval
    # executor_main(steps=[llama_3_1_8b, tulu_3_8b_sft])
    executor_main(steps=llama_3_8b_key_evals + tulu_3_1_8b_key_evals)
