from experiments.evals.evals import default_key_evals
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from marin.execution.executor import executor_main

llama_8b_control_key_evals = default_key_evals(
    step="gs://marin-us-central2/checkpoints/llama-8b-control-00f31b/hf/step-210388",
    resource_config=SINGLE_TPU_V6E_8,
    model_name="llama-8b-control-00f31b",
)

llama_8b_nemotron_post_training_v1_sft_seed1_key_evals = default_key_evals(
    step="gs://marin-us-central2/checkpoints/llama3.1_8b_nemotron_post_training_v1_sft_seed1-11a61e/hf/step-2499",
    resource_config=SINGLE_TPU_V6E_8,
    model_name="llama3.1_8b_nemotron_post_training_v1_sft_seed1-11a61e",
)

if __name__ == "__main__":
    executor_main(
        steps=[
            # *llama_8b_control_key_evals,
            *llama_8b_nemotron_post_training_v1_sft_seed1_key_evals
        ]
    )
