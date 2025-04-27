from experiments.evals.evals import default_key_evals
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from marin.execution.executor import executor_main

# Insert your model path here
# model_path = "gs://marin-us-central2/checkpoints/llama-8b-control-00f31b/hf/step-210388"

model_path = "gs://marin-us-central2/checkpoints/train_lm_llama3_tulu_sft/hf/seed_0/ivl3pggc/step-3833/"
key_evals = default_key_evals(
    step=model_path,
    resource_config=SINGLE_TPU_V6E_8,
    # model_name="llama-8b-control-00f31b",
    model_name="llama3_tulu_5e-6_3eps_retry",
)

if __name__ == "__main__":
    executor_main(steps=key_evals)
