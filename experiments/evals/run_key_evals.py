from experiments.evals.evals import default_key_evals
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from marin.execution.executor import executor_main

# Insert your model path here
model_path = "gs://levanter-checkpoints/marin/llama_3.1_mixture_total/hf/seed_0/pzdvslee/step-4000/"

key_evals = default_key_evals(
    step=model_path,
    resource_config=SINGLE_TPU_V6E_8,
    model_name="llama_3.1_mixture_total_seed_0_step4000",
)

if __name__ == "__main__":
    executor_main(steps=key_evals)
