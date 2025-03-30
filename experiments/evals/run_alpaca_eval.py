from experiments.evals.evals import evaluate_alpaca_eval
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from marin.execution.executor import ExecutorMainConfig, executor_main

executor_main_config = ExecutorMainConfig()
steps = [
    evaluate_alpaca_eval(
        model_name="soft_raccon_tulu3_sft_temp0.7_dont_repeat",
        model_path="gs://marin-us-central2/checkpoints/soft-raccoon-3-step830k_tulu_3_seed1_packed-54cc4a/hf/step-3833/",
        resource_config=SINGLE_TPU_V6E_8,
        temperature=0.7,
        frequency_penalty=2.0,
        repetition_penalty=2.0,
    ),
]

if __name__ == "__main__":
    executor_main(executor_main_config, steps=steps)
