from experiments.evals.evals import evaluate_alpaca_eval
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from marin.execution.executor import ExecutorMainConfig, executor_main

executor_main_config = ExecutorMainConfig(force_run_failed=True)
steps = [
    evaluate_alpaca_eval(
        model_name="llama3_tulu_5e-6_3eps_fix_tokenizer",
        model_path="gs://marin-us-central2/checkpoints/train_lm_llama3_tulu_sft/hf/seed_0/ivl3pggc/step-3833/",
        resource_config=SINGLE_TPU_V6E_8,
        temperature=0.7,
        # llama 3
        stop_token_ids=[128009],
        # olmo stop token ids
        # stop_token_ids=[100257],
    ),
]

if __name__ == "__main__":
    executor_main(executor_main_config, steps=steps)
