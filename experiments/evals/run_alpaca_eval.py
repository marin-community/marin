# nodryrun
from experiments.evals.evals import evaluate_alpaca_eval
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from marin.execution.executor import ExecutorMainConfig, executor_main

executor_main_config = ExecutorMainConfig(force_run_failed=True)
steps = [
    evaluate_alpaca_eval(
        model_name="debug_double_check_best_main",
        model_path="gs://marin-us-central2/checkpoints/hypnotic_spoonbill_tulu_lr1e-4/hf/seed_0/u5uhwdcn/step-4500/",
        resource_config=SINGLE_TPU_V6E_8,
        temperature=0.7,
    ),
]

if __name__ == "__main__":
    executor_main(executor_main_config, steps=steps)
