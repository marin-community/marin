from experiments.isoflop_sweep import generate_isoflop_sweep
from experiments.dolmino.tokenize_dolmino import get_dolmino_step_llama3
from marin.execution.executor import executor_main

dataset = get_dolmino_step_llama3("dclm")
sweep = generate_isoflop_sweep(dataset, experiment_name="dolmino-dclm-sweep")

if __name__ == "__main__":
    executor_main(sweep)
