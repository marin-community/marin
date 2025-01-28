from experiments.pretraining_datasets import nemotron_cc
from marin.execution.executor import executor_main

if __name__ == "__main__":
    executor_main(
        steps=[
            nemotron_cc,
        ],
    )
