from experiments.anneal_config import AnnealConfig
from experiments.defaults import default_anneal
from experiments.midtraining_datasets import finemath_3_plus
from marin.execution.executor import executor_main

finemath_anneal_config = AnnealConfig(
    target_dataset=finemath_3_plus,
)

finemath_anneal_model = default_anneal(
    name="llama-8b-anneal-finemath",
    anneal_config=finemath_anneal_config,
)

control_model = default_anneal(name="llama-8b-anneal-fineweb-edu", anneal_config=AnnealConfig())
annealed_model = default_anneal(name="llama-8b-anneal-finemath", anneal_config=finemath_anneal_config)

if __name__ == "__main__":
    executor_main(
        steps=[
            annealed_model,
            # control_model,
        ],
        description="Train 8B model on DCLM using WSD-S.",
    )
