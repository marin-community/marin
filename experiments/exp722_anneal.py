from experiments.anneal_config import AnnealConfig
from experiments.defaults import default_anneal
from experiments.dolmino.tokenize_dolmino import get_dolmino_step
from experiments.midtraining_datasets import finemath_3_plus_tokenized
from marin.execution.executor import executor_main

dolmino_dclm = get_dolmino_step("dclm")

finemath_anneal_config = AnnealConfig(
    target_dataset=finemath_3_plus_tokenized,
    high_quality_web_text_dataset=dolmino_dclm,
)

# control_model = default_anneal(name="llama-8b-anneal-dclm",
# anneal_config=AnnealConfig(high_quality_web_text_dataset=dolmino_dclm))
annealed_model = default_anneal(name="llama-8b-anneal-finemath-dclm", anneal_config=finemath_anneal_config)

if __name__ == "__main__":
    executor_main(
        steps=[
            annealed_model,
            # control_model,
        ],
        description="Train 8B model on DCLM using WSD-S.",
    )
