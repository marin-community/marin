from experiments.anneal_config import AnnealConfig
from experiments.defaults import default_anneal
from experiments.dolmino.tokenize_dolmino import get_dolmino_step
from marin.execution.executor import executor_main

dolmino_dclm = get_dolmino_step("dclm")
stackexchange_tokenized = get_dolmino_step("stackexchange")

# Dolmino Stack Exchange dataset has 1.26B tokens.
# Our mixed dataset is 30% dolmino and 70% high-quality web data.
# This means we will epoch dolmino dataset 2 times.
stackexchange_anneal_config = AnnealConfig(
    target_dataset=stackexchange_tokenized,
    high_quality_web_text_dataset=dolmino_dclm,
    high_quality_web_text_proportion=0.70,
    target_dataset_proportion=0.30,
    num_anneal_training_tokens=8_400_000_000,
)

stackexchange_anneal_model = default_anneal(
    name="llama-8b-anneal-stackexchange-0",
    anneal_config=stackexchange_anneal_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            stackexchange_anneal_model,
        ],
    )
