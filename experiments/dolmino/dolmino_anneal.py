from experiments.anneal_config import AnnealConfig
from experiments.defaults import default_anneal
from experiments.dolmino.tokenize_dolmino import get_dolmino_step
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config

dolmino_dclm = get_dolmino_step("dclm")

# Control model is 100% dolmino DCLM dataset
control_dataset_config = lm_mixture_data_config(
    components={"dolmino_dclm": dolmino_dclm},
    weights={"dolmino_dclm": 1.0},
)
control_anneal_config = AnnealConfig(
    dataset_config=control_dataset_config,
    num_anneal_training_tokens=8_400_000_000,
)

control_model = default_anneal(
    name="llama-8b-anneal-control",
    anneal_config=control_anneal_config,
)

stackexchange_tokenized = get_dolmino_step("stackexchange")

dataset_config = lm_mixture_data_config(
    components={
        "stackexchange": stackexchange_tokenized,
        "dclm": dolmino_dclm,
    },
    weights={"stackexchange": 0.30, "dclm": 0.70},
)
# Dolmino Stack Exchange dataset has 1.26B tokens.
# Our mixed dataset is 30% dolmino and 70% high-quality web data.
# This means we will epoch dolmino dataset 2 times.
stackexchange_anneal_config = AnnealConfig(
    dataset_config=dataset_config,
    num_anneal_training_tokens=8_400_000_000,
)

stackexchange_anneal_model = default_anneal(
    name="llama-8b-anneal-stackexchange-0",
    anneal_config=stackexchange_anneal_config,
)

# eval_stackexchange_anneal_model = default_eval(
#     step="gs://marin-us-central2/checkpoints/llama-8b-anneal-stackexchange-0-db2e46/hf/step-200470",
#     evals=MMLU_TASKS,
# )

wiki_tokenized = get_dolmino_step("wiki")
wiki_dataset_config = lm_mixture_data_config(
    components={"wiki": wiki_tokenized, "dolmino_dclm": dolmino_dclm},
    weights={"wiki": 0.3, "dolmino_dclm": 0.7},
)
wiki_anneal_config = AnnealConfig(
    dataset_config=wiki_dataset_config,
    num_anneal_training_tokens=8_400_000_000,
)

wiki_anneal_model = default_anneal(
    name="llama-8b-anneal-wiki-0",
    anneal_config=wiki_anneal_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            stackexchange_anneal_model,
            control_model,
            wiki_anneal_model,
        ],
    )
