"""
Train 1.4B models on Fineweb Edu and MultiLegalPile with different proportions of law data.
https://github.com/stanford-crfm/marin/issues/231
"""

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama_1_4b, llama_1_4b_train_config
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config
from experiments.pretraining_datasets import fineweb_edu, multilegalpile

fineweb_edu_tokenized = default_tokenize(
    name="fineweb_edu",
    dataset=fineweb_edu,
    tokenizer="meta-llama/Meta-Llama-3.1-8B",
)
multilegalpile_tokenized = default_tokenize(
    name="multilegalpile",
    dataset=multilegalpile,
    tokenizer="meta-llama/Meta-Llama-3.1-8B",
)

LAW_MIXTURE_COMPONENTS = {
    "fineweb_edu": fineweb_edu_tokenized,
    "multilegalpile": multilegalpile_tokenized,
}

FINEWEB_EDU_MIXTURE_WEIGHTS = {
    "fineweb_edu": 10,
    "multilegalpile": 0,
}

MULTILEGALPILE_10_MIXTURE_WEIGHTS = {
    "fineweb_edu": 9,
    "multilegalpile": 1,
}

MULTILEGALPILE_20_MIXTURE_WEIGHTS = {
    "fineweb_edu": 8,
    "multilegalpile": 2,
}

MULTILEGALPILE_50_MIXTURE_WEIGHTS = {
    "fineweb_edu": 5,
    "multilegalpile": 5,
}

fineweb_edu_mixture_config = lm_mixture_data_config(
    components=LAW_MIXTURE_COMPONENTS, weights=FINEWEB_EDU_MIXTURE_WEIGHTS
)
multilegalpile_10_mixture_config = lm_mixture_data_config(
    components=LAW_MIXTURE_COMPONENTS, weights=MULTILEGALPILE_10_MIXTURE_WEIGHTS
)
multilegalpile_20_mixture_config = lm_mixture_data_config(
    components=LAW_MIXTURE_COMPONENTS, weights=MULTILEGALPILE_20_MIXTURE_WEIGHTS
)
multilegalpile_50_mixture_config = lm_mixture_data_config(
    components=LAW_MIXTURE_COMPONENTS, weights=MULTILEGALPILE_50_MIXTURE_WEIGHTS
)

fineweb_edu_model = default_train(
    name="fineweb_edu",
    tokenized=fineweb_edu_mixture_config,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

multilegalpile_10_model = default_train(
    name="multilegalpile_10",
    tokenized=multilegalpile_10_mixture_config,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

multilegalpile_20_model = default_train(
    name="multilegalpile_20",
    tokenized=multilegalpile_20_mixture_config,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

multilegalpile_50_model = default_train(
    name="multilegalpile_50",
    tokenized=multilegalpile_50_mixture_config,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            fineweb_edu_model,
            multilegalpile_10_model,
            multilegalpile_20_model,
            multilegalpile_50_model
        ],
        description="Train 1.4B models on Fineweb Edu and MultiLegalPile with different proportions of law data.",
    )
