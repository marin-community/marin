"""
Train Llama 150M and run internal eval.
https://github.com/stanford-crfm/marin/issues/456
"""

from experiments.defaults import default_train, supervised_data
from experiments.dolma.tokenize_dolma import DOLMA_OLMO_MIXTURE_WEIGHTS, tokenize_dolma_steps
from experiments.llama import llama_1_4b_train_config, llama_150m
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

EXPERIMENT_TAG = ["456-llama-eval"]

mixture_config = lm_mixture_data_config(
    components=tokenize_dolma_steps(),
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS,
)


train_step = default_train(
    name="llama-150m-eval",
    tokenized=mixture_config,
    model_config=llama_150m,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG,
    supervised_data=supervised_data,
)

if __name__ == "__main__":
    executor_main(steps=[*tokenize_dolma_steps().values(), train_step])
