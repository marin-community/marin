"""
Train Dolma/OLMo models.
https://github.com/stanford-crfm/marin/issues/442
"""

from experiments.defaults import default_train
from experiments.dolma.tokenize_dolma import DOLMA_OLMO_MIXTURE_WEIGHTS, tokenize_dolma_steps
from experiments.llama import llama_1_4b, llama_1_4b_train_config
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config

EXPERIMENT_TAG = ["442_dolma"]

mixture_config = lm_mixture_data_config(
    components=tokenize_dolma_steps(),
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS,
)

train_step = default_train(
    name="dolma-1.4b",
    tokenized=mixture_config,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG,
)

if __name__ == "__main__":
    executor_main(steps=[*tokenize_dolma_steps().values(), train_step])
