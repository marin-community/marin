"""An experiment to evaluate the quality of individual splits of the Dolma dataset.

We cooldown a 8B model on a 30/70 mixture of some high quality Dolma split and Dolmino DCLM.
Link to issue: https://github.com/stanford-crfm/marin/issues/820
"""

from experiments.anneal_config import AnnealConfig
from experiments.defaults import default_anneal
from experiments.dolma.tokenize_dolma import tokenize_dolma_steps
from experiments.dolmino.tokenize_dolmino import get_dolmino_step_llama3
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config

dolmino_dclm = get_dolmino_step_llama3("dclm")

starcoder_tokenized = tokenize_dolma_steps()["dolma/starcoder"]

dataset_config = lm_mixture_data_config(
    components={
        "starcoder": starcoder_tokenized,
        "dclm": dolmino_dclm,
    },
    weights={"starcoder": 0.30, "dclm": 0.70},
)
# Starcoder dataset has 250B tokens.
starcoder_anneal_config = AnnealConfig(
    dataset_config=dataset_config,
    num_anneal_training_tokens=8_400_000_000,
)

starcoder_anneal_model = default_anneal(
    name="llama-8b-anneal-starcoder",
    anneal_config=starcoder_anneal_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            starcoder_anneal_model,
        ],
    )
