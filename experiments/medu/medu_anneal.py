"""An experiment to evaluate the quality of individual splits of the Dolmino dataset.

We cooldown a 8B model on a 30/70 mixture of some high quality Dolmino split and Dolmino DCLM.
Link to issue: https://github.com/stanford-crfm/marin/issues/820
"""

from experiments.anneal_config import AnnealConfig
from experiments.defaults import default_anneal, default_tokenize
from experiments.dolmino.tokenize_dolmino import get_dolmino_step
from experiments.llama import llama3_tokenizer
from experiments.medu.medu_infer import medu_consolidate
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config

medu_economics3plus_tokenized = default_tokenize(
    name="medu-economics-3plus",
    dataset=medu_consolidate,
    tokenizer=llama3_tokenizer,
)


dolmino_dclm = get_dolmino_step("dclm")
dataset_config = lm_mixture_data_config(
    components={
        "medu-economics-3plus": medu_economics3plus_tokenized,
        "dclm": dolmino_dclm,
    },
    weights={"medu-economics-3plus": 0.30, "dclm": 0.70},
)
# Medu Economics 3+ has 67B tokens consolidated.
medu_anneal_config = AnnealConfig(
    dataset_config=dataset_config,
    num_anneal_training_tokens=8_400_000_000,
)

medu_anneal_model = default_anneal(
    name="llama-8b-anneal-medu-economics-3plus",
    anneal_config=medu_anneal_config,
)

if __name__ == "__main__":
    executor_main([medu_anneal_model])
