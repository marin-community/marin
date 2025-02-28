"""An experiment to evaluate the quality of individual splits of the Dolmino dataset.

We cooldown a 8B model on a 30/70 mixture of some high quality Dolmino split and Dolmino DCLM.
Link to issue: https://github.com/stanford-crfm/marin/issues/820
"""

from experiments.anneal_config import AnnealConfig
from experiments.defaults import default_anneal
from experiments.dolmino.tokenize_dolmino import get_dolmino_step
from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import lm_mixture_data_config

BASE_MEDU_PATH = "gs://marin-us-east5/documents/quality_filtering/dclm-global-shard-01-of-10-medu-economics-3plus-bbb96b"
FILE_PATTERN = "**/*.jsonl.zst"
medu_economics3plus_tokenized = ExecutorStep(
    name="tokenized/medu-economics-3plus",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[f"{BASE_MEDU_PATH}/{FILE_PATTERN}"],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=llama3_tokenizer,
    ),
    pip_dependency_groups=["tokenize_train"],
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
    executor_main([medu_anneal_model, medu_economics3plus_tokenized])
