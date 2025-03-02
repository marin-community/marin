"""An experiment to evaluate the quality of individual splits of the Dolmino dataset.

We cooldown a 8B model on a 30/70 mixture of some high quality Dolmino split and Dolmino DCLM.
Link to issue: https://github.com/stanford-crfm/marin/issues/820
"""

from experiments.anneal_config import AnnealConfig
from experiments.dclm.tokenize_dclm import dclm_components_llama3
from experiments.defaults import default_anneal
from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
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
        tokenizer=versioned(llama3_tokenizer),
    ),
    pip_dependency_groups=["tokenize_train"],
)

dclm = dclm_components_llama3["dclm_baseline"]
dataset_config = lm_mixture_data_config(
    components={
        "medu-economics-3plus": medu_economics3plus_tokenized,
        "dclm": dclm,
    },
    weights={"medu-economics-3plus": 0.30, "dclm": 0.70},
)

num_anneal_training_tokens = 50_000_000_000
checkpoint_path = "gs://marin-eu-west4/checkpoints/llama-8b-tootsie-0.001-19ad63/checkpoints/step-660000"
# Medu Economics 3+ has 67B tokens consolidated.
medu_anneal_config = AnnealConfig(
    dataset_config=dataset_config,
    num_anneal_training_tokens=num_anneal_training_tokens,
    tpu_type="v5litepod-128",
    initialize_from_checkpoint_path=checkpoint_path,
)

medu_anneal_model = default_anneal(
    name="llama-8b-660k-dclm-baseline-medu-econ-3plus-50b",
    anneal_config=medu_anneal_config,
)

control_model = default_anneal(
    name="llama-8b-660k-dclm-baseline-50b",
    anneal_config=AnnealConfig(
        dataset_config=lm_mixture_data_config(
            components={"dclm": dclm},
            weights={"dclm": 1.0},
        ),
        num_anneal_training_tokens=num_anneal_training_tokens,
        tpu_type="v5litepod-128",
        initialize_from_checkpoint_path=checkpoint_path,
    ),
)

if __name__ == "__main__":
    executor_main([medu_anneal_model, medu_economics3plus_tokenized, control_model])
