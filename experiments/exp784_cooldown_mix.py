"""
Prepare high-quality cooldown mix for annealing.
"""

import dataclasses
import os

from experiments.dclm.tokenize_dclm import dclm_tokenized_llama3
from experiments.dolma.tokenize_dolma import tokenize_dolma_steps
from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_150m, llama_150m_train_config
from experiments.pretraining_datasets import finemath, dclm_baseline
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned, output_path_of
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config

TOKENIZER = llama3_tokenizer
assert "us-central2" in os.environ["MARIN_PREFIX"], "Update the below code to refer to (1) the correct Dolma dataset since dolma/tokenize_dolma currently hardcodes us-central2 and (2) the correct finemath dataset (after running finemath in pretraining_datasets.py)"

finemath_filename = "gs://marin-us-central2/raw/finemath-7090a5/finemath-3plus/train-{00000..00128}-of-00128.parquet"

steps: dict[str, ExecutorStep[TokenizeConfig]] = {}

# Rarer high-quality sources

dolma_splits = ["dolma/algebraic-stack", "dolma/arxiv", "dolma/megawika", "dolma/open-web-math", "dolma/pes2o", "dolma/stackexchange", "dolma/wiki"]
all_dolma_steps = tokenize_dolma_steps()
steps.update({dataset: step for dataset, step in all_dolma_steps.items() if dataset in dolma_splits})

steps["finemath_3_plus"] = default_tokenize(name="finemath_3_plus", dataset=finemath.cd("finemath-3plus"), tokenizer=llama3_tokenizer)

# DCLM data (to serve as web data), TODO: filter for higher quality
steps["dclm"] = default_tokenize(name="dclm_baseline", dataset=dclm_baseline, tokenizer=llama3_tokenizer)

# these counts are done with llama tokens (https://docs.google.com/spreadsheets/d/1ykVJ1EGJvA1zwF67FZGFBzlm7P0ZBIMuCpBW9Pqp7cY/edit?gid=0#gid=0), slightly different from standard olmo tokenizer token counts
# the first number is the number of tokens in the dataset, the second is the desired mixing portion
high_quality_token_counts = {
    "dolma/algebraic-stack": 11.5 * 1.0,
    "dolma/arxiv"          : 27.9 * 1.0,
    "dolma/megawika"       : 4.44 * 1.0,
    "dolma/open-web-math"  : 5.06 * 1.0,
    "dolma/pes2o"          : 58.1 * 1.0,
    "dolma/stackexchange"  : 17.1 * 1.0,
    "dolma/wiki"           : 3.65 * 1.0,
    "finemath_3_plus"      : 34.0 * 1.0, # https://huggingface.co/datasets/HuggingFaceTB/finemath
}

total_high_quality_token_count = sum()

# reweight data so that 70% of the tokens are dclm and 30% are high-quality sources
cooldown_mixture_weights = {
    **{dataset: 0.3 * token_count / total_high_quality_token_count for dataset, token_count in high_quality_token_counts.items()},
    "dclm": 0.7,
}

data_config = lm_varying_mixture_data_config(
    components=steps,
    weights_list=[
        (0, cooldown_mixture_weights),  # using varying mixture only for illustration
    ],
)

llama_150m_train_config = dataclasses.replace(
    llama_150m_train_config,
    num_train_steps=2000,  # 2000 * 1024 * 1024 = 2B tokens
    tpu_type="v4-128",
)

cooldown_model = default_train(
    name="cooldown-trial",
    tokenized=data_config,
    model_config=llama_150m,
    train_config=llama_150m_train_config,
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            cooldown_model,
        ],
        description="Setting up cooldown mix.",
    )
