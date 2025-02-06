"""
Prepare high-quality cooldown mix for annealing.
"""

import dataclasses
import os

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_150m, llama_150m_train_config
from experiments.pretraining_datasets import dclm_baseline
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config


TOKENIZER = llama3_tokenizer
assert "us-central2" in os.environ["MARIN_PREFIX"]
BASE_DIR_DOLMA = "gs://marin-us-central2/raw/dolma/v1.7"

# Rarer high-quality sources

# TODO: add finemath

dolma_datasets = {
    "algebraic-stack": ["algebraic-stack-train-{0000..0015}.json.gz"],
    "arxiv": ["arxiv-{0000..0099}.json.gz"],
    "megawika": ["megawika-{0000..0261}.json.gz"],
    "open-web-math": ["open-web-math-train-{0000..0012}.json.gz"],
    "pes2o": ["pes2o-{0000..0025}.json.gz"],
    "stackexchange": ["stackexchange-{0000..0025}.json.gz"],
    "wiki": ["wiki-{0000..0001}.json.gz"],
}

# since there are done with llama tokens, slightly different from standard olmo tokenizer token counts
high_quality_token_counts = {
    "algebraic-stack": 11.5,
    "arxiv": 27.9,
    "megawika": 44.4,
    "open-web-math": 50.6,
    "pes2o": 58.1,
    "stackexchange": 17.1,
    "wiki": 36.5,
}

total_high_quality_token_count = sum(high_quality_token_counts.values())

steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
for dataset, files in dolma_datasets.items():
    steps[dataset] = ExecutorStep(
        name=os.path.join("tokenized/dolma", dataset),
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=versioned([f"{BASE_DIR_DOLMA}/{file}" for file in files]),
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(TOKENIZER),
        ),
        pip_dependency_groups=["sentencepiece"],
    )

# DCLM data (to serve as web data), filtered for higher quality
steps["dclm"] = default_tokenize(name="dclm", dataset=dclm_baseline, tokenizer=llama3_tokenizer)

cooldown_mixture_weights = {
    **{k: 0.3 * v / total_high_quality_token_count for k, v in high_quality_token_counts.items()},
    "dclm": 0.7,
}

# unnecessarily using a varying mixture so its easier to copy
data_config = lm_varying_mixture_data_config(
    components=steps,
    weights_list=[
        (0, cooldown_mixture_weights),  # At step 0, start with mostly SlimPajama
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
