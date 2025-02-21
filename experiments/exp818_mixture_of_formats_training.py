"""
Tokenizes the Dolma 1.7 datasets.
"""

import copy
import os.path

from experiments.defaults import default_train
from experiments.dolma.tokenize_dolma import BASE_DIR_DOLMA, DOLMA_DATASETS, DOLMA_OLMO_MIXTURE_WEIGHTS
from experiments.evals.evals import default_eval
from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep, lm_mixture_data_config
from marin.utils import fsspec_glob
from experiments.llama import llama_1_4b, llama_1_4b_train_config

EXPERIMENT_TAG = ["mixture-of-formats-training"]

def tokenize_dolma_mixture_steps(
    *, base_path="tokenized/", tokenizer=llama3_tokenizer, DOLMA_DATASETS: dict[str, list[str]] = DOLMA_DATASETS
) -> dict[str, TokenizerStep]:
    """
    Tokenizes the Dolma 1.7 datasets.

    Args:
        base_path (str): The base path for the tokenized datasets.
        tokenizer (Callable[[], Tokenizer]): The tokenizer to use.

    Returns:
        dict[str, ExecutorStep[TokenizeConfig]]: The steps for the tokenized datasets.
    """
    dolma_steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for dataset, files in DOLMA_DATASETS.items():
        data_files = files if "markdownified" in dataset else [f"{BASE_DIR_DOLMA}/{file}" for file in files]

        dolma_steps[os.path.join("dolma", dataset)] = ExecutorStep(
            name=os.path.join(base_path, "dolma", dataset),
            fn=tokenize,
            config=TokenizeConfig(
                train_paths=versioned(data_files),
                validation_paths=versioned([]),
                cache_path=this_output_path(),
                tokenizer=versioned(tokenizer),
            ),
            pip_dependency_groups=["sentencepiece"],
        )

    return dolma_steps


# BASELINE DOLMA TRAINING
no_mixture_tokenized = tokenize_dolma_mixture_steps(DOLMA_DATASETS=DOLMA_DATASETS)
no_mixture_llama3_tokenized = lm_mixture_data_config(
    components=no_mixture_tokenized,
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS,
)

no_mixture_dolma_model = default_train(
    name="dolma-mixture-of-formats-1.4b-no-mixture",
    tokenized=no_mixture_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG + ["no-mixture"],
)
no_mixture_dolma_evals = default_eval(step=no_mixture_dolma_model)

# Clone the weights to avoid mixing up the weights between the different experiments
DOLMA_OLMO_MIXTURE_WEIGHTS_CLONE = copy.deepcopy(DOLMA_OLMO_MIXTURE_WEIGHTS)

# ARXIV ONLY MIXING FOR MARKDOWNIFIED DATASETS
DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/arxiv"] = DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/arxiv"] / 2
DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/arxiv-markdownified"] = DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/arxiv"]

arxiv_markdownified_path = "gs://marin-us-central2/documents/ar5iv/ar5iv-04-2024-no-problem-3971ff/resiliparse-custom-fork"
arxiv_markdownified_files = fsspec_glob(f"{arxiv_markdownified_path}/*.jsonl.gz")

DOLMA_DATASETS["arxiv-markdownified"] = arxiv_markdownified_files

arxiv_only_subbed_tokenized = tokenize_dolma_mixture_steps(DOLMA_DATASETS=DOLMA_DATASETS)
arxiv_only_subbed_llama3_tokenized = lm_mixture_data_config(
    components=arxiv_only_subbed_tokenized,
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS,
)

arxiv_only_subbed_dolma_model = default_train(
    name="dolma-mixture-of-formats-1.4b-arxiv-only-subbed",
    tokenized=arxiv_only_subbed_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG + ["arxiv-only-subbed"],
)

arxiv_only_subbed_dolma_evals = default_eval(step=arxiv_only_subbed_dolma_model)

# WIKI ONLY MIXING FOR MARKDOWNIFIED DATASETS
DOLMA_OLMO_MIXTURE_WEIGHTS_CLONE["dolma/wiki"] = DOLMA_OLMO_MIXTURE_WEIGHTS_CLONE["dolma/wiki"] / 2
DOLMA_OLMO_MIXTURE_WEIGHTS_CLONE["dolma/wiki-markdownified"] = DOLMA_OLMO_MIXTURE_WEIGHTS_CLONE["dolma/wiki"]

wiki_markdownified_path = "gs://marin-us-central2/documents/wikipedia-resiliparse-custom-fork-2569de/20241201"
wiki_markdownified_files = fsspec_glob(f"{wiki_markdownified_path}/*.jsonl.gz")

DOLMA_DATASETS.pop("arxiv-markdownified")
DOLMA_DATASETS["wiki-markdownified"] = wiki_markdownified_files

wiki_only_subbed_tokenized = tokenize_dolma_mixture_steps(DOLMA_DATASETS=DOLMA_DATASETS)
wiki_only_subbed_llama3_tokenized = lm_mixture_data_config(
    components=wiki_only_subbed_tokenized,
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS_CLONE,
)

wiki_only_subbed_dolma_model = default_train(
    name="dolma-mixture-of-formats-1.4b-wiki-only-subbed",
    tokenized=wiki_only_subbed_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG + ["wiki-only-subbed"],
)

wiki_only_subbed_dolma_evals = default_eval(step=wiki_only_subbed_dolma_model)

# WIKI AND ARXIV MIXING FOR MARKDOWNIFIED DATASETS
DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/wiki"] = DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/wiki"] / 2
DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/wiki-markdownified"] = DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/wiki"]

DOLMA_DATASETS["arxiv-markdownified"] = arxiv_markdownified_files

wiki_and_arxiv_subbed_tokenized = tokenize_dolma_mixture_steps(DOLMA_DATASETS=DOLMA_DATASETS)
wiki_and_arxiv_subbed_llama3_tokenized = lm_mixture_data_config(
    components=wiki_and_arxiv_subbed_tokenized,
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS,
)

wiki_and_arxiv_subbed_dolma_model = default_train(
    name="dolma-mixture-of-formats-1.4b-wiki-and-arxiv-subbed",
    tokenized=wiki_and_arxiv_subbed_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG + ["wiki-and-arxiv-subbed"],
)

wiki_and_arxiv_subbed_dolma_evals = default_eval(step=wiki_and_arxiv_subbed_dolma_model)

if __name__ == "__main__":
    tokenize_steps = list(arxiv_only_subbed_tokenized.values()) + list(wiki_only_subbed_tokenized.values()) + list(wiki_and_arxiv_subbed_tokenized.values())
    executor_main(steps=[
        *tokenize_steps,
        no_mixture_dolma_model,
        no_mixture_dolma_evals,
        arxiv_only_subbed_dolma_model,
        arxiv_only_subbed_dolma_evals,
        wiki_only_subbed_dolma_model,
        wiki_only_subbed_dolma_evals,
        wiki_and_arxiv_subbed_dolma_model,
        wiki_and_arxiv_subbed_dolma_evals,
    ])
