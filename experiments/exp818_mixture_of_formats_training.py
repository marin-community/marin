"""
This experiment compares different mixtures of data formats(markdownified and unmarkdownified)
for training language models. We evaluate four scenarios:

1. Baseline: Using the standard Dolma dataset mixture
2. ArXiv Mixture: Adding markdownified ArXiv data alongside the original Dolma ArXiv data
3. Wikipedia Mixture: Adding markdownified Wikipedia data alongside the original Dolma Wikipedia data
4. Wiki and Arxiv Mixture: Adding markdownified Wikipedia and ArXiv data alongside the original Dolma Wikipedia
                           and ArXiv data

The goal is to determine if exposing models to multiple formats of the same content source
improves model performance.

Reference Issue: https://github.com/stanford-crfm/marin/issues/818

"""

from experiments.defaults import default_train
from experiments.dolma.tokenize_dolma import DOLMA_OLMO_MIXTURE_WEIGHTS, tokenize_dolma_steps
from experiments.dolma.utils import dolma_pipeline_with_modified_dataset
from experiments.evals.evals import default_eval
from experiments.exp575_wikipedia_markdownify import wikipedia_resiliparse_custom_fork
from experiments.exp579_ar5iv_markdownify import ar5iv_no_problem_resiliparse_custom_fork
from experiments.llama import llama_1_4b, llama_1_4b_train_config
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config

EXPERIMENT_TAG = ["mixture-of-formats-training"]

tokenized_dolma_steps = tokenize_dolma_steps()

# BASELINE DOLMA TRAINING
no_mixture_llama3_tokenized = lm_mixture_data_config(
    components=tokenized_dolma_steps,
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS,
)

no_mixture_dolma_model = default_train(
    name="mixture-of-formats-1.4b-no-mixture",
    tokenized=no_mixture_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=[*EXPERIMENT_TAG, "no-mixture"],
)
no_mixture_dolma_evals = default_eval(step=no_mixture_dolma_model)


# ARXIV ONLY MIXING FOR MARKDOWNIFIED DATASETS
arxiv_markdownified_tokenized, arxiv_only_subbed_dolma_model, arxiv_only_subbed_dolma_evals = (
    dolma_pipeline_with_modified_dataset(
        path_prefix="mixture-of-formats-1.4b-arxiv-only-subbed",
        dataset=ar5iv_no_problem_resiliparse_custom_fork,
        dolma_dataset="arxiv",
        experiment_tag=[*EXPERIMENT_TAG, "arxiv-only-subbed"],
        substitute_dolma_dataset=False,
    )
)


# WIKI ONLY MIXING FOR MARKDOWNIFIED DATASETS
wiki_markdownified_tokenized, wiki_only_subbed_dolma_model, wiki_only_subbed_dolma_evals = (
    dolma_pipeline_with_modified_dataset(
        path_prefix="mixture-of-formats-1.4b-wiki-only-subbed",
        dataset=wikipedia_resiliparse_custom_fork,
        dolma_dataset="wiki",
        experiment_tag=[*EXPERIMENT_TAG, "wiki-only-subbed"],
        substitute_dolma_dataset=False,
    )
)

# WIKI AND ARXIV MIXING FOR MARKDOWNIFIED DATASETS
wiki_and_arxiv_tokenization_steps = {
    "mixture-of-formats-1.4b-wiki-only-subbed": wiki_markdownified_tokenized,
    "mixture-of-formats-1.4b-arxiv-only-subbed": arxiv_markdownified_tokenized,
}
wiki_and_arxiv_weights = {
    # **DOLMA_OLMO_MIXTURE_WEIGHTS,
    "mixture-of-formats-1.4b-wiki-only-subbed": DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/wiki"],
    "mixture-of-formats-1.4b-arxiv-only-subbed": DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/arxiv"],
}

wiki_and_arxiv_subbed_llama3_tokenized = lm_mixture_data_config(
    components=wiki_and_arxiv_tokenization_steps,
    weights=wiki_and_arxiv_weights,
)

wiki_and_arxiv_subbed_dolma_model = default_train(
    name="mixture-of-formats-1.4b-wiki-and-arxiv-subbed",
    tokenized=wiki_and_arxiv_subbed_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=[*EXPERIMENT_TAG, "wiki-and-arxiv-subbed"],
)

wiki_and_arxiv_subbed_dolma_evals = default_eval(step=wiki_and_arxiv_subbed_dolma_model)

if __name__ == "__main__":
    tokenize_steps = [
        *list(tokenized_dolma_steps.values()),
        wiki_markdownified_tokenized,
        arxiv_markdownified_tokenized,
        *list(wiki_and_arxiv_tokenization_steps.values()),
    ]

    executor_main(
        steps=[
            *tokenize_steps,
            no_mixture_dolma_model,
            no_mixture_dolma_evals,
            arxiv_only_subbed_dolma_model,
            arxiv_only_subbed_dolma_evals,
            wiki_only_subbed_dolma_model,
            wiki_only_subbed_dolma_evals,
            wiki_and_arxiv_subbed_dolma_model,
            wiki_and_arxiv_subbed_dolma_evals,
        ]
    )
