"""
Test different html->text transformation methods (on Ar5iv Dump, train 1.4B models).
https://github.com/stanford-crfm/marin/issues/648
"""

import logging

from experiments.defaults import default_tokenize, default_train
from experiments.dolma.tokenize_dolma import DOLMA_OLMO_MIXTURE_WEIGHTS, tokenize_dolma_steps
from experiments.evals.evals import default_eval
from experiments.exp579_ar5iv_markdownify import (
    ar5iv_no_problem_resiliparse_no_references_no_links,
    ar5iv_no_problem_resiliparse_no_references_with_links,
    ar5iv_no_problem_resiliparse_with_references_no_links,
    ar5iv_no_problem_resiliparse_with_references_with_links,
)
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ray")

EXPERIMENT_TAG = ["arxiv_no_problem_subbed_dolma"]

tokenized_dolma_steps = tokenize_dolma_steps()

arxiv_no_problem_resiliparse_no_references_no_links_tokenized = default_tokenize(
    name="dolma-arxiv-no-problem-resiliparse-no-references-no-links",
    dataset=ar5iv_no_problem_resiliparse_no_references_no_links,
    tokenizer=llama3_tokenizer,
)

dolma_arxiv_no_problem_resiliparse_no_references_no_links_tokenization_steps = dict(
    tokenized_dolma_steps,
    {
        "arxiv-no-problem-resiliparse-no-references-no-links": arxiv_no_problem_resiliparse_no_references_no_links_tokenized  # noqa
    },
)
dolma_arxiv_no_problem_resiliparse_no_references_no_links_weights = dict(
    DOLMA_OLMO_MIXTURE_WEIGHTS,
    {"arxiv-no-problem-resiliparse-no-references-no-links": DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/arxiv"]},
)

arxiv_no_problem_resiliparse_no_references_no_links_llama3_tokenized = lm_mixture_data_config(
    components=dolma_arxiv_no_problem_resiliparse_no_references_no_links_tokenization_steps,
    weights=dolma_arxiv_no_problem_resiliparse_no_references_no_links_weights,
)

arxiv_no_problem_resiliparse_1_4b_no_references_no_links_model = default_train(
    name="dolma-arxiv_no_problem-subbed-1.4b-resiliparse-no-references-no-links",
    tokenized=arxiv_no_problem_resiliparse_no_references_no_links_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG,
)

arxiv_no_problem_resiliparse_1_4b_no_references_no_links_evals = default_eval(
    step=arxiv_no_problem_resiliparse_1_4b_no_references_no_links_model
)


arxiv_no_problem_resiliparse_no_references_with_links_tokenized = default_tokenize(
    name="dolma-arxiv-no-problem-resiliparse-no-references-with-links",
    dataset=ar5iv_no_problem_resiliparse_no_references_with_links,
    tokenizer=llama3_tokenizer,
)

dolma_arxiv_no_problem_resiliparse_no_references_with_links_tokenization_steps = dict(
    tokenized_dolma_steps,
    {
        "arxiv-no-problem-resiliparse-no-references-with-links": arxiv_no_problem_resiliparse_no_references_with_links_tokenized  # noqa
    },
)
dolma_arxiv_no_problem_resiliparse_no_references_with_links_weights = dict(
    DOLMA_OLMO_MIXTURE_WEIGHTS,
    {"arxiv-no-problem-resiliparse-no-references-with-links": DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/arxiv"]},
)
arxiv_no_problem_resiliparse_with_pf_subbed_no_references_with_links_llama3_tokenized = lm_mixture_data_config(
    components=dolma_arxiv_no_problem_resiliparse_no_references_with_links_tokenization_steps,
    weights=dolma_arxiv_no_problem_resiliparse_no_references_with_links_weights,
)
arxiv_no_problem_resiliparse_with_pf_subbed_no_references_with_links_dolma_1_4b_model = default_train(
    name="dolma-arxiv_no_problem-subbed-1.4b-resiliparse-with-pf-no-references-with-links",
    tokenized=arxiv_no_problem_resiliparse_with_pf_subbed_no_references_with_links_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG,
)

arxiv_no_problem_resiliparse_with_pf_subbed_no_references_with_links_dolma_1_4b_evals = default_eval(
    step=arxiv_no_problem_resiliparse_with_pf_subbed_no_references_with_links_dolma_1_4b_model
)


arxiv_no_problem_resiliparse_with_references_no_links_tokenized = default_tokenize(
    name="dolma-arxiv-no-problem-resiliparse-with-references-no-links",
    dataset=ar5iv_no_problem_resiliparse_with_references_no_links,
    tokenizer=llama3_tokenizer,
)

dolma_arxiv_no_problem_resiliparse_with_references_no_links_tokenization_steps = dict(
    tokenized_dolma_steps,
    {
        "arxiv-no-problem-resiliparse-with-references-no-links": arxiv_no_problem_resiliparse_with_references_no_links_tokenized  # noqa
    },
)
dolma_arxiv_no_problem_resiliparse_with_references_no_links_weights = dict(
    DOLMA_OLMO_MIXTURE_WEIGHTS,
    {"arxiv-no-problem-resiliparse-with-references-no-links": DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/arxiv"]},
)
arxiv_no_problem_resiliparse_with_pf_subbed_no_references_no_links_llama3_tokenized = lm_mixture_data_config(
    components=dolma_arxiv_no_problem_resiliparse_with_references_no_links_tokenization_steps,
    weights=dolma_arxiv_no_problem_resiliparse_with_references_no_links_weights,
)
arxiv_no_problem_resiliparse_with_pf_subbed_no_references_no_links_1_4b_model = default_train(
    name="dolma-arxiv_no_problem-subbed-1.4b-resiliparse-with-pf-no-references-no-links",
    tokenized=arxiv_no_problem_resiliparse_with_pf_subbed_no_references_no_links_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG,
)

arxiv_no_problem_resiliparse_with_pf_subbed_no_references_no_links_1_4b_evals = default_eval(
    step=arxiv_no_problem_resiliparse_with_pf_subbed_no_references_no_links_1_4b_model
)


arxiv_no_problem_resiliparse_with_references_with_links_tokenized = default_tokenize(
    name="dolma-arxiv-no-problem-resiliparse-with-references-with-links",
    dataset=ar5iv_no_problem_resiliparse_with_references_with_links,
    tokenizer=llama3_tokenizer,
)

dolma_arxiv_no_problem_resiliparse_with_references_with_links_tokenization_steps = dict(
    tokenized_dolma_steps,
    {
        "arxiv-no-problem-resiliparse-with-references-with-links": arxiv_no_problem_resiliparse_with_references_with_links_tokenized  # noqa
    },
)
dolma_arxiv_no_problem_resiliparse_with_references_with_links_weights = dict(
    DOLMA_OLMO_MIXTURE_WEIGHTS,
    {"arxiv-no-problem-resiliparse-with-references-with-links": DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/arxiv"]},
)
arxiv_no_problem_resiliparse_with_pf_subbed_no_references_with_links_llama3_tokenized = lm_mixture_data_config(
    components=dolma_arxiv_no_problem_resiliparse_with_references_with_links_tokenization_steps,
    weights=dolma_arxiv_no_problem_resiliparse_with_references_with_links_weights,
)
arxiv_no_problem_resiliparse_with_pf_subbed_no_references_with_links_1_4b_model = default_train(
    name="dolma-arxiv_no_problem-subbed-1.4b-resiliparse-with-pf-no-references-with-links",
    tokenized=arxiv_no_problem_resiliparse_with_pf_subbed_no_references_with_links_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG,
)

arxiv_no_problem_resiliparse_with_pf_subbed_no_references_with_links_1_4b_evals = default_eval(
    step=arxiv_no_problem_resiliparse_with_pf_subbed_no_references_with_links_1_4b_model
)

if __name__ == "__main__":
    tokenization_steps = (
        list(tokenized_dolma_steps.values())
        + list(dolma_arxiv_no_problem_resiliparse_no_references_no_links_tokenization_steps.values())
        + list(dolma_arxiv_no_problem_resiliparse_no_references_with_links_tokenization_steps.values())
        + list(dolma_arxiv_no_problem_resiliparse_with_references_with_links_tokenization_steps.values())
    )

    executor_main(
        steps=[
            *tokenization_steps,
            arxiv_no_problem_resiliparse_1_4b_no_references_no_links_model,
            arxiv_no_problem_resiliparse_1_4b_no_references_no_links_evals,
            arxiv_no_problem_resiliparse_with_pf_subbed_no_references_with_links_1_4b_model,
            arxiv_no_problem_resiliparse_with_pf_subbed_no_references_with_links_1_4b_evals,
            arxiv_no_problem_resiliparse_with_pf_subbed_no_references_no_links_1_4b_model,
            arxiv_no_problem_resiliparse_with_pf_subbed_no_references_no_links_1_4b_evals,
            arxiv_no_problem_resiliparse_with_pf_subbed_no_references_with_links_1_4b_model,
            arxiv_no_problem_resiliparse_with_pf_subbed_no_references_with_links_1_4b_evals,
        ]
    )
