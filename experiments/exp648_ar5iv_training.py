"""
This experiment compares different settings for converting Arxiv HTML to text for training.
We use Resiliparse to extract the main content from the Arxiv HTML dumps and then convert the text
to markdown using markdownify. We generate 4 variants with combinations of references and links
removed or kept.

We then train 1.4B parameter models on these modified datasets to evaluate which preprocessing
approach produces better training data for language models. Idea is to see if references and links
are important for the model training or if they are just noise.

Reference Issue: https://github.com/stanford-crfm/marin/issues/648
"""

import logging

from experiments.dolma.tokenize_dolma import tokenize_dolma_steps
from experiments.dolma.utils import dolma_pipeline_with_modified_dataset
from experiments.exp579_ar5iv_markdownify import (
    ar5iv_no_problem_resiliparse_no_references_no_links,
    ar5iv_no_problem_resiliparse_no_references_with_links,
    ar5iv_no_problem_resiliparse_with_references_no_links,
    ar5iv_no_problem_resiliparse_with_references_with_links,
)
from marin.execution.executor import executor_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ray")

EXPERIMENT_TAG = ["arxiv_no_problem_subbed_dolma"]

# Tokenize the Dolma dataset
tokenized_dolma_steps = tokenize_dolma_steps()

# Conduct the experiment with no references and no links
(
    arxiv_no_problem_resiliparse_no_references_no_links_tokenized,
    arxiv_no_problem_resiliparse_1_4b_no_references_no_links_model,
    arxiv_no_problem_resiliparse_1_4b_no_references_no_links_evals,
) = dolma_pipeline_with_modified_dataset(
    path_prefix="arxiv-no-problem-resiliparse-no-references-no-links",
    dataset=ar5iv_no_problem_resiliparse_no_references_no_links,
    dolma_dataset="arxiv",
    experiment_tag=EXPERIMENT_TAG,
)

# Conduct the experiment with no references and with links
(
    arxiv_no_problem_resiliparse_no_references_with_links_tokenized,
    arxiv_no_problem_resiliparse_1_4b_no_references_with_links_model,
    arxiv_no_problem_resiliparse_1_4b_no_references_with_links_evals,
) = dolma_pipeline_with_modified_dataset(
    path_prefix="arxiv-no-problem-resiliparse-no-references-with-links",
    dataset=ar5iv_no_problem_resiliparse_no_references_with_links,
    dolma_dataset="arxiv",
    experiment_tag=EXPERIMENT_TAG,
)

# Conduct the experiment with references and no links
(
    arxiv_no_problem_resiliparse_with_references_no_links_tokenized,
    arxiv_no_problem_resiliparse_1_4b_with_references_no_links_model,
    arxiv_no_problem_resiliparse_1_4b_with_references_no_links_evals,
) = dolma_pipeline_with_modified_dataset(
    path_prefix="arxiv-no-problem-resiliparse-with-references-no-links",
    dataset=ar5iv_no_problem_resiliparse_with_references_no_links,
    dolma_dataset="arxiv",
    experiment_tag=EXPERIMENT_TAG,
)

# Conduct the experiment with references and with links
(
    arxiv_no_problem_resiliparse_with_references_with_links_tokenized,
    arxiv_no_problem_resiliparse_1_4b_with_references_with_links_model,
    arxiv_no_problem_resiliparse_1_4b_with_references_with_links_evals,
) = dolma_pipeline_with_modified_dataset(
    path_prefix="arxiv-no-problem-resiliparse-with-references-with-links",
    dataset=ar5iv_no_problem_resiliparse_with_references_with_links,
    dolma_dataset="arxiv",
    experiment_tag=EXPERIMENT_TAG,
)

if __name__ == "__main__":
    tokenization_steps = (
        list(tokenized_dolma_steps.values())
        + list(arxiv_no_problem_resiliparse_no_references_no_links_tokenized.values())
        + list(arxiv_no_problem_resiliparse_no_references_with_links_tokenized.values())
        + list(arxiv_no_problem_resiliparse_with_references_with_links_tokenized.values())
    )

    executor_main(
        steps=[
            *tokenization_steps,
            arxiv_no_problem_resiliparse_1_4b_no_references_no_links_model,
            arxiv_no_problem_resiliparse_1_4b_no_references_no_links_evals,
            arxiv_no_problem_resiliparse_1_4b_no_references_with_links_model,
            arxiv_no_problem_resiliparse_1_4b_no_references_with_links_evals,
            arxiv_no_problem_resiliparse_1_4b_with_references_no_links_model,
            arxiv_no_problem_resiliparse_1_4b_with_references_no_links_evals,
            arxiv_no_problem_resiliparse_1_4b_with_references_with_links_model,
            arxiv_no_problem_resiliparse_1_4b_with_references_with_links_evals,
        ]
    )
