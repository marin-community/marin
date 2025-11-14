# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This experiment compares different settings for converting Arxiv HTML to text for training.
We use Resiliparse to extract the main content from the Arxiv HTML dumps and then convert the text
to markdown using markdownify. We generate 4 variants with combinations of references and links
removed or kept.

We then train 1.4B parameter models on these modified datasets to evaluate which preprocessing
approach produces better training data for language models. Idea is to see if references and links
are important for the model training or if they are just noise.

Reference Issue: https://github.com/marin-community/marin/issues/648
"""

import logging

from experiments.pretraining_datasets import tokenize_dolma_steps
from experiments.exp579_ar5iv_markdownify import (
    ar5iv_no_problem_resiliparse_no_references_no_links,
    ar5iv_no_problem_resiliparse_no_references_with_links,
    ar5iv_no_problem_resiliparse_with_references_no_links,
    ar5iv_no_problem_resiliparse_with_references_with_links,
)
from experiments.html2text.utils import dolma_pipeline_with_modified_dataset
from marin.execution.executor import executor_main

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
    tokenization_steps = [
        *list(tokenized_dolma_steps.values()),
        arxiv_no_problem_resiliparse_no_references_no_links_tokenized,
        arxiv_no_problem_resiliparse_no_references_with_links_tokenized,
        arxiv_no_problem_resiliparse_with_references_no_links_tokenized,
        arxiv_no_problem_resiliparse_with_references_with_links_tokenized,
    ]

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
