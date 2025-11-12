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
This experiment compares different methods for converting Wikipedia HTML to text for training.
We substitute the Wikipedia portion of the Dolma dataset with our own processed Wikipedia data using:
1. Readability which extracts simplifies HTML DOM tree and is then combined with markdownify
  for markdown conversion.
2. Resiliparse with preserve_formatting, which removes boilerplate but preserves text formatting

We then train 1.4B parameter models on these modified datasets to evaluate which preprocessing
approach produces better training data for language models. Idea is to see which main content
extraction method improves the training for the models.

Reference Issue: https://github.com/marin-community/marin/issues/647
"""

import logging

from experiments.dolma.tokenize_dolma import tokenize_dolma_steps
from experiments.exp575_wikipedia_markdownify import wikipedia_readability, wikipedia_resiliparse_with_pf
from experiments.html2text.utils import dolma_pipeline_with_modified_dataset
from marin.execution.executor import executor_main

logger = logging.getLogger("ray")

EXPERIMENT_TAG = ["wiki_subbed_dolma"]

tokenized_dolma_steps = tokenize_dolma_steps()

# Conduct the experiment with readability
wiki_readability_tokenized, wiki_readability_1_4b_model, wiki_readability_1_4b_evals = (
    dolma_pipeline_with_modified_dataset(
        path_prefix="wiki-readability",
        dataset=wikipedia_readability,
        dolma_dataset="wiki",
        experiment_tag=EXPERIMENT_TAG,
    )
)

# Conduct the experiment with preserving formatting
(
    wiki_resiliparse_with_preserve_formatting_tokenized,
    wiki_resiliparse_with_preserve_formatting_1_4b_model,
    wiki_resiliparse_with_preserve_formatting_1_4b_evals,
) = dolma_pipeline_with_modified_dataset(
    path_prefix="wiki-resiliparse-with-preserving-formatting",
    dataset=wikipedia_resiliparse_with_pf,
    dolma_dataset="wiki",
    experiment_tag=EXPERIMENT_TAG,
)

if __name__ == "__main__":
    tokenization_step = [
        *list(tokenized_dolma_steps.values()),
        wiki_readability_tokenized,
        wiki_resiliparse_with_preserve_formatting_tokenized,
    ]

    executor_main(
        steps=[
            *tokenization_step,
            wiki_readability_1_4b_model,
            wiki_readability_1_4b_evals,
            wiki_resiliparse_with_preserve_formatting_1_4b_model,
            wiki_resiliparse_with_preserve_formatting_1_4b_evals,
        ]
    )
