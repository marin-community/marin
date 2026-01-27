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
Validation dataset configurations for Marin.

This module provides configurations for standard validation sets like Paloma,
without dependencies on experiments/.
"""

import os

from levanter.data.text import TextLmDatasetFormat
from marin.execution.executor import ExecutorStep, ensure_versioned, this_output_path
from marin.processing.tokenize import TokenizeConfig, TokenizerStep

# The datasets in the Paloma eval set and their paths within the HF dataset
# https://huggingface.co/datasets/allenai/paloma
PALOMA_DATASETS_TO_DIR = {
    "4chan": "4chan_meta_sep",
    "c4_100_domains": "c4_100_domains",
    "c4_en": "c4_en",
    "dolma-v1_5": "dolma-v1_5",
    "dolma_100_programing_languages": "dolma_100_programing_languages",
    "dolma_100_subreddits": "dolma_100_subreddits",
    "falcon-refinedweb": "falcon-refinedweb",
    "gab": "gab",
    "m2d2_s2orc_unsplit": "m2d2_s2orc_unsplit",
    "m2d2_wikipedia_unsplit": "m2d2_wikipedia_unsplit",
    "manosphere_meta_sep": "manosphere_meta_sep",
    "mc4": "mc4",
    "ptb": "ptb",
    "redpajama": "redpajama",
    "twitterAAE_HELM_fixed": "twitterAAE_HELM_fixed",
    "wikitext_103": "wikitext_103",
}


def paloma_tokenized(
    *,
    base_path: str = "tokenized/",
    tokenizer: str = "meta-llama/Meta-Llama-3.1-8B",
    paloma_raw: ExecutorStep | None = None,
) -> dict[str, TokenizerStep]:
    """
    Returns a dictionary of steps to tokenize the Paloma eval sets.

    This function creates tokenization steps for all 16 Paloma validation datasets
    without depending on experiments/ code.

    Args:
        base_path: Base path prefix for output (prepended to "paloma/{dataset}").
        tokenizer: HuggingFace tokenizer name to use.
        paloma_raw: Optional ExecutorStep pointing to raw Paloma data. If None, uses default path pattern.

    Returns:
        Dictionary mapping "paloma/{dataset}" keys to TokenizerStep instances.
    """
    from marin.processing.tokenize import tokenize

    paloma_steps: dict[str, TokenizerStep] = {}

    for dataset, path_part in PALOMA_DATASETS_TO_DIR.items():
        name = os.path.join("paloma", dataset)
        if paloma_raw is not None:
            dataset_input = paloma_raw.cd(f"{path_part}/val/val*.jsonl.gz")
        else:
            dataset_input = f"{path_part}/val/val*.jsonl.gz"

        config = TokenizeConfig(
            train_paths=[],
            validation_paths=[dataset_input],
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=TextLmDatasetFormat(),
            sample_count=None,
        )

        step = ExecutorStep(
            name=os.path.join(base_path, name),
            description=f"Tokenize Paloma {dataset} using the {tokenizer} tokenizer.",
            fn=tokenize,
            config=config,
        )

        paloma_steps[name] = step

    return paloma_steps
