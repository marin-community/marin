# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0
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

This module provides library-owned helpers for standard validation sets used by
default training flows (Paloma and Uncheatable Eval), without dependencies on
experiments/.
"""

import os
from functools import lru_cache

from levanter.data.text import TextLmDatasetFormat
from marin.download.uncheatable_eval.download import make_uncheatable_eval_step
from marin.execution.executor import ExecutorStep, ensure_versioned, this_output_path
from marin.processing.tokenize import TokenizeConfig, TokenizerStep, tokenize

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

# Complete mapping of available Uncheatable Eval datasets.
# Reference: https://github.com/Jellyfish042/uncheatable_eval
ALL_UNCHEATABLE_EVAL_DATASETS = {
    "wikipedia_arabic": "wikipedia_arabic_*.jsonl.gz",
    "wikipedia_english": "wikipedia_english_*.jsonl.gz",
    "wikipedia_french": "wikipedia_french_*.jsonl.gz",
    "wikipedia_german": "wikipedia_german_*.jsonl.gz",
    "wikipedia_japanese": "wikipedia_japanese_*.jsonl.gz",
    "wikipedia_spanish": "wikipedia_spanish_*.jsonl.gz",
    "github_python": "github_python_*.jsonl.gz",
    "github_cpp": "github_cpp_*.jsonl.gz",
    "bbc_news": "bbc_news_*.jsonl.gz",
    "arxiv_physics": "arxiv_physics_*.jsonl.gz",
    "arxiv_computer_science": "arxiv_computer_science_*.jsonl.gz",
    "ao3_chinese": "ao3_chinese_*.jsonl.gz",
    "ao3_english": "ao3_english_*.jsonl.gz",
}

# Keep parity with experiments defaults: only English + code-oriented subsets.
ACTIVE_UNCHEATABLE_EVAL_DATASETS = (
    "wikipedia_english",
    "github_python",
    "github_cpp",
    "bbc_news",
    "arxiv_physics",
    "arxiv_computer_science",
    "ao3_english",
)

UNCHEATABLE_EVAL_RAW = make_uncheatable_eval_step()


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


def uncheatable_eval_tokenized(
    *,
    base_path: str = "tokenized/",
    tokenizer: str = "meta-llama/Meta-Llama-3.1-8B",
    uncheatable_eval_raw: ExecutorStep | None = None,
) -> dict[str, TokenizerStep]:
    """
    Return tokenization steps for active Uncheatable Eval datasets.

    Args:
        base_path: Base path prefix for output (prepended to "uncheatable_eval/{dataset}").
        tokenizer: HuggingFace tokenizer name to use.
        uncheatable_eval_raw: Optional ExecutorStep pointing to raw Uncheatable Eval data.
            If None, a default download step is used.

    Returns:
        Dictionary mapping "uncheatable_eval/{dataset}" keys to TokenizerStep instances.
    """
    if uncheatable_eval_raw is None:
        uncheatable_eval_raw = UNCHEATABLE_EVAL_RAW

    uncheatable_eval_steps: dict[str, TokenizerStep] = {}
    for dataset in ACTIVE_UNCHEATABLE_EVAL_DATASETS:
        path_part = ALL_UNCHEATABLE_EVAL_DATASETS[dataset]
        name = os.path.join("uncheatable_eval", dataset)
        step = ExecutorStep(
            name=os.path.join(base_path, name),
            description=f"Tokenize Uncheatable Eval {dataset} using the {tokenizer} tokenizer.",
            fn=tokenize,
            config=TokenizeConfig(
                train_paths=[],
                validation_paths=[uncheatable_eval_raw.cd(path_part)],
                cache_path=this_output_path(),
                tokenizer=ensure_versioned(tokenizer),
                format=TextLmDatasetFormat(),
                sample_count=None,
            ),
        )
        uncheatable_eval_steps[name] = step

    return uncheatable_eval_steps


@lru_cache
def default_validation_sets(tokenizer: str, base_path: str = "tokenized/") -> dict[str, TokenizerStep]:
    """
    Return the default validation suites used by default training flows.

    This matches experiments parity: Paloma + active Uncheatable Eval datasets.
    """
    validation_sets = dict(paloma_tokenized(base_path=base_path, tokenizer=tokenizer))
    validation_sets.update(uncheatable_eval_tokenized(base_path=base_path, tokenizer=tokenizer))
    return validation_sets
