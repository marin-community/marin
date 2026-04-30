# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DOLMA 1.7 dataset definitions and tokenization."""

import os.path

from marin.execution.executor import ExecutorStep, InputName, this_output_path, versioned
from marin.datakit.download.dolma import DOLMA_DATASETS, download_dolma_step
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

_dolma_download = download_dolma_step().as_executor_step()

# Backward compat — some consumers import this
downloads = {"dolma": _dolma_download}

# Sampling proportion comes from https://huggingface.co/datasets/allenai/dolma
DOLMA_OLMO_MIXTURE_WEIGHTS = {
    "dolma/algebraic-stack": 12.6,
    "dolma/arxiv": 28.0,
    "dolma/gutenberg": 5.3,
    "dolma/c4": 124.95,
    "dolma/cc": 597.75,
    "dolma/cc-news": 14.3,
    "dolma/falcon": 456.4,
    "dolma/megawika": 4.6,
    "dolma/open-web-math": 12.6,
    "dolma/pes2o": 57.2,
    "dolma/reddit": 79.9,
    "dolma/stackexchange": 19.6,
    "dolma/starcoder": 263.8,
    "dolma/flan": 16.5,
    "dolma/wiki": 7.4,
}

# For dolma 1.7, we hardcode the path since it was added before versioning
_DOLMA_V1_7_PATH = InputName.hardcoded("raw/dolma/v1.7")

# NB: we changed how hashes were computed for this corpus and we'd like to avoid recomputing them
DOLMA_LLAMA3_OVERRIDES = {
    "c4": "tokenized/dolma/c4-e0e5ec",
    "cc": "tokenized/dolma/cc-74b017",
    "cc-news": "tokenized/dolma/cc-news-625d3e",
    "falcon": "tokenized/dolma/falcon-da8fd0",
    "flan": "tokenized/dolma/flan-a99cb2",
    "gutenberg": "tokenized/dolma/gutenberg-f9eb99",
    "reddit": "tokenized/dolma/reddit-62a64a",
    "starcoder": "tokenized/dolma/starcoder-8b6089",
    "algebraic-stack": "tokenized/dolma/algebraic-stack-cc00cf",
    "arxiv": "tokenized/dolma/arxiv-07a51f",
    "megawika": "tokenized/dolma/megawika-34abf2",
    "open-web-math": "tokenized/dolma/open-web-math-79823d",
    "pes2o": "tokenized/dolma/pes2o-538363",
    "stackexchange": "tokenized/dolma/stackexchange-adfc49",
    "wiki": "tokenized/dolma/wiki-212315",
}


def tokenize_dolma(*, tokenizer: str | None = None) -> dict[str, TokenizerStep]:
    """Generate tokenization steps for all Dolma 1.7 dataset splits."""
    from experiments.llama import llama3_tokenizer

    if tokenizer is None:
        tokenizer = llama3_tokenizer

    dolma_steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for dataset, files in DOLMA_DATASETS.items():
        step = ExecutorStep(
            name=os.path.join("tokenized", "dolma", dataset),
            fn=tokenize,
            config=TokenizeConfig(
                train_paths=[_DOLMA_V1_7_PATH / file for file in files],
                validation_paths=versioned([]),
                cache_path=this_output_path(),
                tokenizer=versioned(tokenizer),
            ),
        )

        if tokenizer == llama3_tokenizer and dataset in DOLMA_LLAMA3_OVERRIDES:
            step = step.with_output_path(DOLMA_LLAMA3_OVERRIDES[dataset])
        dolma_steps[os.path.join("dolma", dataset)] = step

    return dolma_steps
