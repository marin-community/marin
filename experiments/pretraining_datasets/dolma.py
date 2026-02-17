# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
DOLMA 1.7 dataset definitions and tokenization.

This module defines the raw DOLMA dataset download and tokenization
logic for all 15 splits.
"""

import os
import os.path

from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.step_model import StepSpec
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

# Raw dataset download step
downloads = {
    "dolma": StepSpec(
        name="raw/dolma",
        hash_attrs={"hf_dataset_id": "allenai/dolma", "revision": "7f48140"},
        override_output_path="raw/dolma",
        fn=lambda output_path: download_hf(
            DownloadConfig(
                hf_dataset_id="allenai/dolma",
                revision="7f48140",
                gcs_output_path=output_path,
                wait_for_completion=True,
            )
        ),
    )
}


# For dolma 1.7, we hardcode the path since it was added before versioning
_DOLMA_V1_7_PATH = "raw/dolma/v1.7"


# Sampling proportion comes from https://huggingface.co/datasets/allenai/dolma
DOLMA_OLMO_MIXTURE_WEIGHTS = {
    "dolma/algebraic-stack": 12.6,  # 12.6 * 1.0
    "dolma/arxiv": 28.0,  # 28.0 * 1.0
    "dolma/gutenberg": 5.3,  # 5.3 * 1.0
    "dolma/c4": 124.95,  # 249.9 * 0.5
    "dolma/cc": 597.75,  # 1,195.5 * 0.5
    "dolma/cc-news": 14.3,  # 1.0
    "dolma/falcon": 456.4,  # 1.0, refined web
    "dolma/megawika": 4.6,  # 1.0
    "dolma/open-web-math": 12.6,  # 1.0
    "dolma/pes2o": 57.2,  # 1.0
    "dolma/reddit": 79.9,  # 1.0
    "dolma/stackexchange": 19.6,  # 1.0
    "dolma/starcoder": 263.8,  # 1.0
    "dolma/flan": 16.5,  # 6.5 * 1.0
    "dolma/wiki": 7.4,  # 3.7 * 2.0
}

DOLMA_DATASETS = {
    "algebraic-stack": ["algebraic-stack-train-{0000..0015}.json.gz"],
    "arxiv": ["arxiv-{0000..0099}.json.gz"],
    "gutenberg": ["books-{0000..0002}.json.gz"],
    "c4": ["c4-{0000..0170}.json.gz"],
    "cc": [
        "cc_en_head-{0000..0274}.json.gz",
        "cc_en_middle-{0000..0238}.json.gz",
        "cc_en_middle-{0240..0379}.json.gz",
        "cc_en_tail-{0000..0152}.json.gz",
        "cc_en_tail-{0154..0444}.json.gz",
    ],
    "cc-news": ["cc_news_head-{0000..0004}.json.gz", "cc_news_middle-{0000..0002}.json.gz", "cc_news_tail-0000.json.gz"],
    "falcon": ["falcon-{0000..0499}.json.gz"],
    "megawika": ["megawika-{0000..0261}.json.gz"],
    "open-web-math": ["open-web-math-train-{0000..0012}.json.gz"],
    "pes2o": ["pes2o-{0000..0025}.json.gz"],
    "reddit": ["reddit-{0000..0077}.json.gz"],
    "stackexchange": ["stackexchange-{0000..0025}.json.gz"],
    "starcoder": ["starcoder-{0000..0048}.json.gz"],
    "flan": ["tulu_flan-{0000..0065}.json.gz"],
    "wiki": ["wiki-{0000..0001}.json.gz"],
}

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

    dolma_steps: dict[str, StepSpec] = {}
    for dataset, files in DOLMA_DATASETS.items():
        train_paths = [os.path.join(_DOLMA_V1_7_PATH, file) for file in files]
        override = None
        if tokenizer == llama3_tokenizer and dataset in DOLMA_LLAMA3_OVERRIDES:
            override = DOLMA_LLAMA3_OVERRIDES[dataset]

        # Capture loop variables for the closure
        _train_paths = train_paths
        _tokenizer = tokenizer

        step = StepSpec(
            name=os.path.join("tokenized", "dolma", dataset),
            hash_attrs={"tokenizer": tokenizer, "validation_paths": []},
            override_output_path=override,
            fn=lambda output_path, _tp=_train_paths, _tk=_tokenizer: tokenize(
                TokenizeConfig(
                    train_paths=_tp,
                    validation_paths=[],
                    cache_path=output_path,
                    tokenizer=_tk,
                )
            ),
        )

        dolma_steps[os.path.join("dolma", dataset)] = step

    return dolma_steps
