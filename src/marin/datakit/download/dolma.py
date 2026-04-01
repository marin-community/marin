# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dolma 1.7 dataset download definition and split metadata."""

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_spec import StepSpec

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


def download_dolma_step() -> StepSpec:
    """Download the Dolma 1.7 dataset from HuggingFace."""
    return download_hf_step(
        "raw/dolma",
        hf_dataset_id="allenai/dolma",
        revision="7f48140",
        override_output_path="raw/dolma",
    )
