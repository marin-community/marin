# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
FinePDFs dataset definitions and tokenization.

HuggingFaceFW/finepdfs is a multilingual PDF text extraction dataset with 100+ language
configs organized as data/{lang_code}_{script}/train/*.parquet. Each language subset is
downloaded independently via hf_urls_glob to avoid pulling the entire (very large) repo.
"""

import os.path

from levanter.data.text import TextLmDatasetFormat

from experiments.llama import llama3_tokenizer
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

FINEPDFS_HF_ID = "HuggingFaceFW/finepdfs"
FINEPDFS_REVISION = "89f5411"

# Language subsets to expose. Add entries here to make new languages available.
FINEPDFS_LANGUAGES = [
    "eng_Latn",
]


def _download_language(lang: str) -> ExecutorStep:
    return ExecutorStep(
        name=f"raw/finepdfs/{lang}",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id=FINEPDFS_HF_ID,
            revision=FINEPDFS_REVISION,
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
            hf_urls_glob=[f"data/{lang}/*/*.parquet"],
        ),
    )


downloads = {lang: _download_language(lang) for lang in FINEPDFS_LANGUAGES}


def _tokenize_language(
    lang: str,
    tokenizer: str | None = None,
) -> ExecutorStep[TokenizeConfig]:
    if tokenizer is None:
        tokenizer = llama3_tokenizer

    raw = downloads[lang]
    return ExecutorStep(
        name=os.path.join("tokenized", "finepdfs", lang),
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[raw.cd(f"data/{lang}/train")],
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(tokenizer),
            format=TextLmDatasetFormat(text_key="text"),
        ),
    )


def tokenize_finepdfs(*, tokenizer: str | None = None) -> dict[str, TokenizerStep]:
    """Generate tokenization steps for all configured FinePDFs language subsets."""
    return {f"finepdfs/{lang}": _tokenize_language(lang, tokenizer=tokenizer) for lang in FINEPDFS_LANGUAGES}


tokenized = {lang: _tokenize_language(lang) for lang in FINEPDFS_LANGUAGES}
