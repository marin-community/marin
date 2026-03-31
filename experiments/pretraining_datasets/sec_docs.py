# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SEC filings dataset definitions and tokenization."""

import os.path

from levanter.data.text import TextLmDatasetFormat
from marin.datakit.download.huggingface import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, output_path_of, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

# Raw dataset download step
downloads = {
    "sec_docs": ExecutorStep(
        name="raw/cleaned_sec_docs",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id="chiheemwong/cleaned_sec_docs",
            revision="14108db",
            hf_urls_glob=["data/*/*.parquet"],
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
        ),
        override_output_path="raw/cleaned_sec_docs-14108db",
    )
}

SEC_DOCS_DATASETS = {
    "10k": ["data/10K/train-*.parquet"],
    "10q": ["data/10Q/train-*.parquet"],
    "8k": ["data/8K/train-*.parquet"],
    "s4": ["data/S4/train-*.parquet"],
}


def _get_sec_docs_subset_paths(subset: str) -> list[str]:
    """Helper to get file paths for a SEC docs subset."""
    patterns = SEC_DOCS_DATASETS[subset]
    return [output_path_of(downloads["sec_docs"], pattern) for pattern in patterns]


def tokenize_sec_docs(
    *,
    tokenizer: str | None = None,
    max_workers: int = 4096,
    writer_batch_size: int = 65536,
) -> dict[str, TokenizerStep]:
    """Generate tokenization steps for all SEC docs subsets."""
    if tokenizer is None:
        from experiments.llama import llama3_tokenizer

        tokenizer = llama3_tokenizer

    sec_docs_steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for subset in SEC_DOCS_DATASETS:
        sec_docs_subset_paths = _get_sec_docs_subset_paths(subset)
        step = ExecutorStep(
            name=os.path.join("tokenized", "sec_docs", subset),
            fn=tokenize,
            config=TokenizeConfig(
                train_paths=sec_docs_subset_paths,
                validation_paths=versioned([]),
                cache_path=this_output_path(),
                tokenizer=versioned(tokenizer),
                format=TextLmDatasetFormat(text_key="renderedText"),
                max_workers=max_workers,
                writer_batch_size=writer_batch_size,
            ),
        )
        sec_docs_steps[os.path.join("sec_docs", subset)] = step

    return sec_docs_steps


def tokenize_sec_docs_subset(name: str, tokenizer: str | None = None) -> ExecutorStep[TokenizeConfig]:
    """Get a specific SEC docs subset tokenization step."""
    assert name in SEC_DOCS_DATASETS, f"Subset {name} not found in SEC_DOCS_DATASETS"
    return tokenize_sec_docs(tokenizer=tokenizer)[f"sec_docs/{name}"]
