# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Simple single-corpus dataset definitions and tokenization.

This module defines raw dataset downloads and their tokenized versions
for simple datasets that don't have multiple splits.
"""

import os.path

from levanter.data.text import TextLmDatasetFormat
from levanter.store.cache import CacheOptions
from marin.datakit.download.pretraining import (
    dclm_baseline_download,
    dclm_baseline_wrong_download,
    dolma3_mix_150b_1025_download,
    fineweb_download,
    fineweb_edu_download,
    proofpile_2_download,
    slimpajama_6b_download,
    slimpajama_download,
    starcoderdata_download,
    the_pile_openwebtext2_download,
    the_stack_dedup_download,
)
from marin.execution.executor import ExecutorStep, InputName, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize

from experiments.llama import llama3_tokenizer

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _tokenize_simple(
    name: str,
    raw_dataset: ExecutorStep | InputName,
    tokenizer: str | None = None,
    override_path: str | None = None,
    text_format: TextLmDatasetFormat = TextLmDatasetFormat(),
    cache_options: CacheOptions | None = None,
) -> ExecutorStep[TokenizeConfig]:
    """Helper to create a simple tokenized dataset."""

    config = TokenizeConfig(
        train_paths=[raw_dataset],
        validation_paths=versioned([]),
        cache_path=this_output_path(),
        tokenizer=versioned(tokenizer),
        format=text_format,
    )

    step = ExecutorStep(
        name=os.path.join("tokenized", name),
        fn=tokenize,
        config=config,
    )

    if override_path is not None:
        step = step.with_output_path(override_path)

    return step


# ============================================================================
# RAW DATASET DOWNLOADS
# ============================================================================


def _build_downloads() -> dict[str, ExecutorStep | InputName]:
    """Build the downloads dict from canonical StepSpec definitions in pretraining.py."""
    fineweb_edu_base = fineweb_edu_download().as_executor_step()

    return {
        "fineweb": fineweb_download().as_executor_step(),
        "fineweb_edu": fineweb_edu_base.cd("data"),
        "fineweb_edu_sample_10bt": fineweb_edu_base.cd("sample/10BT"),
        "fineweb_edu_sample_100bt": fineweb_edu_base.cd("sample/100BT"),
        "fineweb_edu_sample_350bt": fineweb_edu_base.cd("sample/350BT"),
        "slimpajama": (
            slimpajama_download()
            .as_executor_step()
            .cd("2d0accd/huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/2d0accd")
        ),
        "slimpajama_6b": slimpajama_6b_download().as_executor_step().cd("data"),
        "dolma3_mix_150b_1025": dolma3_mix_150b_1025_download().as_executor_step().cd("15d04ee"),
        "dclm_baseline_wrong": dclm_baseline_wrong_download().as_executor_step(),
        "dclm_baseline": dclm_baseline_download().as_executor_step().cd("a3b142c"),
        "the_stack_dedup": the_stack_dedup_download().as_executor_step().cd("17cad72"),
        "proofpile_2": (
            proofpile_2_download()
            .as_executor_step()
            .cd("901a927/huggingface.co/datasets/EleutherAI/proof-pile-2/resolve/901a927")
        ),
        "the_pile_openwebtext2": (
            the_pile_openwebtext2_download()
            .as_executor_step()
            .cd("1de27c6/huggingface.co/datasets/vietgpt/the_pile_openwebtext2/resolve/1de27c6")
        ),
        "starcoderdata": starcoderdata_download().as_executor_step(),
    }


downloads = _build_downloads()


# ============================================================================
# TOKENIZED DATASETS
# ============================================================================

tokenized = {
    "dclm_baseline": _tokenize_simple(
        "dclm_baseline",
        downloads["dclm_baseline"],
        tokenizer=llama3_tokenizer,
        override_path="tokenized/dclm_baseline-0206f1/",
    ),
    "starcoderdata": _tokenize_simple(
        "starcoderdata",
        downloads["starcoderdata"],
        tokenizer=llama3_tokenizer,
        text_format=TextLmDatasetFormat(text_key="content"),
        override_path="tokenized/starcoderdata-12f018/",
    ),
    "proofpile_2": _tokenize_simple(
        "proofpile_2",
        downloads["proofpile_2"],
        tokenizer=llama3_tokenizer,
        override_path="tokenized/proofpile_2-4a35c7/",
    ),
    "slimpajama_6b": _tokenize_simple(
        "SlimPajama-6B",
        downloads["slimpajama_6b"],
        tokenizer=llama3_tokenizer,
    ),
    "fineweb_edu": _tokenize_simple(
        "fineweb-edu",
        downloads["fineweb_edu"],
        tokenizer=llama3_tokenizer,
    ),
    "dolma3_mix_150b_1025": _tokenize_simple(
        "dolma3_mix-150B-1025",
        downloads["dolma3_mix_150b_1025"],
        tokenizer=llama3_tokenizer,
        override_path="tokenized/dolma3_mix-150B-1025-15d04ee/",
    ),
}
