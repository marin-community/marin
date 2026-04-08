# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Simple single-corpus dataset definitions and tokenization.

This module defines raw dataset downloads and their tokenized versions
for simple datasets that don't have multiple splits.
"""
from fray import ResourceConfig
from marin.execution.remote import remote

import os.path

from levanter.data.text import TextLmDatasetFormat
from levanter.store.cache import CacheOptions
from marin.datakit.canonical.fineweb_edu import download as fineweb_edu_download
from marin.datakit.download.huggingface import download_hf_step
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
        # TODO: `tokenize` shouldn't require this much RAM - fix after levanter store consolidation
        fn=remote(tokenize, resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="10g")),
        config=config,
    )

    if override_path is not None:
        step = step.with_output_path(override_path)

    return step


def _dl(
    name: str, hf_dataset_id: str, revision: str, output_path: str, *, append_sha_to_path: bool = False
) -> ExecutorStep:
    """Create a download ExecutorStep from a StepSpec."""
    return download_hf_step(
        name,
        hf_dataset_id=hf_dataset_id,
        revision=revision,
        append_sha_to_path=append_sha_to_path,
        override_output_path=output_path,
    ).as_executor_step()


# ============================================================================
# RAW DATASET DOWNLOADS
# ============================================================================


def _build_downloads() -> dict[str, ExecutorStep | InputName]:
    fineweb_edu_base = fineweb_edu_download().as_executor_step()

    return {
        "fineweb_edu": fineweb_edu_base.cd("data"),
        "fineweb_edu_sample_10bt": fineweb_edu_base.cd("sample/10BT"),
        "fineweb_edu_sample_100bt": fineweb_edu_base.cd("sample/100BT"),
        "fineweb_edu_sample_350bt": fineweb_edu_base.cd("sample/350BT"),
        "slimpajama": (
            _dl("raw/SlimPajama-627B", "cerebras/SlimPajama-627B", "2d0accd", "raw/SlimPajama-627B-262830").cd(
                "2d0accd/huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/2d0accd"
            )
        ),
        "slimpajama_6b": (
            _dl("raw/SlimPajama-6B", "DKYoon/SlimPajama-6B", "b5f90f4", "raw/SlimPajama-6B-be35b7").cd("data")
        ),
        "dolma3_mix_150b_1025": (
            _dl(
                "raw/dolma3_mix-150B-1025",
                "allenai/dolma3_mix-150B-1025",
                "15d04ee",
                "raw/dolma3_mix-150B-1025-15d04ee",
                append_sha_to_path=True,
            ).cd("15d04ee")
        ),
        "dclm_baseline_wrong": _dl(
            "raw/dclm-baseline-1.0", "mlfoundations/dclm-baseline-1.0", "a3b142c", "raw/dclm_WRONG_20250211/"
        ),
        "dclm_baseline": (
            _dl("raw/dclm-baseline-1.0", "mlfoundations/dclm-baseline-1.0", "a3b142c", "raw/dclm").cd("a3b142c")
        ),
        "proofpile_2": _dl("raw/proof-pile-2", "EleutherAI/proof-pile-2", "901a927", "raw/proof-pile-2-f1b1d8"),
        "starcoderdata": _dl("raw/starcoderdata", "bigcode/starcoderdata", "9fc30b5", "raw/starcoderdata-720c8c"),
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
