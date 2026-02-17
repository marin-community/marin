# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Simple single-corpus dataset definitions and tokenization.

This module defines raw dataset downloads and their tokenized versions
for simple datasets that don't have multiple splits.
"""

import os.path

from levanter.data.text import TextLmDatasetFormat
from levanter.store.cache import CacheOptions
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.step_model import StepSpec
from marin.processing.tokenize import TokenizeConfig, tokenize

from experiments.llama import llama3_tokenizer

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _tokenize_simple(
    name: str,
    raw_dataset_path: str,
    tokenizer: str | None = None,
    override_path: str | None = None,
    text_format: TextLmDatasetFormat = TextLmDatasetFormat(),
    cache_options: CacheOptions | None = None,
    deps: list[StepSpec] | None = None,
) -> StepSpec:
    """Helper to create a simple tokenized dataset."""

    return StepSpec(
        name=os.path.join("tokenized", name),
        hash_attrs={"tokenizer": tokenizer, "validation_paths": []},
        deps=deps or [],
        override_output_path=override_path,
        fn=lambda output_path: tokenize(
            TokenizeConfig(
                train_paths=[raw_dataset_path],
                validation_paths=[],
                cache_path=output_path,
                tokenizer=tokenizer,
                format=text_format,
            )
        ),
    )


# ============================================================================
# RAW DATASET DOWNLOADS
# ============================================================================

_fineweb = StepSpec(
    name="raw/fineweb",
    hash_attrs={"hf_dataset_id": "HuggingFaceFW/fineweb", "revision": "cd85054"},
    override_output_path="raw/fineweb",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="HuggingFaceFW/fineweb",
            revision="cd85054",
            gcs_output_path=output_path,
            wait_for_completion=True,
        )
    ),
)

_fineweb_edu_base = StepSpec(
    name="raw/fineweb-edu",
    hash_attrs={"hf_dataset_id": "HuggingFaceFW/fineweb-edu", "revision": "87f0914"},
    override_output_path="raw/fineweb-edu-87f0914",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="HuggingFaceFW/fineweb-edu",
            revision="87f0914",
            gcs_output_path=output_path,
            wait_for_completion=True,
        )
    ),
)

_slimpajama = StepSpec(
    name="raw/SlimPajama-627B",
    hash_attrs={"hf_dataset_id": "cerebras/SlimPajama-627B", "revision": "2d0accd"},
    override_output_path="raw/SlimPajama-627B-262830",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="cerebras/SlimPajama-627B",
            revision="2d0accd",
            gcs_output_path=output_path,
            wait_for_completion=True,
        )
    ),
)

_slimpajama_6b = StepSpec(
    name="raw/SlimPajama-6B",
    hash_attrs={"hf_dataset_id": "DKYoon/SlimPajama-6B", "revision": "b5f90f4"},
    override_output_path="raw/SlimPajama-6B-be35b7",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="DKYoon/SlimPajama-6B",
            revision="b5f90f4",
            gcs_output_path=output_path,
            wait_for_completion=True,
        )
    ),
)

_dolma3_mix_150b_1025 = StepSpec(
    name="raw/dolma3_mix-150B-1025",
    hash_attrs={"hf_dataset_id": "allenai/dolma3_mix-150B-1025", "revision": "15d04ee"},
    override_output_path="raw/dolma3_mix-150B-1025-15d04ee",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="allenai/dolma3_mix-150B-1025",
            revision="15d04ee",
            gcs_output_path=output_path,
            wait_for_completion=True,
            append_sha_to_path=True,
        )
    ),
)

_dclm_baseline_wrong = StepSpec(
    name="raw/dclm-baseline-1.0",
    hash_attrs={"hf_dataset_id": "mlfoundations/dclm-baseline-1.0", "revision": "a3b142c"},
    override_output_path="raw/dclm_WRONG_20250211/",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="mlfoundations/dclm-baseline-1.0",
            revision="a3b142c",
            gcs_output_path=output_path,
            wait_for_completion=True,
        )
    ),
)

_dclm_baseline_base = StepSpec(
    name="raw/dclm-baseline-1.0",
    hash_attrs={"hf_dataset_id": "mlfoundations/dclm-baseline-1.0", "revision": "a3b142c"},
    override_output_path="raw/dclm",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="mlfoundations/dclm-baseline-1.0",
            revision="a3b142c",
            gcs_output_path=output_path,
            wait_for_completion=True,
        )
    ),
)

_the_stack_dedup = StepSpec(
    name="raw/the-stack-dedup",
    hash_attrs={"hf_dataset_id": "bigcode/the-stack-dedup", "revision": "17cad72"},
    override_output_path="raw/the-stack-dedup-4ba450",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="bigcode/the-stack-dedup",
            revision="17cad72",
            gcs_output_path=output_path,
            wait_for_completion=True,
        )
    ),
)

_proofpile_2 = StepSpec(
    name="raw/proof-pile-2",
    hash_attrs={"hf_dataset_id": "EleutherAI/proof-pile-2", "revision": "901a927"},
    override_output_path="raw/proof-pile-2-f1b1d8",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="EleutherAI/proof-pile-2",
            revision="901a927",
            gcs_output_path=output_path,
            wait_for_completion=True,
        )
    ),
)

_the_pile_openwebtext2 = StepSpec(
    name="raw/the_pile_openwebtext2",
    hash_attrs={"hf_dataset_id": "vietgpt/the_pile_openwebtext2", "revision": "1de27c6"},
    override_output_path="raw/the_pile_openwebtext2",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="vietgpt/the_pile_openwebtext2",
            revision="1de27c6",
            gcs_output_path=output_path,
            wait_for_completion=True,
        )
    ),
)

_starcoderdata = StepSpec(
    name="raw/starcoderdata",
    hash_attrs={"hf_dataset_id": "bigcode/starcoderdata", "revision": "9fc30b5"},
    override_output_path="raw/starcoderdata-720c8c",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="bigcode/starcoderdata",
            revision="9fc30b5",
            gcs_output_path=output_path,
            wait_for_completion=True,
        )
    ),
)

# The downloads dict maps dataset names to paths (strings) or StepSpec objects.
# For datasets that used .cd() to navigate into subdirectories, we now use os.path.join.
downloads: dict[str, StepSpec | str] = {
    "fineweb": _fineweb,
    "fineweb_edu": os.path.join(_fineweb_edu_base.output_path, "data"),
    "fineweb_edu_sample_10bt": os.path.join(_fineweb_edu_base.output_path, "sample/10BT"),
    "fineweb_edu_sample_100bt": os.path.join(_fineweb_edu_base.output_path, "sample/100BT"),
    "fineweb_edu_sample_350bt": os.path.join(_fineweb_edu_base.output_path, "sample/350BT"),
    "slimpajama": os.path.join(
        _slimpajama.output_path,
        "2d0accd/huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/2d0accd",
    ),
    "slimpajama_6b": os.path.join(_slimpajama_6b.output_path, "data"),
    "dolma3_mix_150b_1025": os.path.join(_dolma3_mix_150b_1025.output_path, "15d04ee"),
    "dclm_baseline_wrong": _dclm_baseline_wrong,
    "dclm_baseline": os.path.join(_dclm_baseline_base.output_path, "a3b142c"),
    "the_stack_dedup": os.path.join(_the_stack_dedup.output_path, "17cad72"),
    "proofpile_2": os.path.join(
        _proofpile_2.output_path,
        "901a927/huggingface.co/datasets/EleutherAI/proof-pile-2/resolve/901a927",
    ),
    "the_pile_openwebtext2": os.path.join(
        _the_pile_openwebtext2.output_path,
        "1de27c6/huggingface.co/datasets/vietgpt/the_pile_openwebtext2/resolve/1de27c6",
    ),
    # TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
    #   Migrate the dataset and cd can be removed.
    "starcoderdata": _starcoderdata,
}

# All base StepSpec objects for running downloads
download_steps = [
    _fineweb,
    _fineweb_edu_base,
    _slimpajama,
    _slimpajama_6b,
    _dolma3_mix_150b_1025,
    _dclm_baseline_wrong,
    _dclm_baseline_base,
    _the_stack_dedup,
    _proofpile_2,
    _the_pile_openwebtext2,
    _starcoderdata,
]

# Map from download name to the base StepSpec (not the subpath).
# Used by __init__.py for download step registration.
download_base_steps: dict[str, StepSpec] = {
    "fineweb": _fineweb,
    "fineweb_edu": _fineweb_edu_base,
    "slimpajama": _slimpajama,
    "slimpajama_6b": _slimpajama_6b,
    "dolma3_mix_150b_1025": _dolma3_mix_150b_1025,
    "dclm_baseline_wrong": _dclm_baseline_wrong,
    "dclm_baseline": _dclm_baseline_base,
    "the_stack_dedup": _the_stack_dedup,
    "proofpile_2": _proofpile_2,
    "the_pile_openwebtext2": _the_pile_openwebtext2,
    "starcoderdata": _starcoderdata,
}


def _get_download_path(key: str) -> str:
    """Get the string path for a download entry."""
    val = downloads[key]
    if isinstance(val, StepSpec):
        return val.output_path
    return val


# ============================================================================
# TOKENIZED DATASETS
# ============================================================================

tokenized = {
    "dclm_baseline": _tokenize_simple(
        "dclm_baseline",
        _get_download_path("dclm_baseline"),
        tokenizer=llama3_tokenizer,
        override_path="tokenized/dclm_baseline-0206f1/",
    ),
    "starcoderdata": _tokenize_simple(
        "starcoderdata",
        _get_download_path("starcoderdata"),
        tokenizer=llama3_tokenizer,
        text_format=TextLmDatasetFormat(text_key="content"),
        override_path="tokenized/starcoderdata-12f018/",
    ),
    "proofpile_2": _tokenize_simple(
        "proofpile_2",
        _get_download_path("proofpile_2"),
        tokenizer=llama3_tokenizer,
        override_path="tokenized/proofpile_2-4a35c7/",
    ),
    "slimpajama_6b": _tokenize_simple(
        "SlimPajama-6B",
        _get_download_path("slimpajama_6b"),
        tokenizer=llama3_tokenizer,
    ),
    "fineweb_edu": _tokenize_simple(
        "fineweb-edu",
        _get_download_path("fineweb_edu"),
        tokenizer=llama3_tokenizer,
    ),
    "dolma3_mix_150b_1025": _tokenize_simple(
        "dolma3_mix-150B-1025",
        _get_download_path("dolma3_mix_150b_1025"),
        tokenizer=llama3_tokenizer,
        override_path="tokenized/dolma3_mix-150B-1025-15d04ee/",
    ),
}
