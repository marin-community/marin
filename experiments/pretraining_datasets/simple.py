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
Simple single-corpus dataset definitions and tokenization.

This module defines raw dataset downloads and their tokenized versions
for simple datasets that don't have multiple splits.
"""

import os.path

from levanter.data.text import TextLmDatasetFormat
from levanter.store.cache import CacheOptions
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize

from experiments.llama import llama3_tokenizer

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _tokenize_simple(
    name: str,
    raw_dataset: ExecutorStep,
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

downloads = {
    "fineweb": ExecutorStep(
        name="raw/fineweb",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id="HuggingFaceFW/fineweb",
            revision="cd85054",
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
        ),
        override_output_path="raw/fineweb",
    ),
    "fineweb_edu": (
        (
            fineweb_edu_base_step := ExecutorStep(
                name="raw/fineweb-edu",
                fn=download_hf,
                config=DownloadConfig(
                    hf_dataset_id="HuggingFaceFW/fineweb-edu",
                    revision=(revision := "87f0914"),
                    gcs_output_path=this_output_path(),
                    wait_for_completion=True,
                ),
                override_output_path=f"raw/fineweb-edu-{revision}",
            )
        ).cd("data")
    ),
    "fineweb_edu_sample_10bt": fineweb_edu_base_step.cd("sample/10BT"),
    "fineweb_edu_sample_100bt": fineweb_edu_base_step.cd("sample/100BT"),
    "fineweb_edu_sample_350bt": fineweb_edu_base_step.cd("sample/350BT"),
    "slimpajama": (
        ExecutorStep(
            name="raw/SlimPajama-627B",
            fn=download_hf,
            config=DownloadConfig(
                hf_dataset_id="cerebras/SlimPajama-627B",
                revision="2d0accd",
                gcs_output_path=this_output_path(),
                wait_for_completion=True,
            ),
            override_output_path="raw/SlimPajama-627B-262830",
        ).cd("2d0accd/huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/2d0accd")
    ),
    "slimpajama_6b": (
        ExecutorStep(
            name="raw/SlimPajama-6B",
            fn=download_hf,
            config=DownloadConfig(
                hf_dataset_id="DKYoon/SlimPajama-6B",
                revision="b5f90f4",
                gcs_output_path=this_output_path(),
                wait_for_completion=True,
            ),
            override_output_path="raw/SlimPajama-6B-be35b7",
        ).cd("data")
    ),
    "dolma3_mix_150b_1025": (
        ExecutorStep(
            name="raw/dolma3_mix-150B-1025",
            fn=download_hf,
            config=DownloadConfig(
                hf_dataset_id="allenai/dolma3_mix-150B-1025",
                revision="15d04ee",
                gcs_output_path=this_output_path(),
                wait_for_completion=True,
                append_sha_to_path=True,
            ),
            override_output_path="raw/dolma3_mix-150B-1025-15d04ee",
        ).cd("15d04ee")
    ),
    "dclm_baseline_wrong": ExecutorStep(
        name="raw/dclm-baseline-1.0",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id="mlfoundations/dclm-baseline-1.0",
            revision="a3b142c",
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
        ),
        override_output_path="raw/dclm_WRONG_20250211/",
    ),
    "dclm_baseline": (
        ExecutorStep(
            name="raw/dclm-baseline-1.0",
            fn=download_hf,
            config=DownloadConfig(
                hf_dataset_id="mlfoundations/dclm-baseline-1.0",
                revision="a3b142c",
                gcs_output_path=this_output_path(),
                wait_for_completion=True,
            ),
            override_output_path="raw/dclm",
        ).cd("a3b142c")
    ),
    "the_stack_dedup": (
        ExecutorStep(
            name="raw/the-stack-dedup",
            fn=download_hf,
            config=DownloadConfig(
                hf_dataset_id="bigcode/the-stack-dedup",
                revision="17cad72",
                gcs_output_path=this_output_path(),
                wait_for_completion=True,
            ),
            override_output_path="raw/the-stack-dedup-4ba450",
        ).cd("17cad72")
    ),
    "proofpile_2": (
        ExecutorStep(
            name="raw/proof-pile-2",
            fn=download_hf,
            config=DownloadConfig(
                hf_dataset_id="EleutherAI/proof-pile-2",
                revision="901a927",
                gcs_output_path=this_output_path(),
                wait_for_completion=True,
            ),
            override_output_path="raw/proof-pile-2-f1b1d8",
        ).cd("901a927/huggingface.co/datasets/EleutherAI/proof-pile-2/resolve/901a927")
    ),
    "the_pile_openwebtext2": (
        ExecutorStep(
            name="raw/the_pile_openwebtext2",
            fn=download_hf,
            config=DownloadConfig(
                hf_dataset_id="vietgpt/the_pile_openwebtext2",
                revision="1de27c6",
                gcs_output_path=this_output_path(),
                wait_for_completion=True,
            ),
            override_output_path="raw/the_pile_openwebtext2",
        ).cd("1de27c6/huggingface.co/datasets/vietgpt/the_pile_openwebtext2/resolve/1de27c6")
    ),
    # TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
    #   Migrate the dataset and cd can be removed.
    "starcoderdata": ExecutorStep(
        name="raw/starcoderdata",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id="bigcode/starcoderdata",
            revision="9fc30b5",
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
        ),
        override_output_path="raw/starcoderdata-720c8c",
    ),
}


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
