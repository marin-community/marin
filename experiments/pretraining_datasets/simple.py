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
from marin.execution.executor import ExecutorStep, StepRef, versioned
from marin.execution import step, StepContext
from marin.processing.tokenize import TokenizeConfig, tokenize

from experiments.llama import llama3_tokenizer

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def tokenize_simple(
    name: str,
    raw_dataset: ExecutorStep,
    tokenizer: str | None = None,
    override_path: str | None = None,
    text_format: TextLmDatasetFormat = TextLmDatasetFormat(),
    cache_options: CacheOptions | None = None,
) -> ExecutorStep[TokenizeConfig]:
    """Helper to create a simple tokenized dataset."""

    @step(name=os.path.join("tokenized", name), fn=tokenize)
    def tokenize_step(ctx: StepContext):
        return TokenizeConfig(
            train_paths=[ctx.require(raw_dataset)],
            validation_paths=versioned([]),
            cache_path=ctx.output,
            tokenizer=versioned(tokenizer),
            format=text_format,
        )

    result = tokenize_step()
    if override_path is not None:
        result = result.with_output_path(override_path)

    return result


# ============================================================================
# RAW DATASET DOWNLOADS
# ============================================================================


@step(name="raw/fineweb", fn=download_hf, override_output_path="raw/fineweb")
def fineweb_download(ctx: StepContext):
    return DownloadConfig(
        hf_dataset_id="HuggingFaceFW/fineweb",
        revision="cd85054",
        gcs_output_path=ctx.output,
        wait_for_completion=True,
    )


@step(name="raw/fineweb-edu", fn=download_hf, override_output_path="raw/fineweb-edu-c2beb4")
def fineweb_edu_download(ctx: StepContext):
    return DownloadConfig(
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision="3c452cb",
        gcs_output_path=ctx.output,
        wait_for_completion=True,
    )


@step(name="raw/SlimPajama-627B", fn=download_hf, override_output_path="raw/SlimPajama-627B-262830")
def slimpajama_download(ctx: StepContext):
    return DownloadConfig(
        hf_dataset_id="cerebras/SlimPajama-627B",
        revision="2d0accd",
        gcs_output_path=ctx.output,
        wait_for_completion=True,
    )


@step(name="raw/SlimPajama-6B", fn=download_hf, override_output_path="raw/SlimPajama-6B-be35b7")
def slimpajama_6b_download(ctx: StepContext):
    return DownloadConfig(
        hf_dataset_id="DKYoon/SlimPajama-6B",
        revision="b5f90f4",
        gcs_output_path=ctx.output,
        wait_for_completion=True,
    )


@step(name="raw/dolma3_mix-150B-1025", fn=download_hf, override_output_path="raw/dolma3_mix-150B-1025-15d04ee")
def dolma3_mix_download(ctx: StepContext):
    return DownloadConfig(
        hf_dataset_id="allenai/dolma3_mix-150B-1025",
        revision="15d04ee",
        gcs_output_path=ctx.output,
        wait_for_completion=True,
        append_sha_to_path=True,
    )


@step(name="raw/dclm-baseline-1.0", fn=download_hf, override_output_path="raw/dclm_WRONG_20250211/")
def dclm_baseline_wrong_download(ctx: StepContext):
    return DownloadConfig(
        hf_dataset_id="mlfoundations/dclm-baseline-1.0",
        revision="a3b142c",
        gcs_output_path=ctx.output,
        wait_for_completion=True,
    )


@step(name="raw/dclm-baseline-1.0", fn=download_hf, override_output_path="raw/dclm")
def dclm_baseline_download(ctx: StepContext):
    return DownloadConfig(
        hf_dataset_id="mlfoundations/dclm-baseline-1.0",
        revision="a3b142c",
        gcs_output_path=ctx.output,
        wait_for_completion=True,
    )


@step(name="raw/the-stack-dedup", fn=download_hf, override_output_path="raw/the-stack-dedup-4ba450")
def the_stack_dedup_download(ctx: StepContext):
    return DownloadConfig(
        hf_dataset_id="bigcode/the-stack-dedup",
        revision="17cad72",
        gcs_output_path=ctx.output,
        wait_for_completion=True,
    )


@step(name="raw/proof-pile-2", fn=download_hf, override_output_path="raw/proof-pile-2-f1b1d8")
def proofpile_2_download(ctx: StepContext):
    return DownloadConfig(
        hf_dataset_id="EleutherAI/proof-pile-2",
        revision="901a927",
        gcs_output_path=ctx.output,
        wait_for_completion=True,
    )


@step(name="raw/the_pile_openwebtext2", fn=download_hf, override_output_path="raw/the_pile_openwebtext2")
def the_pile_openwebtext2_download(ctx: StepContext):
    return DownloadConfig(
        hf_dataset_id="vietgpt/the_pile_openwebtext2",
        revision="1de27c6",
        gcs_output_path=ctx.output,
        wait_for_completion=True,
    )


# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
@step(name="raw/starcoderdata", fn=download_hf, override_output_path="raw/starcoderdata-720c8c")
def starcoderdata_download(ctx: StepContext):
    return DownloadConfig(
        hf_dataset_id="bigcode/starcoderdata",
        revision="9fc30b5",
        gcs_output_path=ctx.output,
        wait_for_completion=True,
    )
