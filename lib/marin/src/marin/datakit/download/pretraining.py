# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pre-defined download steps for common pretraining datasets.

Each function returns a StepSpec for downloading a specific dataset from
HuggingFace. These are the canonical definitions — experiments should
import from here rather than defining download steps inline.

For datasets where the actual data lives in a subdirectory of the download
(e.g. fineweb-edu has data under ``data/``), the function returns the
StepSpec for the base download. Consumers that need the subdirectory path
should use ``step.output_path + "/data"`` or convert to ExecutorStep and
use ``.cd("data")``.
"""

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_spec import StepSpec


def fineweb_download() -> StepSpec:
    return download_hf_step(
        "raw/fineweb",
        hf_dataset_id="HuggingFaceFW/fineweb",
        revision="cd85054",
        override_output_path="raw/fineweb",
    )


def fineweb_edu_download() -> StepSpec:
    """Base download for fineweb-edu. Data is under the ``data/`` subdirectory."""
    return download_hf_step(
        "raw/fineweb-edu",
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision="87f0914",
        override_output_path="raw/fineweb-edu-87f0914",
    )


def slimpajama_download() -> StepSpec:
    return download_hf_step(
        "raw/SlimPajama-627B",
        hf_dataset_id="cerebras/SlimPajama-627B",
        revision="2d0accd",
        override_output_path="raw/SlimPajama-627B-262830",
    )


def slimpajama_6b_download() -> StepSpec:
    return download_hf_step(
        "raw/SlimPajama-6B",
        hf_dataset_id="DKYoon/SlimPajama-6B",
        revision="b5f90f4",
        override_output_path="raw/SlimPajama-6B-be35b7",
    )


def dolma3_mix_150b_1025_download() -> StepSpec:
    return download_hf_step(
        "raw/dolma3_mix-150B-1025",
        hf_dataset_id="allenai/dolma3_mix-150B-1025",
        revision="15d04ee",
        override_output_path="raw/dolma3_mix-150B-1025-15d04ee",
    )


def dclm_baseline_download() -> StepSpec:
    return download_hf_step(
        "raw/dclm-baseline-1.0",
        hf_dataset_id="mlfoundations/dclm-baseline-1.0",
        revision="a3b142c",
        override_output_path="raw/dclm",
    )


def the_stack_dedup_download() -> StepSpec:
    return download_hf_step(
        "raw/the-stack-dedup",
        hf_dataset_id="bigcode/the-stack-dedup",
        revision="17cad72",
        override_output_path="raw/the-stack-dedup-4ba450",
    )


def proofpile_2_download() -> StepSpec:
    return download_hf_step(
        "raw/proof-pile-2",
        hf_dataset_id="EleutherAI/proof-pile-2",
        revision="901a927",
        override_output_path="raw/proof-pile-2-f1b1d8",
    )


def the_pile_openwebtext2_download() -> StepSpec:
    return download_hf_step(
        "raw/the_pile_openwebtext2",
        hf_dataset_id="vietgpt/the_pile_openwebtext2",
        revision="1de27c6",
        override_output_path="raw/the_pile_openwebtext2",
    )


def starcoderdata_download() -> StepSpec:
    return download_hf_step(
        "raw/starcoderdata",
        hf_dataset_id="bigcode/starcoderdata",
        revision="9fc30b5",
        override_output_path="raw/starcoderdata-720c8c",
    )


def dclm_baseline_wrong_download() -> StepSpec:
    """Legacy download with incorrect path. Kept for backward compat."""
    return download_hf_step(
        "raw/dclm-baseline-1.0",
        hf_dataset_id="mlfoundations/dclm-baseline-1.0",
        revision="a3b142c",
        override_output_path="raw/dclm_WRONG_20250211/",
    )
