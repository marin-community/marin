# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download subsets of the bigcode/starcoder2data-extras dataset from HuggingFace.

Subsets: ir_cpp, ir_python, ir_rust, ir_low_resource, documentation, kaggle.
"""

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "bigcode/starcoder2data-extras"
HF_REVISION = "1ba0d4f"

SUBSETS = ["ir_cpp", "ir_python", "ir_rust", "ir_low_resource", "documentation", "kaggle"]


def download_starcoder2_extras_step(subset: str) -> StepSpec:
    """Download a single subset of the starcoder2data-extras dataset."""
    return download_hf_step(
        f"raw/starcoder2_extras/{subset}",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[f"{subset}/*.parquet"],
        override_output_path=f"raw/starcoder2_extras-{HF_REVISION}/{subset}",
    )


def download_all_starcoder2_extras_steps() -> list[StepSpec]:
    """Download all selected subsets of starcoder2data-extras."""
    return [download_starcoder2_extras_step(subset) for subset in SUBSETS]
