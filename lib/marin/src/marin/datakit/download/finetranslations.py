# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FineTranslations download definition."""

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_spec import StepSpec

FINETRANSLATIONS_HF_ID = "HuggingFaceFW/finetranslations"
FINETRANSLATIONS_REVISION = "af3f4ca"


def download_finetranslations_step() -> StepSpec:
    return download_hf_step(
        "raw/finetranslations",
        hf_dataset_id=FINETRANSLATIONS_HF_ID,
        revision=FINETRANSLATIONS_REVISION,
        hf_urls_glob=["data/**/*.parquet"],
    )
