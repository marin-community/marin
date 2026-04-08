# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NuminaMath download definition."""

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_spec import StepSpec

NUMINAMATH_HF_ID = "AI-MO/NuminaMath-1.5"
NUMINAMATH_REVISION = "1b05109"


def download_numinamath_step() -> StepSpec:
    return download_hf_step(
        "raw/numinamath_1_5",
        hf_dataset_id=NUMINAMATH_HF_ID,
        revision=NUMINAMATH_REVISION,
    )
