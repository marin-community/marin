# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Institutional Books download definition."""

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_spec import StepSpec

INSTITUTIONAL_BOOKS_HF_ID = "institutional/institutional-books-1.0"
INSTITUTIONAL_BOOKS_REVISION = "d2f504a"


def download_institutional_books_step() -> StepSpec:
    return download_hf_step(
        "raw/institutional-books-1.0",
        hf_dataset_id=INSTITUTIONAL_BOOKS_HF_ID,
        revision=INSTITUTIONAL_BOOKS_REVISION,
        override_output_path="raw/institutional-books-d2f504a",
    )
