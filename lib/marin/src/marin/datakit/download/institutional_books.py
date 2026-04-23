# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""institutional/institutional-books-1.0 download + normalize helpers.

Plain HF dataset; no custom preprocessing. Thin wrapper over
:func:`hf_normalize_steps` so the registry can import from here for
consistency with the other download modules.
"""

from __future__ import annotations

from marin.datakit.download.hf_simple_util import hf_normalize_steps
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "institutional/institutional-books-1.0"
HF_REVISION = "d2f504a"


def institutional_books_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the ``(download, normalize)`` chain for institutional_books."""
    return hf_normalize_steps(
        marin_name="institutional_books",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        staged_path=f"raw/institutional-books-{HF_REVISION}",
    )
