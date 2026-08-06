# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""marin-community/identity-data download and normalization."""

from marin.datakit.download.hf_simple_util import hf_normalize_steps
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "marin-community/identity-data"
HF_REVISION = "665ac6e"
MARIN_NAME = "identity-data/content"


def identity_data_content_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the ``(download, normalize)`` chain for rendered identity conversations."""
    return hf_normalize_steps(
        marin_name=MARIN_NAME,
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=("data/train-*.parquet",),
        id_field="seed_id",
        text_field="content",
    )
