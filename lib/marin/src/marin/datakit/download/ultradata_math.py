# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned UltraData-Math L2 source.

The upstream repository calls the public config ``UltraData-Math-L2-preview``.
Its dataset card reports the complete 33.7B-token L2 release under that config;
the standalone ``openbmb/UltraData-Math-L2`` repository is not publicly
accessible as of the pinned revision.
"""

from marin.datakit.download.hf_simple_util import NormalizationSchema, hf_normalize_steps
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "openbmb/UltraData-Math"
HF_REVISION = "fe10db8efd35597fd7fcff8ff576b5ec4ea5ff87"
MARIN_NAME = "ultradata-math/l2"
ROUGH_TOKEN_COUNT_B = 33.7


def ultradata_math_l2_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the public L2 download and normalization chain."""
    return hf_normalize_steps(
        marin_name=MARIN_NAME,
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        staged_path="raw/ultradata-math-l2",
        hf_urls_glob=("data/UltraData-Math-L2-preview/**/*.parquet",),
        text_field="content",
        file_extensions=(".parquet",),
        normalization_schema=NormalizationSchema.BARE,
        zephyr_max_parallelism=32,
    )
