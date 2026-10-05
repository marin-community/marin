# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned SwallowMath-v2 text sources.

The upstream ``qa`` release is already rendered as Markdown documents that can
contain several question/answer pairs. It is therefore registered as text, not
as structured chat.
"""

from marin.datakit.download.hf_simple_util import NormalizationSchema, hf_normalize_steps
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "tokyotech-llm/swallow-math-v2"
HF_REVISION = "b59a686bca9bb92e290f558ea7dcd73ec707b602"

SWALLOW_MATH_TOKEN_COUNTS_B = {
    "swallow-math-v2/qa": 13.6,
    "swallow-math-v2/textbook": 18.3,
}


def swallow_math_v2_normalize_steps() -> dict[str, tuple[StepSpec, ...]]:
    """Return independent text pipelines for the Q&A and textbook rewrites."""
    return {
        name: hf_normalize_steps(
            marin_name=name,
            hf_dataset_id=HF_DATASET_ID,
            revision=HF_REVISION,
            staged_path=f"raw/swallow-math-v2-{variant}",
            hf_urls_glob=(f"stage3-{variant}/train*.jsonl",),
            file_extensions=(".jsonl",),
            normalization_schema=NormalizationSchema.BARE,
            zephyr_max_parallelism=32,
        )
        for variant, name in (
            ("qa", "swallow-math-v2/qa"),
            ("textbook", "swallow-math-v2/textbook"),
        )
    }
