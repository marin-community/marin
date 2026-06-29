# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AutoMathText-V2 math and reasoning midtraining sources."""

from dataclasses import dataclass
from functools import cache

from marin.datakit.download.hf_simple_util import hf_normalize_steps
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "OpenSQZ/AutoMathText-V2"
HF_REVISION = "2a8d19c8edff7eeab35ffaa36f7845e86e2b3417"
HF_REVISION_SHORT = HF_REVISION[:7]


@dataclass(frozen=True)
class AutoMathTextV2Subset:
    """One AutoMathText-V2 domain exposed as a separate Datakit source."""

    marin_name: str
    hf_urls_glob: tuple[str, ...]
    data_subdir: str
    rough_token_count_b: float


AUTOMATHTEXT_V2_SUBSETS: dict[str, AutoMathTextV2Subset] = {
    "math_web": AutoMathTextV2Subset(
        marin_name="automathtext_v2/math_web",
        hf_urls_glob=("math_web/*/*.parquet",),
        data_subdir="math_web",
        rough_token_count_b=68.3,
    ),
    "reasoning_qa": AutoMathTextV2Subset(
        marin_name="automathtext_v2/reasoning_qa",
        hf_urls_glob=("reasoning_qa/*/*.parquet",),
        data_subdir="reasoning_qa",
        rough_token_count_b=86.2,
    ),
}
# The globs intentionally select every quality-percentile shard under each
# domain. Keep aggregate ``automathtext-v2-*`` configs and per-percentile splits
# out of this first registration.

AUTOMATHTEXT_V2_ROUGH_TOKENS_B = {
    subset.marin_name: subset.rough_token_count_b for subset in AUTOMATHTEXT_V2_SUBSETS.values()
}


@cache
def automathtext_v2_normalize_steps() -> dict[str, tuple[StepSpec, ...]]:
    """Return independent ``(download, normalize)`` chains for selected domains."""
    return {
        subset.marin_name: hf_normalize_steps(
            marin_name=subset.marin_name,
            hf_dataset_id=HF_DATASET_ID,
            revision=HF_REVISION,
            staged_path=f"raw/automathtext_v2/{subset_name}-{HF_REVISION_SHORT}",
            hf_urls_glob=subset.hf_urls_glob,
            data_subdir=subset.data_subdir,
            id_field="id",
            text_field="text",
            file_extensions=(".parquet",),
        )
        for subset_name, subset in AUTOMATHTEXT_V2_SUBSETS.items()
    }
