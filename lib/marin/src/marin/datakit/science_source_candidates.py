# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materializable science sources awaiting completed Datakit artifacts.

Candidates stay separate from :func:`marin.datakit.sources.all_sources` until
their normalized and Harrier artifacts have been built and added to the checked-
in hero-data manifests.
"""

from functools import cache

from marin.datakit.download.arxiv_physics import (
    ARXIV_PHYSICS_MANIFEST,
    ARXIV_PHYSICS_NAME,
    arxiv_physics_normalize_steps,
)
from marin.datakit.download.hal_math import (
    HAL_MATH_ABSTRACTS_MANIFEST,
    HAL_MATH_ABSTRACTS_NAME,
    hal_math_abstracts_normalize_steps,
)
from marin.datakit.download.mit_ocw import MIT_OCW_SCIENCE_COURSES, mit_ocw_science_normalize_steps
from marin.datakit.download.openstax import OPENSTAX_BOOKS, openstax_science_normalize_steps
from marin.datakit.download.swallow_math import SWALLOW_MATH_TOKEN_COUNTS_B, swallow_math_v2_normalize_steps
from marin.datakit.download.ultradata_math import MARIN_NAME as ULTRADATA_MATH_L2_NAME
from marin.datakit.download.ultradata_math import ROUGH_TOKEN_COUNT_B as ULTRADATA_MATH_L2_TOKEN_COUNT_B
from marin.datakit.download.ultradata_math import ultradata_math_l2_normalize_steps
from marin.datakit.sources import DatakitSource


@cache
def science_source_candidates() -> dict[str, DatakitSource]:
    """Return training-approved source pipelines that have not completed ferries."""
    swallow_chains = swallow_math_v2_normalize_steps()
    candidates = {
        name: DatakitSource(name=name, normalize_steps=swallow_chains[name], rough_token_count_b=count)
        for name, count in SWALLOW_MATH_TOKEN_COUNTS_B.items()
    }
    candidates[ULTRADATA_MATH_L2_NAME] = DatakitSource(
        name=ULTRADATA_MATH_L2_NAME,
        normalize_steps=ultradata_math_l2_normalize_steps(),
        rough_token_count_b=ULTRADATA_MATH_L2_TOKEN_COUNT_B,
    )
    candidates[HAL_MATH_ABSTRACTS_NAME] = DatakitSource(
        name=HAL_MATH_ABSTRACTS_NAME,
        normalize_steps=hal_math_abstracts_normalize_steps(),
        rough_token_count_b=HAL_MATH_ABSTRACTS_MANIFEST.rough_tokens_b,
    )
    candidates[ARXIV_PHYSICS_NAME] = DatakitSource(
        name=ARXIV_PHYSICS_NAME,
        normalize_steps=arxiv_physics_normalize_steps(),
        rough_token_count_b=ARXIV_PHYSICS_MANIFEST.rough_tokens_b,
    )
    openstax_chains = openstax_science_normalize_steps()
    candidates.update(
        {
            name: DatakitSource(
                name=name,
                normalize_steps=openstax_chains[name],
                rough_token_count_b=book.rough_tokens_b,
            )
            for name, book in OPENSTAX_BOOKS.items()
        }
    )
    ocw_chains = mit_ocw_science_normalize_steps()
    candidates.update(
        {
            name: DatakitSource(
                name=name,
                normalize_steps=ocw_chains[name],
                rough_token_count_b=course.rough_tokens_b,
            )
            for name, course in MIT_OCW_SCIENCE_COURSES.items()
        }
    )
    return candidates
