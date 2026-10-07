# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""chemistry contracts and normalization policies."""

from taskcompendium.datasets.nemotron_ultra.source import quality_policy
from taskcompendium.pipeline.models import TaskPolicy

CRITERIA = (
    (
        "Check public molecular inputs and requested properties against retained target/validator "
        "fields and their units or format."
    ),
    (
        "The pinned rdkit_chemistry_agent grades the stored numeric property target; it does not "
        "recompute molecular properties or test stereochemical equivalence. It requires the row-selected "
        "boxed or double-parentheses answer wrapper and compares Python-rounded predicted and expected "
        "values. Check whether the public request agrees with these extraction and scoring rules."
    ),
)


def policy(selector: str, rubric_id: str) -> TaskPolicy:
    return quality_policy("chemistry", selector, rubric_id, CRITERIA, rubric_version="2")
