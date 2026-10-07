# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""competitive-programming contracts and normalization policies."""

from taskcompendium.datasets.nemotron_ultra.source import quality_policy
from taskcompendium.pipeline.models import TaskPolicy

CRITERIA = (
    ("Check complete input/output definitions, boundaries, examples, and consistency with retained unit_tests."),
    (
        "Special judges, alternative valid constructions, and function versus stdio delivery must "
        "retain their source contracts. No reference solution or unavailable execution alone is a "
        "quality defect."
    ),
)


def policy(selector: str, rubric_id: str) -> TaskPolicy:
    """Build the competitive-programming normalization and review policy."""
    return quality_policy("competitive-programming", selector, rubric_id, CRITERIA, rubric_version="2")
