# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""reasoning-gym contracts and normalization policies."""

from taskcompendium.datasets.nemotron_ultra.source import quality_policy
from taskcompendium.pipeline.models import TaskPolicy

CRITERIA = (
    (
        "Compare the complete question and private answer/metadata, checking cheap contradictions "
        "and missing puzzle context."
    ),
    (
        "The source_dataset can determine scoring, aliases and partial credit. A hard puzzle or "
        "several surface forms of the same answer is not automatically a defect."
    ),
)


def policy(selector: str, rubric_id: str) -> TaskPolicy:
    """Build the reasoning-gym normalization and review policy."""
    return quality_policy("reasoning-gym", selector, rubric_id, CRITERIA, rubric_version="2")
