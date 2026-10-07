# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""instruction-following contracts and normalization policies."""

from taskcompendium.datasets.nemotron_ultra.source import quality_policy
from taskcompendium.pipeline.models import TaskPolicy

CRITERIA = (
    (
        "Identify the underlying content request and verify that all supplied formal constraints "
        "and semantic rubric requirements can hold together."
    ),
    (
        "Preserve every conversation turn. Public schemas/examples are legitimate context. "
        "Distinguish factual extraction from authorized arbitrary schema generation, and compare "
        "private constraints with public instructions."
    ),
)


def policy(selector: str, rubric_id: str) -> TaskPolicy:
    """Build the instruction-following normalization and review policy."""
    return quality_policy("instruction-following", selector, rubric_id, CRITERIA, rubric_version="2")
