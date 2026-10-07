# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""swe-repo contracts and normalization policies."""

from taskcompendium.datasets.nemotron_ultra.source import ACTION_COMPARISON_CRITERION, quality_policy
from taskcompendium.pipeline.models import TaskPolicy

CRITERIA = (
    (
        "The public repository/environment reference supplies code context; a pinned checkout and "
        "supplied issue can be coherent without inline repository files."
    ),
    (
        "Compare expected_action, ref_patch, issue, historical observations and environment to "
        "detect unrelated hidden repair requirements. Preserve SWE-Gym versus SWE-rebench "
        "attribution; a shared agent selector is not proof of source equivalence."
    ),
    (ACTION_COMPARISON_CRITERION),
)


def policy(selector: str, rubric_id: str) -> TaskPolicy:
    """Build the swe-repo normalization and review policy."""
    return quality_policy("swe-repo", selector, rubric_id, CRITERIA, rubric_version="2")
