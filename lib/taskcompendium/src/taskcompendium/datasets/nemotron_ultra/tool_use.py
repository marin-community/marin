# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""tool-use contracts and normalization policies."""

from taskcompendium.datasets.nemotron_ultra.source import ACTION_COMPARISON_CRITERION, quality_policy
from taskcompendium.pipeline.models import TaskPolicy

CRITERIA = (
    (
        "Check the complete role/tool sequence and advertised schemas against expected_action, "
        "scenario, and source environment state."
    ),
    (
        "Historical observations are public context. Expected future tool calls and reward state "
        "are private evidence; multiple valid actions require the original comparison policy "
        "rather than invented exact matching."
    ),
    (ACTION_COMPARISON_CRITERION),
)


def policy(selector: str, rubric_id: str) -> TaskPolicy:
    """Build the tool-use normalization and review policy."""
    return quality_policy("tool-use", selector, rubric_id, CRITERIA, rubric_version="2")
