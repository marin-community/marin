# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""qa-multiple-choice contracts and normalization policies."""

from taskcompendium.datasets.nemotron_ultra.source import quality_policy
from taskcompendium.pipeline.models import TaskPolicy

CRITERIA = (
    "Check option labels and answer encoding against the complete public choices and expected_answer.",
    (
        "Knowledge questions can use ordinary external knowledge. Missing referenced passages, "
        "images, or material contradictions are defects; source labels must not be exposed as "
        "public hints."
    ),
)


def policy(selector: str, rubric_id: str) -> TaskPolicy:
    """Build the qa-multiple-choice normalization and review policy."""
    return quality_policy("qa-multiple-choice", selector, rubric_id, CRITERIA, rubric_version="2")
