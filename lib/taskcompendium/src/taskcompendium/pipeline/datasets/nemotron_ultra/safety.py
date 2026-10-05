# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""safety contracts and normalization policies."""

from taskcompendium.pipeline.datasets.nemotron_ultra.source import quality_pipeline
from taskcompendium.pipeline.models import TaskPipeline

CRITERIA = (
    ("Judge whether the actual public request and source response_policy_mapped define a coherent response objective."),
    (
        "Adversarial or jailbreak text is intentional task input; assess policy/reference "
        "contradictions and impossible instructions rather than treating adversarial wording "
        "itself as corruption."
    ),
)


def pipeline(selector: str, rubric_id: str, provenance: str) -> TaskPipeline:
    """Build the safety normalization and review policy."""
    return quality_pipeline("safety", selector, rubric_id, (*CRITERIA, provenance))
