# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""reasoning-gym contracts and normalization policies."""

from taskcompendium.pipeline.datasets.nemotron_ultra.source import quality_pipeline
from taskcompendium.pipeline.models import TaskPipeline

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


def pipeline(selector: str, rubric_id: str, provenance: str) -> TaskPipeline:
    """Build the reasoning-gym normalization and review policy."""
    return quality_pipeline("reasoning-gym", selector, rubric_id, (*CRITERIA, provenance))
