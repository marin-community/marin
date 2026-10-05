# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""instruction-following contracts and normalization policies."""

from taskcompendium.pipeline.datasets.nemotron_ultra.source import quality_pipeline
from taskcompendium.pipeline.models import TaskPipeline

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


def pipeline(selector: str, rubric_id: str, provenance: str) -> TaskPipeline:
    """Build the instruction-following normalization and review policy."""
    return quality_pipeline("instruction-following", selector, rubric_id, (*CRITERIA, provenance))
