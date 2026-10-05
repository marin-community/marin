# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""qa-abstention contracts and normalization policies."""

from taskcompendium.pipeline.datasets.nemotron_ultra.source import quality_pipeline
from taskcompendium.pipeline.models import TaskPipeline

CRITERIA = (
    "Check whether the actual question is answerable, and whether the private answer is correct.",
    (
        "Abstention policy and any [IDK] output requirements belong to the source contract; do not "
        "substitute exact-only matching for its semantic evaluator."
    ),
)


def pipeline(selector: str, rubric_id: str, provenance: str) -> TaskPipeline:
    """Build the qa-abstention normalization and review policy."""
    return quality_pipeline("qa-abstention", selector, rubric_id, (*CRITERIA, provenance))
