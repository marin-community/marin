# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""math-answer contracts and normalization policies."""

from taskcompendium.pipeline.datasets.nemotron_ultra.source import quality_pipeline
from taskcompendium.pipeline.models import TaskPipeline

CRITERIA = (
    (
        "Verify complete mathematical inputs and agreement of expected_answer with the actual "
        "public problem; difficulty alone is not a defect."
    ),
    (
        "The source can require symbolic, approximate, or judge-assisted scoring. A single stored "
        "expression is evidence, not authority to reject equivalent answers. Unresolved external "
        "question placeholders are acquisition gaps."
    ),
)


def pipeline(selector: str, rubric_id: str, provenance: str) -> TaskPipeline:
    """Build the math-answer normalization and review policy."""
    return quality_pipeline("math-answer", selector, rubric_id, (*CRITERIA, provenance))
