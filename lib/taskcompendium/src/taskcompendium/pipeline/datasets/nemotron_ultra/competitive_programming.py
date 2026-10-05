# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""competitive-programming contracts and normalization policies."""

from taskcompendium.pipeline.datasets.nemotron_ultra.source import quality_pipeline
from taskcompendium.pipeline.models import TaskPipeline

CRITERIA = (
    ("Check complete input/output definitions, boundaries, examples, and consistency with retained unit_tests."),
    (
        "Special judges, alternative valid constructions, and function versus stdio delivery must "
        "retain their source contracts. No reference solution or unavailable execution alone is a "
        "quality defect."
    ),
)


def pipeline(selector: str, rubric_id: str, provenance: str) -> TaskPipeline:
    """Build the competitive-programming normalization and review policy."""
    return quality_pipeline("competitive-programming", selector, rubric_id, (*CRITERIA, provenance))
