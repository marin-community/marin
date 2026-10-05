# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""arc-agi contracts and normalization policies."""

from taskcompendium.pipeline.datasets.nemotron_ultra.source import quality_pipeline
from taskcompendium.pipeline.models import TaskPipeline

CRITERIA = (
    (
        "Check that all training grids, public test inputs, and private expected_output match the "
        "stated grid transformation and dimensions."
    ),
    (
        "Inductive variants require producing a reusable transformation program; transductive "
        "variants request the output grid. Do not replace one grading contract with the other or "
        "expose hidden outputs."
    ),
)


def pipeline(selector: str, rubric_id: str, provenance: str) -> TaskPipeline:
    """Build the arc-agi normalization and review policy."""
    return quality_pipeline("arc-agi", selector, rubric_id, (*CRITERIA, provenance))
