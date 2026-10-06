# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""math-proof contracts and normalization policies."""

from taskcompendium.pipeline.datasets.nemotron_ultra.source import quality_pipeline
from taskcompendium.pipeline.models import TaskPipeline

CRITERIA = (
    (
        "Check that the complete Lean header, formal_statement, imports, and holes to be filled "
        "are present or available through the stated environment."
    ),
    (
        "Hard proofs and absent reference proofs alone are not defects. Verify the formal target "
        "agrees with informal text and preserve exact Lean/toolchain requirements."
    ),
)


def pipeline(selector: str, rubric_id: str, provenance: str) -> TaskPipeline:
    """Build the math-proof normalization and review policy."""
    return quality_pipeline("math-proof", selector, rubric_id, (*CRITERIA, provenance))
