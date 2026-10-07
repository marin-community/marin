# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""math-proof contracts and normalization policies."""

from taskcompendium.datasets.nemotron_ultra.source import quality_policy
from taskcompendium.pipeline.models import TaskPolicy

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


def policy(selector: str, rubric_id: str) -> TaskPolicy:
    """Build the math-proof normalization and review policy."""
    return quality_policy("math-proof", selector, rubric_id, CRITERIA, rubric_version="2")
