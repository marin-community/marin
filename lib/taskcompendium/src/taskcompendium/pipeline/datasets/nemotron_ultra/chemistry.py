# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""chemistry contracts and normalization policies."""

from taskcompendium.pipeline.datasets.nemotron_ultra.source import quality_pipeline
from taskcompendium.pipeline.models import TaskPipeline

CRITERIA = (
    (
        "Check public molecular inputs and requested properties against retained target/validator "
        "fields and their units or format."
    ),
    (
        "RDKit validity, stereochemistry, equivalence, and numerical tolerances belong to the "
        "original evaluator. Its absent runtime is distinct from a malformed or contradictory "
        "chemistry task."
    ),
)


def pipeline(selector: str, rubric_id: str, provenance: str) -> TaskPipeline:
    """Build the chemistry normalization and review policy."""
    return quality_pipeline("chemistry", selector, rubric_id, (*CRITERIA, provenance))
