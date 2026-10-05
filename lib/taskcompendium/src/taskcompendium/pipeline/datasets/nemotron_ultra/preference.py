# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""preference contracts and normalization policies."""

from taskcompendium.pipeline.datasets.nemotron_ultra.source import preference_pipeline
from taskcompendium.pipeline.models import TaskPipeline

CRITERIA = (
    (
        "These are generation prompts with a GenRM principle, not stored chosen/rejected pairs; do "
        "not invent pair labels."
    ),
    ("The original principle and agent settings remain private grader evidence; no exact-answer key is supplied."),
    ("Reject missing context or contradictory requirements, separating those defects from an unbound GenRM evaluator."),
    ("Do not assume records in another blend with the same selector are byte-identical or a verified alias."),
)


SCOPE_BY_SELECTOR = {
    "hs3_en": (
        "Assess the full English conversation and private GenRM principle; earlier assistant errors are "
        "context, not a new gold answer."
    ),
    "hs3_multi": (
        "Preserve every multilingual turn and assess the final request in its actual language; multilingual "
        "context alone is not incoherent."
    ),
    "hs3_multiturn": (
        "Follow the complete conversation and earlier requirements; do not judge only the final short request "
        "without its history."
    ),
    "language_mixing_hs3_ultra_genrm_fmt": (
        "Check the actual language instructions against the full multilingual history and private evaluation principle."
    ),
    "safety_en": (
        "Compare the request with its safety principle: a benign craft request mentioning a gun may call for "
        "a glue gun and helpful guidance."
    ),
}


def pipeline(selector: str, rubric_id: str) -> TaskPipeline:
    """Build the GenRM normalization and review policy for a selection."""
    return preference_pipeline(selector, rubric_id, (SCOPE_BY_SELECTOR[selector], *CRITERIA))
