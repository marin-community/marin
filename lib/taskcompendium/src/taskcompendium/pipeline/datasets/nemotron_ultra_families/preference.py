# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""preference contracts and explicit pinned blend selections."""

from taskcompendium.pipeline.datasets.nemotron_ultra_source import preference_source

FAMILY_MODULE = __name__
CRITERIA = (
    (
        "These are generation prompts with a GenRM principle, not stored chosen/rejected pairs; do "
        "not invent pair labels."
    ),
    ("The original principle and agent settings remain private grader evidence; no exact-answer " "key is supplied."),
    (
        "Reject missing context or contradictory requirements, separating those defects from an "
        "unbound GenRM evaluator."
    ),
    ("Do not assume records in another blend with the same selector are byte-identical or a " "verified alias."),
)
UPSTREAM = "https://huggingface.co/datasets/nvidia/Nemotron-RLHF-GenRM-v1"

SOURCES = tuple(
    preference_source(blend, selector, UPSTREAM, (scope, *CRITERIA), FAMILY_MODULE)
    for blend, selector, scope in (
        (
            "mopd",
            "hs3_en",
            (
                "Assess the full English conversation and private GenRM principle; earlier assistant "
                "errors are context, not a new gold answer."
            ),
        ),
        (
            "mopd",
            "hs3_multi",
            (
                "Preserve every multilingual turn and assess the final request in its actual language; "
                "multilingual context alone is not incoherent."
            ),
        ),
        (
            "mopd",
            "hs3_multiturn",
            (
                "Follow the complete conversation and earlier requirements; do not judge only the final "
                "short request without its history."
            ),
        ),
        (
            "mopd",
            "safety_en",
            (
                "Compare the request with its safety principle: a benign craft request mentioning a gun "
                "may call for a glue gun and helpful guidance."
            ),
        ),
        (
            "rlvr1",
            "language_mixing_hs3_ultra_genrm_fmt",
            (
                "Check the actual language instructions against the full multilingual history and private "
                "evaluation principle."
            ),
        ),
        (
            "rlvr2",
            "language_mixing_hs3_ultra_genrm_fmt",
            (
                "Check the actual language instructions against the full multilingual history and private "
                "evaluation principle."
            ),
        ),
    )
)
