# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""math-answer contracts and explicit pinned blend selections."""

from taskcompendium.pipeline.datasets.nemotron_ultra.source import quality_source

FAMILY_MODULE = __name__
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
UPSTREAM = "https://huggingface.co/datasets/nvidia/Nemotron-RL-Math-v2"

SOURCES = tuple(
    quality_source(blend, selector, UPSTREAM, "math-answer", CRITERIA, FAMILY_MODULE)
    for blend, selector in (
        ("mopd", "ultra_sft_step3200_math_cot"),
        ("mopd", "ultra_sft_step3200_math_tir"),
        ("rlvr1", "ultra_sft_step3200_math_cot"),
        ("rlvr1", "ultra_sft_step3200_math_tir"),
        ("rlvr2", "ultra_sft_step3200_math_cot"),
        ("rlvr2", "ultra_sft_step3200_math_tir"),
    )
)
