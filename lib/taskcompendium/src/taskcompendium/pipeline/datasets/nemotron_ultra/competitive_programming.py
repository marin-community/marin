# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""competitive-programming contracts and explicit pinned blend selections."""

from taskcompendium.pipeline.datasets.nemotron_ultra.source import quality_source

FAMILY_MODULE = __name__
CRITERIA = (
    (("Check complete input/output definitions, boundaries, examples, and consistency with " "retained unit_tests.")),
    (
        "Special judges, alternative valid constructions, and function versus stdio delivery must "
        "retain their source contracts. No reference solution or unavailable execution alone is a "
        "quality defect."
    ),
)
UPSTREAM = "https://huggingface.co/datasets/nvidia/Nemotron-RL-coding-competitive_coding"

SOURCES = tuple(
    quality_source(blend, selector, UPSTREAM, "competitive-programming", CRITERIA, FAMILY_MODULE)
    for blend, selector in (
        ("mopd", "ultra_sft_step3200_comp_coding"),
        ("rlvr1", "ultra_sft_step3200_comp_coding"),
        ("rlvr2", "ultra_sft_step3200_comp_coding"),
    )
)
