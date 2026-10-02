# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""reasoning-gym contracts and explicit pinned blend selections."""

from taskcompendium.pipeline.datasets.nemotron_ultra_source import quality_source

FAMILY_MODULE = __name__
CRITERIA = (
    (
        "Compare the complete question and private answer/metadata, checking cheap contradictions "
        "and missing puzzle context."
    ),
    (
        "The source_dataset can determine scoring, aliases and partial credit. A hard puzzle or "
        "several surface forms of the same answer is not automatically a defect."
    ),
)
UPSTREAM = "https://huggingface.co/datasets/nvidia/Nemotron-RL-ReasoningGym-v1"

SOURCES = tuple(
    quality_source(blend, selector, UPSTREAM, "reasoning-gym", CRITERIA, FAMILY_MODULE)
    for blend, selector in (
        ("mopd", "ultra_sft_step3200_reasoning_gym"),
        ("rlvr1", "ultra_sft_step3200_reasoning_gym"),
        ("rlvr2", "ultra_sft_step3200_reasoning_gym"),
    )
)
