# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""qa-abstention contracts and explicit pinned blend selections."""

from taskcompendium.pipeline.datasets.nemotron_ultra.source import quality_source

FAMILY_MODULE = __name__
CRITERIA = (
    "Check whether the actual question is answerable, and whether the private answer is correct.",
    (
        "Abstention policy and any [IDK] output requirements belong to the source contract; do not "
        "substitute exact-only matching for its semantic evaluator."
    ),
)
UPSTREAM = "https://huggingface.co/datasets/nvidia/Nemotron-RL-QA-Abstention-v1"

SOURCES = tuple(
    quality_source(blend, selector, UPSTREAM, "qa-abstention", CRITERIA, FAMILY_MODULE)
    for blend, selector in (
        ("mopd", "ultra_sft_step3200_abstention"),
        ("rlvr1", "ultra_sft_step3200_abstention"),
        ("rlvr2", "ultra_sft_step3200_abstention"),
    )
)
