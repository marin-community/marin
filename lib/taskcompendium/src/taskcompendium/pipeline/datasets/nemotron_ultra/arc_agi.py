# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""arc-agi contracts and explicit pinned blend selections."""

from taskcompendium.pipeline.datasets.nemotron_ultra.source import quality_source

FAMILY_MODULE = __name__
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
UPSTREAM = "https://huggingface.co/datasets/nvidia/Nemotron-RL-ARC-AGI-v1"

SOURCES = tuple(
    quality_source(blend, selector, UPSTREAM, "arc-agi", CRITERIA, FAMILY_MODULE)
    for blend, selector in (
        ("mopd", "ultra_sft_step3200_nvarc_inductive"),
        ("mopd", "ultra_sft_step3200_nvarc_transductive"),
        ("rlvr1", "ultra_sft_step3200_nvarc_inductive"),
        ("rlvr1", "ultra_sft_step3200_nvarc_transductive"),
        ("rlvr2", "ultra_sft_step3200_nvarc_inductive"),
        ("rlvr2", "ultra_sft_step3200_nvarc_transductive"),
    )
)
