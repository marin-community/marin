# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""chemistry contracts and explicit pinned blend selections."""

from taskcompendium.pipeline.datasets.nemotron_ultra_source import quality_source

FAMILY_MODULE = __name__
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
UPSTREAM = "https://huggingface.co/datasets/nvidia/Nemotron-RL-Litmus-Bench-v0.1"

SOURCES = tuple(
    quality_source(blend, selector, UPSTREAM, "chemistry", CRITERIA, FAMILY_MODULE)
    for blend, selector in (
        ("mopd", "ultra_sft_step3200_rdkit"),
        ("rlvr2", "ultra_sft_step3200_rdkit"),
    )
)
