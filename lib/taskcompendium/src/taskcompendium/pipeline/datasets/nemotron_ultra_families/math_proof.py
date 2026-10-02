# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""math-proof contracts and explicit pinned blend selections."""

from taskcompendium.pipeline.datasets.nemotron_ultra_source import quality_source

FAMILY_MODULE = __name__
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
UPSTREAM = "https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v1"

SOURCES = tuple(
    quality_source(blend, selector, UPSTREAM, "math-proof", CRITERIA, FAMILY_MODULE)
    for blend, selector in (
        ("mopd", "ultra_sft_step3200_lean"),
        ("rlvr1", "ultra_sft_step3200_lean"),
        ("rlvr2", "ultra_sft_step3200_lean"),
    )
)
