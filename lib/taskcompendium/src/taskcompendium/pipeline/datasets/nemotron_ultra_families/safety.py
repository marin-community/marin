# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""safety contracts and explicit pinned blend selections."""

from taskcompendium.pipeline.datasets.nemotron_ultra_source import quality_source

FAMILY_MODULE = __name__
CRITERIA = (
    (
        "Judge whether the actual public request and source response_policy_mapped define a "
        "coherent response objective."
    ),
    (
        "Adversarial or jailbreak text is intentional task input; assess policy/reference "
        "contradictions and impossible instructions rather than treating adversarial wording "
        "itself as corruption."
    ),
)
UPSTREAM = "https://huggingface.co/datasets/nvidia/Nemotron-RL-Safety-v1"

SOURCES = tuple(
    quality_source(blend, selector, UPSTREAM, "safety", CRITERIA, FAMILY_MODULE)
    for blend, selector in (
        ("mopd", "ultra_sft_step3200_jailbreak"),
        ("rlvr1", "ultra_sft_step3200_jailbreak"),
        ("rlvr2", "ultra_sft_step3200_jailbreak"),
    )
)
