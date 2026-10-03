# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""swe-repo contracts and explicit pinned blend selections."""

from taskcompendium.pipeline.datasets.nemotron_ultra.source import ACTION_COMPARISON_CRITERION, quality_source

FAMILY_MODULE = __name__
CRITERIA = (
    (
        "The public repository/environment reference supplies code context; a pinned checkout and "
        "supplied issue can be coherent without inline repository files."
    ),
    (
        "Compare expected_action, ref_patch, issue, historical observations and environment to "
        "detect unrelated hidden repair requirements. Preserve SWE-Gym versus SWE-rebench "
        "attribution; a shared agent selector is not proof of source equivalence."
    ),
    (ACTION_COMPARISON_CRITERION),
)

SOURCES = tuple(
    quality_source(blend, selector, upstream, "swe-repo", CRITERIA, FAMILY_MODULE, component=component)
    for blend, selector, upstream, component in (
        (
            "mopd",
            "agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent",
            "https://huggingface.co/datasets/nebius/SWE-rebench-V2",
            "agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/nebius/SWE-rebench-V2",
        ),
        (
            "mopd",
            "agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent",
            "https://huggingface.co/datasets/SWE-Gym/SWE-Gym",
            "agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/SWE-Gym/SWE-Gym",
        ),
        (
            "mopd",
            "swe_pivot_len40k",
            "https://huggingface.co/datasets/nebius/SWE-rebench-V2",
            "swe_pivot_len40k/nebius/SWE-rebench-V2",
        ),
        (
            "mopd",
            "swe_pivot_len40k",
            "https://huggingface.co/datasets/SWE-Gym/SWE-Gym",
            "swe_pivot_len40k/SWE-Gym/SWE-Gym",
        ),
        (
            "mopd",
            "ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k",
            "https://huggingface.co/datasets/nebius/SWE-rebench-V2",
            "ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/nebius/SWE-rebench-V2",
        ),
        (
            "mopd",
            "ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k",
            "https://huggingface.co/datasets/SWE-Gym/SWE-Gym",
            "ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/SWE-Gym/SWE-Gym",
        ),
        (
            "rlvr1",
            "ultra_sft_step3200_swe_pivot_len40k",
            "https://huggingface.co/datasets/nebius/SWE-rebench-V2",
            "ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2",
        ),
        (
            "rlvr1",
            "ultra_sft_step3200_swe_pivot_len40k",
            "https://huggingface.co/datasets/SWE-Gym/SWE-Gym",
            "ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym",
        ),
        (
            "rlvr2",
            "ultra_sft_step3200_swe_pivot_len40k",
            "https://huggingface.co/datasets/nebius/SWE-rebench-V2",
            "ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2",
        ),
        (
            "rlvr2",
            "ultra_sft_step3200_swe_pivot_len40k",
            "https://huggingface.co/datasets/SWE-Gym/SWE-Gym",
            "ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym",
        ),
    )
)
