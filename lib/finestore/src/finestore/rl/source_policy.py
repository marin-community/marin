# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen catalog exceptions shared by ingestion and the Atlas API."""

GRANDFATHERED_MARINSKYRL_SOURCES: frozenset[str] = frozenset(
    {
        # Nemotron Ultra mix components: JSONL rows produced by Marin's own RL
        # pipeline, predating the task-directory format.
        "MarinSkyRL:nemotron_ultra_mopd/agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/SWE-Gym/SWE-Gym",
        "MarinSkyRL:nemotron_ultra_mopd/agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/nebius/SWE-rebench-V2",
        "MarinSkyRL:nemotron_ultra_mopd/makeshn_ultra_v3_ipi_train",
        "MarinSkyRL:nemotron_ultra_mopd/swe_pivot_len40k/SWE-Gym/SWE-Gym",
        "MarinSkyRL:nemotron_ultra_mopd/swe_pivot_len40k/nebius/SWE-rebench-V2",
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_lean",
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_math_tir",
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_rdkit",
        "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_toolcall_schema",
        "MarinSkyRL:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/SWE-Gym/SWE-Gym",
        "MarinSkyRL:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/nebius/SWE-rebench-V2",
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_lean",
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_math_tir",
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym",
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2",
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_tau_pivot",
        "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_toolcall_schema",
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_lean",
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_math_tir",
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym",
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2",
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_tau_pivot",
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_toolcall_schema",
        "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_rdkit",
    }
)
