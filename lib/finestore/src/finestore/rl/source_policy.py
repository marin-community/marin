# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen catalog exceptions shared by ingestion and the Atlas API."""

GRANDFATHERED_MARINSKYRL_SOURCES: frozenset[str] = frozenset(
    {
        # Nemotron Ultra mix components: JSONL rows produced by Marin's own RL
        # pipeline, predating the task-directory format.
        "Nemotron:nemotron_ultra_mopd/agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/SWE-Gym/SWE-Gym",
        "Nemotron:nemotron_ultra_mopd/agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/nebius/SWE-rebench-V2",
        "Nemotron:nemotron_ultra_mopd/makeshn_ultra_v3_ipi_train",
        "Nemotron:nemotron_ultra_mopd/swe_pivot_len40k/SWE-Gym/SWE-Gym",
        "Nemotron:nemotron_ultra_mopd/swe_pivot_len40k/nebius/SWE-rebench-V2",
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_lean",
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_math_tir",
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_rdkit",
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_toolcall_schema",
        "Nemotron:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/SWE-Gym/SWE-Gym",
        "Nemotron:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/nebius/SWE-rebench-V2",
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_lean",
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_math_tir",
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym",
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2",
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_tau_pivot",
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_toolcall_schema",
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_lean",
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_math_tir",
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym",
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2",
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_tau_pivot",
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_toolcall_schema",
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_rdkit",
    }
)
