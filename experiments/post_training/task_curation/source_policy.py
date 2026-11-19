# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source-specific training exclusions and retained MarinSkyRL admission exceptions."""

CROSSCODE_EVAL_SOURCE_IDS: frozenset[str] = frozenset(
    {
        "TaskTrove:laion__exp_rpt_crosscodeeval-csharp-v4",
        "TaskTrove:laion__exp_rpt_crosscodeeval-java-v3",
        "TaskTrove:laion__exp_rpt_crosscodeeval-python-v2",
        "TaskTrove:laion__exp_rpt_crosscodeeval-typescript-v2",
    }
)
CROSSCODE_EVAL_REASON = (
    "Evaluation only for CrossCodeEval-derived sources: Python's native tests contain the Terminal-Bench canary. "
    "An inspected Python tests/test_state.py marker is preserved at private pin "
    "a953f93bc830704525ff2cce0b296dd262009662, file SHA-256 "
    "335184f930fd957094b86f8f5402465766c40674de9044e2feb6faf21d904789. "
    "Within-source dedup applies; lexical contamination is measured only."
)

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

GRANDFATHERED_DAYTONA_EXEMPTION = "Grandfathered blend; Daytona safety is exempt and no verdict is asserted"
