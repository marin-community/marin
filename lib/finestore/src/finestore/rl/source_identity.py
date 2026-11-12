# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Published source IDs held immutable pending explicit retirement approval.

Each target is the canonical generated identity. This frozen map is only for
existing publications; new sources must use their generated ID immediately.
"""

PENDING_SOURCE_ID_RETIREMENTS: dict[str, str] = {
    "MarinSkyRL:aime24": "HuggingFaceH4:aime24",
    "MarinSkyRL:aime_1983_2024": "di-zhang-fdu:aime_1983_2024",
    "MarinSkyRL:asdiv": "ASDiv:asdiv",
    "MarinSkyRL:dapo_math": "BytedTsinghua-SIA:dapo_math",
    "MarinSkyRL:deepscaler": "agentica-org:deepscaler",
    "MarinSkyRL:gsm8k": "OpenAI:gsm8k",
    "MarinSkyRL:hardmath": "pafitis:hardmath",
    "MarinSkyRL:hendrycks_math": "EleutherAI:hendrycks_math",
    "MarinSkyRL:math500": "HuggingFaceH4:math500",
    "MarinSkyRL:numina_math": "AI-MO:numina_math",
    "MarinSkyRL:rlvr_math": "AllenAI:rlvr_math",
    "MarinSkyRL:svamp": "ChilleD:svamp",
    "MarinSkyRL:apps": "CodeParrot:apps",
    "MarinSkyRL:eurus2_code": "PRIME-RL:eurus2_code",
    "MarinSkyRL:verifiable_code": "Open-R1:verifiable_code",
    "MarinSkyRL:gretel_text_to_sql": "Gretel:gretel_text_to_sql",
    "MarinSkyRL:nemotron_if": "Nemotron:nemotron_if",
    "MarinSkyRL:rlvr_ifeval": "AllenAI:rlvr_ifeval",
    "MarinSkyRL:gpqa": "GPQA:gpqa",
    "MarinSkyRL:openscience": "Nemotron:openscience",
    "MarinSkyRL:hh_rlhf/harmless-base": "Anthropic:hh_rlhf/harmless-base",
    "MarinSkyRL:hh_rlhf/helpful-base": "Anthropic:hh_rlhf/helpful-base",
    "MarinSkyRL:hh_rlhf/helpful-online": "Anthropic:hh_rlhf/helpful-online",
    "MarinSkyRL:hh_rlhf/helpful-rejection-sampled": "Anthropic:hh_rlhf/helpful-rejection-sampled",
    "MarinSkyRL:kto_mix/argilla/distilabel-capybara-dpo-7k-binarized": (
        "TRL:kto_mix/argilla/distilabel-capybara-dpo-7k-binarized"
    ),
    "MarinSkyRL:kto_mix/argilla/distilabel-intel-orca-dpo-pairs": "TRL:kto_mix/argilla/distilabel-intel-orca-dpo-pairs",
    "MarinSkyRL:kto_mix/argilla/ultrafeedback-binarized-preferences-cleaned": (
        "TRL:kto_mix/argilla/ultrafeedback-binarized-preferences-cleaned"
    ),
    "Task Trove:DCAgent__code-contests-noblock": "TaskTrove:DCAgent__code-contests-noblock",
    "Task Trove:laion__codeforces-v3": "TaskTrove:laion__codeforces-v3",
    "Task Trove:laion__nemotron-gym-competitive-coding-v2": "TaskTrove:laion__nemotron-gym-competitive-coding-v2",
    "Task Trove:laion__exp_rpt_taco-v2": "TaskTrove:laion__exp_rpt_taco-v2",
    "Task Trove:DCAgent__exp_rpt_curriculum-easy": "TaskTrove:DCAgent__exp_rpt_curriculum-easy",
    "Task Trove:DCAgent__exp_rpt_curriculum-medium-v2": "TaskTrove:DCAgent__exp_rpt_curriculum-medium-v2",
    "Task Trove:DCAgent__exp_rpt_e2egit-v2": "TaskTrove:DCAgent__exp_rpt_e2egit-v2",
    "Task Trove:DCAgent__exp_rpt_e2egit-large": "TaskTrove:DCAgent__exp_rpt_e2egit-large",
    "Task Trove:DCAgent__exp_rpt_multifile-v3": "TaskTrove:DCAgent__exp_rpt_multifile-v3",
    "Task Trove:DCAgent__exp_rpt_pymethods2test-v3": "TaskTrove:DCAgent__exp_rpt_pymethods2test-v3",
    "Task Trove:DCAgent__exp_rpt_pymethods2test-large-v2": "TaskTrove:DCAgent__exp_rpt_pymethods2test-large-v2",
    "Task Trove:DCAgent__exp_rpt_unitsyn-python-v4": "TaskTrove:DCAgent__exp_rpt_unitsyn-python-v4",
    "Task Trove:DCAgent__exp_rpt_unitsyn-python-large-v2": "TaskTrove:DCAgent__exp_rpt_unitsyn-python-large-v2",
    "Task Trove:DCAgent__exp_rpt_stack-pytest-v2": "TaskTrove:DCAgent__exp_rpt_stack-pytest-v2",
    "Task Trove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2": "TaskTrove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2",
    "Task Trove:laion__nemotron-gym-structured-outputs-v4": "TaskTrove:laion__nemotron-gym-structured-outputs-v4",
    "Task Trove:DCAgent__swe_rebench_v2_patched_oracle-v2": "TaskTrove:DCAgent__swe_rebench_v2_patched_oracle-v2",
    "Task Trove:laion__swesmith-oracle-filtered-v2": "TaskTrove:laion__swesmith-oracle-filtered-v2",
    "Task Trove:laion__nemotron-gym-math-v5": "TaskTrove:laion__nemotron-gym-math-v5",
    "Task Trove:laion__nemotron-gym-math-openmathreasoning-v2": (
        "TaskTrove:laion__nemotron-gym-math-openmathreasoning-v2"
    ),
    "Task Trove:SankalpKJ__nemotron-math-oracle-filtered-v2": "TaskTrove:SankalpKJ__nemotron-math-oracle-filtered-v2",
    "Task Trove:laion__nemo-prism-math-v3": "TaskTrove:laion__nemo-prism-math-v3",
    "Task Trove:laion__nemotron-gym-math-stack-overflow-v3": "TaskTrove:laion__nemotron-gym-math-stack-overflow-v3",
    "Task Trove:laion__stackexchange-codereview-sandboxes-verified-v2": (
        "TaskTrove:laion__stackexchange-codereview-sandboxes-verified-v2"
    ),
    "Task Trove:laion__glaive-code-assistant-sandboxes-verified-v2": (
        "TaskTrove:laion__glaive-code-assistant-sandboxes-verified-v2"
    ),
    "Task Trove:laion__nemotron-gym-safety-v3": "TaskTrove:laion__nemotron-gym-safety-v3",
    "Task Trove:laion__stackexchange-overflow-sandboxes-verified-v2": (
        "TaskTrove:laion__stackexchange-overflow-sandboxes-verified-v2"
    ),
    "Task Trove:laion__stackexchange-superuser-sandboxes-verified-v2": (
        "TaskTrove:laion__stackexchange-superuser-sandboxes-verified-v2"
    ),
    "Task Trove:laion__stackexchange-tezos-sandboxes-verified-v2": (
        "TaskTrove:laion__stackexchange-tezos-sandboxes-verified-v2"
    ),
    "Task Trove:laion__stackexchange-unix-sandboxes-verified-v2": (
        "TaskTrove:laion__stackexchange-unix-sandboxes-verified-v2"
    ),
    "Task Trove:laion__wizardlm-orca-v4": "TaskTrove:laion__wizardlm-orca-v4",
    "Task Trove:laion__nemotron-gym-knowledge-openqa-v4": "TaskTrove:laion__nemotron-gym-knowledge-openqa-v4",
    "Task Trove:laion__nemotron-gym-science-so-openq-v3": "TaskTrove:laion__nemotron-gym-science-so-openq-v3",
    "Task Trove:laion__nemotron-gym-knowledge-mcqa-v2": "TaskTrove:laion__nemotron-gym-knowledge-mcqa-v2",
    "Task Trove:laion__nemotron-gym-agent-calendar-v2": "TaskTrove:laion__nemotron-gym-agent-calendar-v2",
    "Task Trove:laion__nemotron-gym-instruction-following-calendar-v3": (
        "TaskTrove:laion__nemotron-gym-instruction-following-calendar-v3"
    ),
    "Task Trove:laion__nemotron-gym-instruction-following-v3": "TaskTrove:laion__nemotron-gym-instruction-following-v3",
    "Task Trove:laion__nemotron-gym-instruction-following-structured-v3": (
        "TaskTrove:laion__nemotron-gym-instruction-following-structured-v3"
    ),
    "Task Trove:laion__nemotron-gym-multichallenge-advanced-v4": (
        "TaskTrove:laion__nemotron-gym-multichallenge-advanced-v4"
    ),
    "Task Trove:laion__all-puzzles-v2": "TaskTrove:laion__all-puzzles-v2",
    "Task Trove:laion__nemotron-gym-arc-agi-python-inductive-v2": (
        "TaskTrove:laion__nemotron-gym-arc-agi-python-inductive-v2"
    ),
    "Task Trove:laion__nemotron-gym-arc-agi-transductive-v3": "TaskTrove:laion__nemotron-gym-arc-agi-transductive-v3",
    "MarinSkyRL:reasoning_gym": "Reasoning Gym:reasoning_gym",
    "Task Trove:laion__nemotron-gym-reasoning-gym-v2": "TaskTrove:laion__nemotron-gym-reasoning-gym-v2",
    "MarinSkyRL:nemotron_ultra_mopd/hs3_en": "Nemotron:nemotron_ultra_mopd/hs3_en",
    "MarinSkyRL:nemotron_ultra_mopd/hs3_multi": "Nemotron:nemotron_ultra_mopd/hs3_multi",
    "MarinSkyRL:nemotron_ultra_mopd/hs3_multiturn": "Nemotron:nemotron_ultra_mopd/hs3_multiturn",
    "MarinSkyRL:nemotron_ultra_mopd/makeshn_ultra_v3_ipi_train": (
        "Nemotron:nemotron_ultra_mopd/makeshn_ultra_v3_ipi_train"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/safety_en": "Nemotron:nemotron_ultra_mopd/safety_en",
    (
        "MarinSkyRL:nemotron_ultra_mopd/agent:swe_pivot_single_step_tool_use_with_argument_comparis"
        "on_agent/SWE-Gym/SWE-Gym"
    ): (
        "Nemotron:nemotron_ultra_mopd/agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/SWE-Gym/SWE-Gym"
    ),
    (
        "MarinSkyRL:nemotron_ultra_mopd/agent:swe_pivot_single_step_tool_use_with_argument_comparis"
        "on_agent/nebius/SWE-rebench-V2"
    ): (
        "Nemotron:nemotron_ultra_mopd/agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/nebius/SWE-rebench-V2"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/swe_pivot_len40k/SWE-Gym/SWE-Gym": (
        "Nemotron:nemotron_ultra_mopd/swe_pivot_len40k/SWE-Gym/SWE-Gym"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/swe_pivot_len40k/nebius/SWE-rebench-V2": (
        "Nemotron:nemotron_ultra_mopd/swe_pivot_len40k/nebius/SWE-rebench-V2"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_abstention": (
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_abstention"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_calendar_v2": (
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_calendar_v2"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_comp_coding": (
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_comp_coding"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_instruction_following": (
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_instruction_following"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_jailbreak": (
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_jailbreak"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_lean": "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_lean",
    "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_math_cot": (
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_math_cot"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_math_tir": (
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_math_tir"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_multichallenge_len40k": (
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_multichallenge_len40k"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_nvarc_inductive": (
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_nvarc_inductive"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_nvarc_transductive": (
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_nvarc_transductive"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_rdkit": "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_rdkit",
    "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_reasoning_gym": (
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_reasoning_gym"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_stem_mcqa": (
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_stem_mcqa"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_stem_mcqa_cot_rima_new": (
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_stem_mcqa_cot_rima_new"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_structured_outputs_v2": (
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_structured_outputs_v2"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_toolcall_schema": (
        "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_toolcall_schema"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_citation_format_v2": (
        "Nemotron:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_citation_format_v2"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_freeform_text_v2": (
        "Nemotron:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_freeform_text_v2"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_structured_outputs_v2": (
        "Nemotron:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_structured_outputs_v2"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/SWE-Gym/SWE-Gym": (
        "Nemotron:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/SWE-Gym/SWE-Gym"
    ),
    "MarinSkyRL:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/nebius/SWE-rebench-V2": (
        "Nemotron:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/nebius/SWE-rebench-V2"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr1/language_mixing_hs3_ultra_genrm_fmt": (
        "Nemotron:nemotron_ultra_rlvr1/language_mixing_hs3_ultra_genrm_fmt"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_abstention": (
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_abstention"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_calendar_v2": (
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_calendar_v2"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_comp_coding": (
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_comp_coding"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_instruction_following": (
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_instruction_following"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_jailbreak": (
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_jailbreak"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_lean": "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_lean",
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_math_cot": (
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_math_cot"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_math_tir": (
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_math_tir"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_multichallenge_len40k": (
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_multichallenge_len40k"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_nvarc_inductive": (
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_nvarc_inductive"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_nvarc_transductive": (
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_nvarc_transductive"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_reasoning_gym": (
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_reasoning_gym"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_stem_mcqa": (
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_stem_mcqa"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_stem_mcqa_cot_rima_new": (
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_stem_mcqa_cot_rima_new"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_structured_outputs_v2": (
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_structured_outputs_v2"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym": (
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2": (
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_tau_pivot": (
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_tau_pivot"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_toolcall_schema": (
        "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_toolcall_schema"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/language_mixing_hs3_ultra_genrm_fmt": (
        "Nemotron:nemotron_ultra_rlvr2/language_mixing_hs3_ultra_genrm_fmt"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_abstention": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_abstention"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_calendar_v2": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_calendar_v2"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_comp_coding": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_comp_coding"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_instruction_following": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_instruction_following"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_jailbreak": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_jailbreak"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_lean": "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_lean",
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_math_cot": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_math_cot"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_math_tir": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_math_tir"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_multichallenge_len40k": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_multichallenge_len40k"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_nvarc_inductive": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_nvarc_inductive"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_nvarc_transductive": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_nvarc_transductive"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_reasoning_gym": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_reasoning_gym"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_stem_mcqa": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_stem_mcqa"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_stem_mcqa_cot_rima_new": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_stem_mcqa_cot_rima_new"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_structured_outputs_v2": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_structured_outputs_v2"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_tau_pivot": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_tau_pivot"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_toolcall_schema": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_toolcall_schema"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_ds2_freeform": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_ds2_freeform"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_ds3_citation": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_ds3_citation"
    ),
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_rdkit": "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_rdkit",
    "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_structured_outputs_v3": (
        "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_structured_outputs_v3"
    ),
    "SnowballV01:structure-closure": "OpenAthena:structure-closure",
    "SnowballV01:structure-closure/validation": "OpenAthena:structure-closure/validation",
    "Campaign Release:tasktrove-code-contests-built-image": "TaskTrove:tasktrove-code-contests-built-image",
    "Campaign Release:tasktrove-taco-built-image": "TaskTrove:tasktrove-taco-built-image",
    "Campaign Release:smartie-calibforge-rebuilt-v1": "TaskTrove:smartie-calibforge-rebuilt-v1",
    "Campaign Release:tasktrove-nl2bash-rebuilt-v1": "TaskTrove:tasktrove-nl2bash-rebuilt-v1",
    "Campaign Release:tasktrove-codeforces-rebuilt-v1": "TaskTrove:tasktrove-codeforces-rebuilt-v1",
    "Campaign Release:tasktrove-competitive-coding-rebuilt-v1": "TaskTrove:tasktrove-competitive-coding-rebuilt-v1",
    "Campaign Release:tasktrove-curriculum-easy-rebuilt-v1": "TaskTrove:tasktrove-curriculum-easy-rebuilt-v1",
    "Campaign Release:tasktrove-curriculum-medium-rebuilt-v1": "TaskTrove:tasktrove-curriculum-medium-rebuilt-v1",
    "Campaign Release:tasktrove-e2egit-v2-rebuilt": "TaskTrove:tasktrove-e2egit-v2-rebuilt",
    "Campaign Release:tasktrove-e2egit-large-rebuilt-v1": "TaskTrove:tasktrove-e2egit-large-rebuilt-v1",
    "Campaign Release:tasktrove-multifile-rebuilt-v1": "TaskTrove:tasktrove-multifile-rebuilt-v1",
    "Campaign Release:tasktrove-pymethods2test-rebuilt-v1": "TaskTrove:tasktrove-pymethods2test-rebuilt-v1",
    "Campaign Release:tasktrove-pymethods2test-large-rebuilt-v1": "TaskTrove:tasktrove-pymethods2test-large-rebuilt-v1",
    "Campaign Release:tasktrove-stack-pytest-rebuilt-v1": "TaskTrove:tasktrove-stack-pytest-rebuilt-v1",
    "Campaign Release:tasktrove-unitsyn-python-rebuilt-v1": "TaskTrove:tasktrove-unitsyn-python-rebuilt-v1",
    "Campaign Release:tasktrove-unitsyn-python-large-rebuilt-v1": "TaskTrove:tasktrove-unitsyn-python-large-rebuilt-v1",
    "Campaign Release:swesmith-verifier-repaired-v5": "TaskTrove:swesmith-verifier-repaired-v5",
    "Campaign Release:tasktrove-math_oracle-qa": "TaskTrove:tasktrove-math_oracle-qa",
    "Campaign Release:tasktrove-math_stack-qa": "TaskTrove:tasktrove-math_stack-qa",
    "Campaign Release:tasktrove-math_openreasoning-qa": "TaskTrove:tasktrove-math_openreasoning-qa",
    "Campaign Release:tasktrove-math_gym-qa": "TaskTrove:tasktrove-math_gym-qa",
    "Campaign Release:tasktrove-math_prism-qa": "TaskTrove:tasktrove-math_prism-qa",
    "Campaign Release:tasktrove-safety-qa": "TaskTrove:tasktrove-safety-qa",
    "Campaign Release:tasktrove-knowledge-openqa-qa": "TaskTrove:tasktrove-knowledge-openqa-qa",
    "Campaign Release:tasktrove-science-openqa-qa": "TaskTrove:tasktrove-science-openqa-qa",
}
