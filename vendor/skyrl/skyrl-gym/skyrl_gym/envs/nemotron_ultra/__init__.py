"""Nemotron 3 Ultra task algorithms and source profiles.

Verifier algorithms in this package are derived from NVIDIA NeMo Gym commit
6283f37f83f727561ce59e0012c00dcd9780a423 and retain its Apache-2.0 notices.
"""

GENRM_AGENTS = frozenset({"genrm_simple_agent", "genrm_simple_agent_reasoning_off"})

NEMOTRON_ULTRA_RLVR1_AGENTS = frozenset(
    {
        "abstention_simple_agent",
        "calendar_simple_agent",
        "code_gen_simple_agent",
        "genrm_simple_agent",
        "genrm_simple_agent_reasoning_off",
        "instruction_following_simple_agent",
        "jailbreak_engagement_with_disclaimer",
        "jailbreak_hard_refusal_no_redirection",
        "jailbreak_hard_refusal_with_helplines",
        "jailbreak_refusal_with_explanation",
        "math_formal_lean_refinement_agent",
        "math_with_judge_simple_agent",
        "mcqa_simple_agent",
        "multichallenge_simple_agent",
        "ns_tools_simple_agent",
        "nvarc_inductive_simple_agent",
        "nvarc_transductive_simple_agent",
        "reasoning_gym_simple_agent",
        "single_step_tool_use_with_argument_comparison_agent",
        "structured_outputs_simple_agent",
        "swe_pivot_single_step_tool_use_with_argument_comparison_agent",
        "toolcall_schema_single_step_tool_use_with_argument_comparison_agent",
    }
)
NEMOTRON_ULTRA_RLVR2_AGENTS = NEMOTRON_ULTRA_RLVR1_AGENTS | {
    "citation_format_simple_agent",
    "freeform_formatting_simple_agent",
    "indirect_prompt_injection_simple_agent",
    "rdkit_chemistry_agent",
    "structured_outputs_v3_simple_agent",
}

NEMOTRON_ULTRA_MOPD_AGENTS = NEMOTRON_ULTRA_RLVR2_AGENTS
