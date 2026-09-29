# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical registry of Datakit sources that retain structured conversations."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache, cached_property
from types import MappingProxyType

from marin.datakit.chat_render import render_chat_step
from marin.datakit.download.agenttrove import agenttrove_chat_normalize_steps
from marin.datakit.download.coderforge import coderforge_chat_normalize_steps
from marin.datakit.download.davinci_dev import davinci_dev_env_native_chat_normalize_steps
from marin.datakit.download.glm53_compaction import glm53_compaction_chat_normalize_steps
from marin.datakit.download.glm53_format_following import glm53_format_following_chat_normalize_steps
from marin.datakit.download.glm_kernelgym_rollouts import glm_kernelgym_rollouts_chat_normalize_steps
from marin.datakit.download.gpt_oss_rollouts import gpt_oss_rollouts_chat_normalize_steps
from marin.datakit.download.identity_data import identity_data_chat_normalize_steps
from marin.datakit.download.massive import massive_chat_normalize_steps
from marin.datakit.download.nemotron_sft_v3 import nemotron_sft_v3_chat_normalize_steps
from marin.datakit.download.nemotron_terminal import nemotron_terminal_chat_normalize_steps
from marin.datakit.download.nemotron_v2 import nemotron_sft_chat_normalize_steps
from marin.datakit.download.numinamath_tir import numinamath_tir_chat_normalize_steps
from marin.datakit.download.numinamath_v1_5 import numinamath_v1_5_chat_normalize_steps
from marin.datakit.download.open_swe_traces import OPEN_SWE_TRACES_PARTITIONS, open_swe_traces_chat_normalize_steps
from marin.datakit.download.openthoughts4_code import openthoughts4_code_chat_normalize_steps
from marin.datakit.download.penfever_rollouts import penfever_rollouts_chat_normalize_steps
from marin.datakit.download.science_tool_use_conversations import science_tool_use_chat_normalize_steps
from marin.datakit.download.superior_reasoning import superior_reasoning_chat_normalize_steps
from marin.datakit.download.swe_rebench_openhands import swe_rebench_openhands_chat_normalize_steps
from marin.datakit.download.swe_zero_12m import swe_zero_12m_chat_normalize_steps
from marin.datakit.download.synthetic1 import synthetic1_chat_normalize_steps
from marin.datakit.download.synthetic_misconceptions import synthetic_misconceptions_chat_normalize_steps
from marin.datakit.download.ultrachat_persona import ultrachat_persona_chat_normalize_steps
from marin.datakit.normalize import normalize_step
from marin.datakit.sources import all_sources
from marin.execution.step_spec import StepSpec


@dataclass(frozen=True)
class DatakitChatSource:
    """An SFT source with structured chat and normalized rendered-text artifacts."""

    name: str
    chat_steps: tuple[StepSpec, ...]
    rough_token_count_b: float

    @property
    def chat_normalized(self) -> StepSpec:
        return self.chat_steps[-1]

    @cached_property
    def rendered(self) -> StepSpec:
        return render_chat_step(name=f"rendered/sft/{self.name}", chat=self.chat_normalized)

    @cached_property
    def normalized(self) -> StepSpec:
        return normalize_step(name=f"normalized/sft/{self.name}", download=self.rendered)

    @property
    def normalize_steps(self) -> tuple[StepSpec, ...]:
        return (*self.chat_steps, self.rendered, self.normalized)


_EXCLUDED_CHAT_SOURCES = frozenset(
    {
        # These exports omit the original user request and the literal prompts
        # needed to recover it and the served tool definitions.
        "penfever-traces/qwen35-122b-131k-opencode/nemotron-gym-agent-workplace-v2",
        "penfever-traces/qwen35-122b-131k-opencode/selfinstruct-naive-sandboxes-2-verified",
        # These traces teach the assistant to identify as Nemotron 3 Super.
        "penfever-traces/minimax-m27-131k/nemotron-gym-identity-following-v2",
        "penfever-traces/qwen35-122b-131k-opencode/nemotron-gym-identity-following-v2",
        "penfever-traces/qwen35-122b-32k/nemotron-gym-identity-following-v2",
    }
)
_ChatSourceRow = tuple[str, Callable[[], tuple[StepSpec, ...]]]


# Measured rendered-text token counts in billions for the immutable tokenizer at
# gs://marin-us-central2/grug_sft/tokenizer/2026.09.18.
NEMOTRON_SFT_V3_TOKEN_COUNTS_B: Mapping[str, float] = MappingProxyType(
    {
        "nemotron_sft_v3/agentic_v1/interactive_agent": 0.090344730,
        "nemotron_sft_v3/agentic_v1/tool_calling": 0.737625081,
        "nemotron_sft_v3/agentic_v2/interactive_agent": 1.354234222,
        "nemotron_sft_v3/agentic_v2/search": 0.143480016,
        "nemotron_sft_v3/agentic_v2/tool_calling": 0.041881674,
        "nemotron_sft_v3/arc_agi_v1/large_no_reasoning_no_tools": 0.007506959,
        "nemotron_sft_v3/arc_agi_v1/large_reasoning_and_tools": 1.712885051,
        "nemotron_sft_v3/arc_agi_v1/large_reasoning_no_tools": 4.306871635,
        "nemotron_sft_v3/arc_agi_v1/large_tools_no_reasoning": 0.020293208,
        "nemotron_sft_v3/arc_agi_v1/small_no_reasoning_no_tools": 0.000757604,
        "nemotron_sft_v3/arc_agi_v1/small_reasoning_and_tools": 0.726212031,
        "nemotron_sft_v3/arc_agi_v1/small_reasoning_no_tools": 1.860839337,
        "nemotron_sft_v3/arc_agi_v1/small_tools_no_reasoning": 0.002025923,
        "nemotron_sft_v3/competitive_programming_v2/competitive_coding_cpp": 12.982922496,
        "nemotron_sft_v3/competitive_programming_v2/competitive_coding_python": 13.055593856,
        "nemotron_sft_v3/competitive_programming_v2/exercism": 0.156045216,
        "nemotron_sft_v3/competitive_programming_v2/text_to_sql": 0.334121757,
        "nemotron_sft_v3/cuda_v1/train": 0.021285717,
        "nemotron_sft_v3/finance_v1/train": 9.430845042,
        "nemotron_sft_v3/instruction_following_chat_v2/reasoning_off": 1.371617196,
        "nemotron_sft_v3/instruction_following_chat_v2/reasoning_on": 2.360933725,
        "nemotron_sft_v3/instruction_following_chat_v3/chat": 3.550414800,
        "nemotron_sft_v3/instruction_following_chat_v3/instruction_following": 0.721726105,
        "nemotron_sft_v3/math_proofs_v1/lean": 9.915510520,
        "nemotron_sft_v3/math_proofs_v2/train": 5.196848645,
        "nemotron_sft_v3/math_v2/high": 16.614002955,
        "nemotron_sft_v3/math_v2/low": 2.391394873,
        "nemotron_sft_v3/math_v2/medium": 6.976018726,
        "nemotron_sft_v3/math_v3/train": 48.888074150,
        "nemotron_sft_v3/math_v4/train": 6.587456753,
        "nemotron_sft_v3/multilingual_v1/code_de": 1.711805655,
        "nemotron_sft_v3/multilingual_v1/code_es": 1.680588999,
        "nemotron_sft_v3/multilingual_v1/code_fr": 1.747907034,
        "nemotron_sft_v3/multilingual_v1/code_it": 1.828054555,
        "nemotron_sft_v3/multilingual_v1/code_ja": 1.602499366,
        "nemotron_sft_v3/multilingual_v1/code_zh": 1.941448441,
        "nemotron_sft_v3/multilingual_v1/math_de": 2.462788745,
        "nemotron_sft_v3/multilingual_v1/math_es": 1.942946328,
        "nemotron_sft_v3/multilingual_v1/math_fr": 2.196381081,
        "nemotron_sft_v3/multilingual_v1/math_it": 2.516674311,
        "nemotron_sft_v3/multilingual_v1/math_ja": 1.856858377,
        "nemotron_sft_v3/multilingual_v1/math_zh": 1.668189501,
        "nemotron_sft_v3/multilingual_v1/stem_de": 0.727808812,
        "nemotron_sft_v3/multilingual_v1/stem_es": 0.729671020,
        "nemotron_sft_v3/multilingual_v1/stem_fr": 0.737324391,
        "nemotron_sft_v3/multilingual_v1/stem_it": 0.742269241,
        "nemotron_sft_v3/multilingual_v1/stem_ja": 0.711069489,
        "nemotron_sft_v3/multilingual_v1/stem_zh": 0.703348874,
        "nemotron_sft_v3/multilingual_v2/code_hi": 0.437100301,
        "nemotron_sft_v3/multilingual_v2/code_ja": 0.407732137,
        "nemotron_sft_v3/multilingual_v2/code_ko": 0.413438172,
        "nemotron_sft_v3/multilingual_v2/code_pt": 0.419953248,
        "nemotron_sft_v3/multilingual_v2/math_hi": 0.286064954,
        "nemotron_sft_v3/multilingual_v2/math_ja": 0.553625228,
        "nemotron_sft_v3/multilingual_v2/math_ko": 0.415028518,
        "nemotron_sft_v3/multilingual_v2/math_pt": 0.344709576,
        "nemotron_sft_v3/multilingual_v2/stem_hi": 0.063027363,
        "nemotron_sft_v3/multilingual_v2/stem_ja": 0.015674932,
        "nemotron_sft_v3/multilingual_v2/stem_ko": 0.041430982,
        "nemotron_sft_v3/multilingual_v2/stem_pt": 0.280376643,
        "nemotron_sft_v3/opencode_v1/agent_skills": 1.161088544,
        "nemotron_sft_v3/opencode_v1/agent_skills_question_tool": 0.700516195,
        "nemotron_sft_v3/opencode_v1/bash_only_tool": 1.369427891,
        "nemotron_sft_v3/opencode_v1/bash_only_tool_skills": 1.254575003,
        "nemotron_sft_v3/opencode_v1/general": 1.146536337,
        "nemotron_sft_v3/opencode_v1/question_tool": 1.158644980,
        "nemotron_sft_v3/safety_v1/train": 0.029529220,
        "nemotron_sft_v3/safety_v2/train": 0.104927031,
        "nemotron_sft_v3/science_v2/rqa": 0.096585137,
        "nemotron_sft_v3/science_v2/so": 0.065203089,
        "nemotron_sft_v3/science_v2/syn_mcq": 0.002716428,
        "nemotron_sft_v3/science_v2/vendor": 8.221738516,
        "nemotron_sft_v3/swe_v1/r2e_gym": 2.346331616,
        "nemotron_sft_v3/swe_v2/agentless": 1.470548931,
        "nemotron_sft_v3/swe_v2/openhands_swe": 2.119534682,
    }
)


# Measured token counts for the other registered SFT chat sources, in billions,
# using gs://marin-us-central2/grug_sft/tokenizer/2026.09.18.
MEASURED_OTHER_SFT_TOKEN_COUNTS_B: Mapping[str, float] = MappingProxyType(
    {
        "agenttrove-glm53-compactions": 0.188537557,
        "coderforge": 10.986776530,
        "davinci-dev/env-native": 2.375828678,
        "glm-5.2-kernelgym-rollouts": 0.016082348,
        "gpt-oss-rollouts": 3.204823754,
        "massive_function_calling": 11.183690013,
        "nemotron_sft/sft_code": 27.200300012,
        "nemotron_sft/sft_general": 2.870621017,
        "nemotron_sft/sft_math": 98.839751146,
        "numinamath-1.5": 0.356560802,
        "numinamath-tir": 0.065277801,
        "open_swe_traces/v1_0/openhands/minimax_m25/swe_rebench_v2": 2.422931014,
        "open_swe_traces/v1_0/openhands/qwen35_122b/swe_rebench_v2": 2.644151452,
        "open_swe_traces/v1_0/sweagent/minimax_m25/swe_rebench_v2": 2.799081838,
        "open_swe_traces/v1_0/sweagent/qwen35_122b/swe_rebench_v2": 1.627076822,
        "open_swe_traces/v1_1/minisweagent/qwen36_27b/scale_swe": 1.387963054,
        "open_swe_traces/v1_1/minisweagent/qwen36_27b/swe_rebench_v2": 1.761409398,
        "open_swe_traces/v1_1/openhands/deepseek_v4_flash/scale_swe": 1.276258801,
        "open_swe_traces/v1_1/openhands/qwen36_27b/scale_swe": 1.985336125,
        "open_swe_traces/v1_1/openhands/qwen36_27b/swe_rebench_v2": 1.845476368,
        "open_swe_traces/v1_1/sweagent/qwen36_27b/scale_swe": 2.450594544,
        "open_swe_traces/v1_1/sweagent/qwen36_27b/swe_rebench_v2": 2.461787471,
        "open_swe_traces/v1_2/minisweagent/qwen38_27b/scale_swe": 3.306329880,
        "open_swe_traces/v1_2/minisweagent/qwen38_27b/swe_rebench_v2": 3.761200521,
        "openthoughts4-code-glm-5.2-n4": 3.353670934,
        "penfever-traces/minimax-m27-131k/nemotron-gym-identity-following-v2": 0.035989230,
        "penfever-traces/qwen35-122b-131k-opencode/code-contests-noblock": 0.044463910,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rle_adversarial": 0.006573892,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_crosscodeeval-csharp-v4": 0.014228901,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_crosscodeeval-java": 0.001599013,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_curriculum-easy": 0.002554790,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_curriculum-hard": 0.000185097,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_curriculum-medium": 0.003100840,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_e2egit-large": 0.017306906,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_e2egit-v2": 0.001723603,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_ghactions-v3": 0.062400079,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_issue": 0.001192573,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_methods2test-large-v3": 0.000015429,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_multifile": 0.000809012,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_nemotron-cpp-v2": 0.000125018,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_nemotron-junit": 0.000947660,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_pr": 0.018629793,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_pymethods2test-large": 0.040545891,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_pymethods2test-v3": 0.003845456,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_stack-junit-v6": 0.000051424,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_stack-pytest-large": 0.004860374,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_stack-pytest-v2": 0.001761856,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_unitsyn-python-large": 0.017473865,
        "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_unitsyn-python-v3": 0.001050559,
        "penfever-traces/qwen35-122b-131k-opencode/inferredbugs-sandboxes-verifier": 0.009116056,
        "penfever-traces/qwen35-122b-131k-opencode/llm-verifier-freelancer": 0.002921330,
        "penfever-traces/qwen35-122b-131k-opencode/mix_h4_binary_easy": 0.007817765,
        "penfever-traces/qwen35-122b-131k-opencode/nemotron-code-oracle-filtered": 0.053080530,
        "penfever-traces/qwen35-122b-131k-opencode/nemotron-gym-agent-calendar": 0.033464451,
        "penfever-traces/qwen35-122b-131k-opencode/nemotron-gym-identity-following-v2": 0.059976508,
        "penfever-traces/qwen35-122b-131k-opencode/nemotron-gym-instruction-following-structured": 0.088983029,
        "penfever-traces/qwen35-122b-131k-opencode/nemotron-gym-knowledge-web-search-mcqa": 0.022687578,
        "penfever-traces/qwen35-122b-131k-opencode/nemotron-gym-math-advanced-calculations-v3": 0.042108623,
        "penfever-traces/qwen35-122b-131k-opencode/nl2bash-tasks-cleaned-oracle": 0.004214928,
        "penfever-traces/qwen35-122b-32k/nemotron-gym-identity-following-v2": 0.015429027,
        "superior-reasoning": 7.088558993,
        "swe-rebench-openhands": 3.479750674,
        "swe-zero-12m": 2.571849160,
        "synthetic-1": 5.448340497,
        "wildchat-glm53-format-completions": 0.009506138,
    }
)


# Measured from the 2026-09-26 corrected Terminus token outputs using
# gs://marin-us-central2/grug_sft/tokenizer/2026.09.18.
TERMINUS_TOKEN_COUNTS_B: Mapping[str, float] = MappingProxyType(
    {
        "agenttrove": 5.054921190,
        "nemotron-terminal": 2.417981372,
        "penfever-traces/glm52-terminus2/exp_rpt_crosscodeeval-csharp-v4": 0.007763540,
        "penfever-traces/glm52-terminus2/exp_rpt_curriculum-easy": 0.002724970,
        "penfever-traces/glm52-terminus2/exp_rpt_curriculum-medium": 0.003416055,
        "penfever-traces/glm52-terminus2/exp_rpt_e2egit-large": 0.012261902,
        "penfever-traces/glm52-terminus2/exp_rpt_e2egit-v2": 0.001881241,
        "penfever-traces/glm52-terminus2/exp_rpt_nemotron-cpp-v2": 0.005741909,
        "penfever-traces/glm52-terminus2/exp_rpt_stack-pytest-large-v2": 0.014318938,
        "penfever-traces/glm52-terminus2/exp_rpt_stack-pytest-v2": 0.003290955,
        "penfever-traces/glm52-terminus2/exp_rpt_unitsyn-python-v3": 0.001396703,
        "penfever-traces/glm52-terminus2/nemotron-gym-agent-calendar": 0.010161822,
        "penfever-traces/glm52-terminus2/nemotron-gym-instruction-following-structured": 0.044428180,
        "penfever-traces/glm52-terminus2/nemotron-gym-knowledge-web-search-mcqa": 0.003064997,
        "penfever-traces/glm52-terminus2/nl2bash-tasks-cleaned-oracle": 0.005197675,
        "penfever-traces/minimax-m27-131k/code-contests-noblock": 0.038527744,
        "penfever-traces/minimax-m27-131k/exp_rle_minimal_instructions-v3": 0.003275117,
        "penfever-traces/minimax-m27-131k/exp_rpt_codenet-python-v2": 0.048793282,
        "penfever-traces/minimax-m27-131k/exp_rpt_crosscodeeval-csharp-v4": 0.006091783,
        "penfever-traces/minimax-m27-131k/exp_rpt_curriculum-easy": 0.004797033,
        "penfever-traces/minimax-m27-131k/exp_rpt_curriculum-medium": 0.006288057,
        "penfever-traces/minimax-m27-131k/exp_rpt_e2egit-large": 0.019748923,
        "penfever-traces/minimax-m27-131k/exp_rpt_e2egit-v2": 0.002765169,
        "penfever-traces/minimax-m27-131k/exp_rpt_ghactions-v3": 0.062812753,
        "penfever-traces/minimax-m27-131k/exp_rpt_methods2test-large-v2": 0.105255647,
        "penfever-traces/minimax-m27-131k/exp_rpt_methods2test-large-v3": 0.104977176,
        "penfever-traces/minimax-m27-131k/exp_rpt_nemotron-cpp": 0.069803842,
        "penfever-traces/minimax-m27-131k/exp_rpt_nemotron-junit": 0.054174474,
        "penfever-traces/minimax-m27-131k/exp_rpt_pr": 0.071400490,
        "penfever-traces/minimax-m27-131k/exp_rpt_pymethods2test-large": 0.034471412,
        "penfever-traces/minimax-m27-131k/exp_rpt_pymethods2test-v3": 0.002680474,
        "penfever-traces/minimax-m27-131k/exp_rpt_stack-bash-v3": 0.111285830,
        "penfever-traces/minimax-m27-131k/exp_rpt_stack-junit-v6": 0.012496678,
        "penfever-traces/minimax-m27-131k/exp_rpt_stack-pytest-large": 0.060265337,
        "penfever-traces/minimax-m27-131k/exp_rpt_stack-pytest-v2": 0.006211463,
        "penfever-traces/minimax-m27-131k/exp_rpt_unitsyn-python-large": 0.034170635,
        "penfever-traces/minimax-m27-131k/exp_rpt_unitsyn-python-v3": 0.003439536,
        "penfever-traces/minimax-m27-131k/inferredbugs-sandboxes-verifier": 0.064922207,
        "penfever-traces/minimax-m27-131k/llm-verifier-freelancer": 0.078738881,
        "penfever-traces/minimax-m27-131k/mix_h10_reward_binary-v2": 0.027031444,
        "penfever-traces/minimax-m27-131k/mix_h10_reward_proportional-v2": 0.026966397,
        "penfever-traces/minimax-m27-131k/mix_h10_reward_staged-v2": 0.031186341,
        "penfever-traces/minimax-m27-131k/mix_h11_single_skill_only-v2": 0.026944254,
        "penfever-traces/minimax-m27-131k/mix_h1_struggle_zone-v2": 0.031174649,
        "penfever-traces/minimax-m27-131k/mix_h2_language_balanced-v2": 0.067459192,
        "penfever-traces/minimax-m27-131k/mix_h2_language_proportional": 0.062582266,
        "penfever-traces/minimax-m27-131k/mix_h4_binary_easy": 0.020682838,
        "penfever-traces/minimax-m27-131k/mix_h8_original_tests-v2": 0.026625856,
        "penfever-traces/minimax-m27-131k/nemotron-code-oracle-filtered": 0.062069757,
        "penfever-traces/minimax-m27-131k/nemotron-gym-agent-calendar": 0.025046569,
        "penfever-traces/minimax-m27-131k/nemotron-gym-agent-workplace-v2": 0.001676944,
        "penfever-traces/minimax-m27-131k/nemotron-gym-competitive-coding": 0.044303759,
        "penfever-traces/minimax-m27-131k/nemotron-gym-instruction-following-calendar": 0.077819736,
        "penfever-traces/minimax-m27-131k/nemotron-gym-instruction-following-structured": 0.077532504,
        "penfever-traces/minimax-m27-131k/nemotron-gym-knowledge-web-search-mcqa": 0.007708756,
        "penfever-traces/minimax-m27-131k/nemotron-gym-math-advanced-calculations-v3": 0.021164537,
        "penfever-traces/minimax-m27-131k/nl2bash-tasks-cleaned-oracle": 0.006619822,
        "penfever-traces/minimax-m27-131k/selfinstruct-naive-sandboxes-2-verified": 0.068857293,
        "penfever-traces/minimax-m27-131k/swegym-tasks-patched-validated-v5": 0.048303534,
        "penfever-traces/qwen35-122b-32k/code-contests-noblock": 0.053630203,
        "penfever-traces/qwen35-122b-32k/exp_rle_minimal_instructions-v3": 0.003415577,
        "penfever-traces/qwen35-122b-32k/exp_rpt_codenet-python-v2": 0.057797510,
        "penfever-traces/qwen35-122b-32k/exp_rpt_crosscodeeval-csharp-v4": 0.006771478,
        "penfever-traces/qwen35-122b-32k/exp_rpt_curriculum-easy": 0.004605129,
        "penfever-traces/qwen35-122b-32k/exp_rpt_curriculum-medium": 0.006147061,
        "penfever-traces/qwen35-122b-32k/exp_rpt_e2egit-large": 0.020951210,
        "penfever-traces/qwen35-122b-32k/exp_rpt_e2egit-v2": 0.003031681,
        "penfever-traces/qwen35-122b-32k/exp_rpt_ghactions-v3": 0.073185787,
        "penfever-traces/qwen35-122b-32k/exp_rpt_methods2test-large-v2": 0.090731177,
        "penfever-traces/qwen35-122b-32k/exp_rpt_methods2test-large-v3": 0.051730474,
        "penfever-traces/qwen35-122b-32k/exp_rpt_nemotron-junit": 0.058441172,
        "penfever-traces/qwen35-122b-32k/exp_rpt_pr": 0.095949224,
        "penfever-traces/qwen35-122b-32k/exp_rpt_pymethods2test-large": 0.037565125,
        "penfever-traces/qwen35-122b-32k/exp_rpt_pymethods2test-v3": 0.003848395,
        "penfever-traces/qwen35-122b-32k/exp_rpt_stack-bash-v3": 0.122502322,
        "penfever-traces/qwen35-122b-32k/exp_rpt_stack-junit-v6": 0.004301224,
        "penfever-traces/qwen35-122b-32k/exp_rpt_stack-pytest-large": 0.062553159,
        "penfever-traces/qwen35-122b-32k/exp_rpt_stack-pytest-v2": 0.006653146,
        "penfever-traces/qwen35-122b-32k/exp_rpt_unitsyn-python-large": 0.037879024,
        "penfever-traces/qwen35-122b-32k/exp_rpt_unitsyn-python-v3": 0.004042406,
        "penfever-traces/qwen35-122b-32k/inferredbugs-sandboxes-verifier": 0.087071590,
        "penfever-traces/qwen35-122b-32k/llm-verifier-freelancer": 0.028095865,
        "penfever-traces/qwen35-122b-32k/mix_h10_reward_binary-v2": 0.015701374,
        "penfever-traces/qwen35-122b-32k/mix_h10_reward_proportional-v2": 0.025554480,
        "penfever-traces/qwen35-122b-32k/mix_h10_reward_staged-v2": 0.029458565,
        "penfever-traces/qwen35-122b-32k/mix_h11_single_skill_only-v2": 0.024528988,
        "penfever-traces/qwen35-122b-32k/mix_h1_struggle_zone-v2": 0.030154504,
        "penfever-traces/qwen35-122b-32k/mix_h2_language_balanced-v2": 0.068338226,
        "penfever-traces/qwen35-122b-32k/mix_h2_language_proportional": 0.063356462,
        "penfever-traces/qwen35-122b-32k/mix_h4_binary_easy": 0.019099150,
        "penfever-traces/qwen35-122b-32k/mix_h8_original_tests-v2": 0.025351208,
        "penfever-traces/qwen35-122b-32k/nemotron-code-oracle-filtered": 0.077597630,
        "penfever-traces/qwen35-122b-32k/nemotron-gym-agent-calendar": 0.024257797,
        "penfever-traces/qwen35-122b-32k/nemotron-gym-agent-workplace-v2": 0.001688347,
        "penfever-traces/qwen35-122b-32k/nemotron-gym-competitive-coding": 0.043253033,
        "penfever-traces/qwen35-122b-32k/nemotron-gym-instruction-following-calendar": 0.074258719,
        "penfever-traces/qwen35-122b-32k/nemotron-gym-instruction-following-structured": 0.068045570,
        "penfever-traces/qwen35-122b-32k/nemotron-gym-instruction-following-v2": 0.127582322,
        "penfever-traces/qwen35-122b-32k/nemotron-gym-knowledge-mcqa": 0.026694853,
        "penfever-traces/qwen35-122b-32k/nemotron-gym-knowledge-openqa-v2": 0.021894806,
        "penfever-traces/qwen35-122b-32k/nemotron-gym-knowledge-web-search-mcqa": 0.009041425,
        "penfever-traces/qwen35-122b-32k/nemotron-gym-math-advanced-calculations-v3": 0.016246222,
        "penfever-traces/qwen35-122b-32k/nemotron-gym-safety-v2": 0.059125333,
        "penfever-traces/qwen35-122b-32k/nemotron-math-oracle-filtered": 0.096336289,
        "penfever-traces/qwen35-122b-32k/nl2bash-tasks-cleaned-oracle": 0.006787999,
        "penfever-traces/qwen35-122b-32k/selfinstruct-naive-sandboxes-2-verified": 0.009110740,
        "penfever-traces/qwen35-122b-32k/swesmith-oracle-filtered": 0.015750459,
    }
)


@cache
def all_sft_sources() -> dict[str, DatakitChatSource]:
    """Return SFT sources with Harmony-to-text normalization pipelines."""
    penfever_steps = cache(penfever_rollouts_chat_normalize_steps)
    nemotron_steps = cache(nemotron_sft_chat_normalize_steps)
    nemotron_v3_steps = cache(nemotron_sft_v3_chat_normalize_steps)
    rows: list[_ChatSourceRow] = [
        ("agenttrove", agenttrove_chat_normalize_steps),
        ("agenttrove-glm53-compactions", glm53_compaction_chat_normalize_steps),
        ("wildchat-glm53-format-completions", glm53_format_following_chat_normalize_steps),
        ("coderforge", coderforge_chat_normalize_steps),
        ("davinci-dev/env-native", davinci_dev_env_native_chat_normalize_steps),
        ("glm-5.2-kernelgym-rollouts", glm_kernelgym_rollouts_chat_normalize_steps),
        ("gpt-oss-rollouts", gpt_oss_rollouts_chat_normalize_steps),
        ("identity-data", identity_data_chat_normalize_steps),
        ("massive_function_calling", massive_chat_normalize_steps),
        ("nemotron-terminal", nemotron_terminal_chat_normalize_steps),
        ("numinamath-1.5", numinamath_v1_5_chat_normalize_steps),
        ("numinamath-tir", numinamath_tir_chat_normalize_steps),
        ("openthoughts4-code-glm-5.2-n4", openthoughts4_code_chat_normalize_steps),
        ("science-tool-use-conversations", science_tool_use_chat_normalize_steps),
        ("superior-reasoning", superior_reasoning_chat_normalize_steps),
        ("swe-rebench-openhands", swe_rebench_openhands_chat_normalize_steps),
        ("swe-zero-12m", swe_zero_12m_chat_normalize_steps),
        ("synthetic-1", synthetic1_chat_normalize_steps),
        ("synthetic-misconceptions-conversations", synthetic_misconceptions_chat_normalize_steps),
        ("ultrachat-persona-conversations", ultrachat_persona_chat_normalize_steps),
    ]
    rows.extend(
        (name, lambda source_name=name: penfever_steps()[source_name])
        for name in all_sources()
        if name.startswith("penfever-traces/") and name not in _EXCLUDED_CHAT_SOURCES
    )
    rows.extend(
        (name, lambda source_name=name: nemotron_steps()[source_name])
        for name in all_sources()
        if name.startswith("nemotron_sft/")
    )
    rows.extend(
        (name, lambda source_name=name: open_swe_traces_chat_normalize_steps(source_name))
        for name in OPEN_SWE_TRACES_PARTITIONS
    )

    v3_chains = nemotron_v3_steps()
    if set(v3_chains) != set(NEMOTRON_SFT_V3_TOKEN_COUNTS_B):
        raise ValueError("Nemotron v3 token counts must cover every registered chat partition")
    rows.extend((name, lambda source_name=name: v3_chains[source_name]) for name in v3_chains)

    token_counts = {name: source.rough_token_count_b for name, source in all_sources().items()}
    token_counts.update(NEMOTRON_SFT_V3_TOKEN_COUNTS_B)
    token_counts.update({name: size for name, (_, size) in OPEN_SWE_TRACES_PARTITIONS.items()})
    token_counts.update(MEASURED_OTHER_SFT_TOKEN_COUNTS_B)
    token_counts.update(TERMINUS_TOKEN_COUNTS_B)
    token_counts["synthetic-misconceptions-conversations"] = 0.001704859
    token_counts["identity-data"] = 0.061711380
    # Reproduced locally on the pinned source with chat normalization, rendering,
    # whitespace compaction, and BOS/EOS tokenization; artifact validation is pending.
    token_counts["science-tool-use-conversations"] = 0.020855889
    token_counts["ultrachat-persona-conversations"] = 0.324455701
    return {
        name: DatakitChatSource(
            name=name,
            chat_steps=factory(),
            rough_token_count_b=token_counts[name],
        )
        for name, factory in rows
    }
