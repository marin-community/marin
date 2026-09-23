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
from marin.datakit.download.massive import massive_chat_normalize_steps
from marin.datakit.download.nemotron_sft_v3 import nemotron_sft_v3_chat_normalize_steps
from marin.datakit.download.nemotron_terminal import nemotron_terminal_chat_normalize_steps
from marin.datakit.download.nemotron_v2 import nemotron_sft_chat_normalize_steps
from marin.datakit.download.numinamath_tir import numinamath_tir_chat_normalize_steps
from marin.datakit.download.numinamath_v1_5 import numinamath_v1_5_chat_normalize_steps
from marin.datakit.download.open_swe_traces import OPEN_SWE_TRACES_PARTITIONS, open_swe_traces_chat_normalize_steps
from marin.datakit.download.openthoughts4_code import openthoughts4_code_chat_normalize_steps
from marin.datakit.download.penfever_rollouts import penfever_rollouts_chat_normalize_steps
from marin.datakit.download.superior_reasoning import superior_reasoning_chat_normalize_steps
from marin.datakit.download.swe_rebench_openhands import swe_rebench_openhands_chat_normalize_steps
from marin.datakit.download.swe_zero_12m import swe_zero_12m_chat_normalize_steps
from marin.datakit.download.synthetic1 import synthetic1_chat_normalize_steps
from marin.datakit.download.synthetic_misconceptions import synthetic_misconceptions_chat_normalize_steps
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
    }
)
_ChatSourceRow = tuple[str, Callable[[], tuple[StepSpec, ...]]]


# These rough mixture weights precede chat conversion and filtering. Safety-v1,
# Math-v3, and Multilingual-v1 Japanese use file-size estimates until materialization.
NEMOTRON_SFT_V3_TOKEN_COUNTS_B: Mapping[str, float] = MappingProxyType(
    {
        "nemotron_sft_v3/agentic_v1/interactive_agent": 0.083861191,
        "nemotron_sft_v3/agentic_v1/tool_calling": 0.907518925,
        "nemotron_sft_v3/agentic_v2/interactive_agent": 1.227454026,
        "nemotron_sft_v3/agentic_v2/search": 0.147017563,
        "nemotron_sft_v3/agentic_v2/tool_calling": 0.038076797,
        "nemotron_sft_v3/arc_agi_v1/large_no_reasoning_no_tools": 0.007520769,
        "nemotron_sft_v3/arc_agi_v1/large_reasoning_and_tools": 1.927429903,
        "nemotron_sft_v3/arc_agi_v1/large_reasoning_no_tools": 4.317173865,
        "nemotron_sft_v3/arc_agi_v1/large_tools_no_reasoning": 0.020800514,
        "nemotron_sft_v3/arc_agi_v1/small_no_reasoning_no_tools": 0.000758374,
        "nemotron_sft_v3/arc_agi_v1/small_reasoning_and_tools": 0.873573674,
        "nemotron_sft_v3/arc_agi_v1/small_reasoning_no_tools": 1.867015171,
        "nemotron_sft_v3/arc_agi_v1/small_tools_no_reasoning": 0.002010211,
        "nemotron_sft_v3/competitive_programming_v2/competitive_coding_cpp": 13.010566683,
        "nemotron_sft_v3/competitive_programming_v2/competitive_coding_python": 13.074348829,
        "nemotron_sft_v3/competitive_programming_v2/exercism": 0.160088736,
        "nemotron_sft_v3/competitive_programming_v2/text_to_sql": 0.335764474,
        "nemotron_sft_v3/cuda_v1/train": 0.024404247,
        "nemotron_sft_v3/finance_v1/train": 9.434139581,
        "nemotron_sft_v3/instruction_following_chat_v2/reasoning_off": 1.380229248,
        "nemotron_sft_v3/instruction_following_chat_v2/reasoning_on": 2.376385569,
        "nemotron_sft_v3/instruction_following_chat_v3/chat": 1.740197225,
        "nemotron_sft_v3/instruction_following_chat_v3/instruction_following": 0.730309827,
        "nemotron_sft_v3/math_proofs_v1/lean": 9.926555549,
        "nemotron_sft_v3/math_proofs_v2/train": 5.197706993,
        "nemotron_sft_v3/math_v2/high": 17.52267726,
        "nemotron_sft_v3/math_v2/low": 2.786466814,
        "nemotron_sft_v3/math_v2/medium": 8.76628449,
        "nemotron_sft_v3/math_v3/train": 38.5,
        "nemotron_sft_v3/math_v4/train": 6.590191116,
        "nemotron_sft_v3/multilingual_v1/code_de": 1.713213187,
        "nemotron_sft_v3/multilingual_v1/code_es": 1.682083794,
        "nemotron_sft_v3/multilingual_v1/code_fr": 1.749340268,
        "nemotron_sft_v3/multilingual_v1/code_it": 1.829606998,
        "nemotron_sft_v3/multilingual_v1/code_ja": 1.75,
        "nemotron_sft_v3/multilingual_v1/code_zh": 1.943310105,
        "nemotron_sft_v3/multilingual_v1/math_de": 2.464101656,
        "nemotron_sft_v3/multilingual_v1/math_es": 1.943996815,
        "nemotron_sft_v3/multilingual_v1/math_fr": 2.197566332,
        "nemotron_sft_v3/multilingual_v1/math_it": 2.518006207,
        "nemotron_sft_v3/multilingual_v1/math_ja": 2.2,
        "nemotron_sft_v3/multilingual_v1/math_zh": 1.669082185,
        "nemotron_sft_v3/multilingual_v1/stem_de": 0.730538293,
        "nemotron_sft_v3/multilingual_v1/stem_es": 0.732458657,
        "nemotron_sft_v3/multilingual_v1/stem_fr": 0.740151717,
        "nemotron_sft_v3/multilingual_v1/stem_it": 0.745118255,
        "nemotron_sft_v3/multilingual_v1/stem_ja": 0.74,
        "nemotron_sft_v3/multilingual_v1/stem_zh": 0.706092417,
        "nemotron_sft_v3/multilingual_v2/code_hi": 0.437524235,
        "nemotron_sft_v3/multilingual_v2/code_ja": 0.40850654,
        "nemotron_sft_v3/multilingual_v2/code_ko": 0.413803472,
        "nemotron_sft_v3/multilingual_v2/code_pt": 0.420317804,
        "nemotron_sft_v3/multilingual_v2/math_hi": 0.28626839,
        "nemotron_sft_v3/multilingual_v2/math_ja": 0.553978973,
        "nemotron_sft_v3/multilingual_v2/math_ko": 0.415381373,
        "nemotron_sft_v3/multilingual_v2/math_pt": 0.345049712,
        "nemotron_sft_v3/multilingual_v2/stem_hi": 0.063312687,
        "nemotron_sft_v3/multilingual_v2/stem_ja": 0.015738635,
        "nemotron_sft_v3/multilingual_v2/stem_ko": 0.041556524,
        "nemotron_sft_v3/multilingual_v2/stem_pt": 0.281053878,
        "nemotron_sft_v3/opencode_v1/agent_skills": 1.415033814,
        "nemotron_sft_v3/opencode_v1/agent_skills_question_tool": 0.709881881,
        "nemotron_sft_v3/opencode_v1/bash_only_tool": 1.377783671,
        "nemotron_sft_v3/opencode_v1/bash_only_tool_skills": 1.276388833,
        "nemotron_sft_v3/opencode_v1/general": 1.205359132,
        "nemotron_sft_v3/opencode_v1/question_tool": 1.158402087,
        "nemotron_sft_v3/safety_v1/train": 0.04,
        "nemotron_sft_v3/safety_v2/train": 0.106256096,
        "nemotron_sft_v3/science_v2/rqa": 2.858648936,
        "nemotron_sft_v3/science_v2/so": 3.783833,
        "nemotron_sft_v3/science_v2/syn_mcq": 0.130918756,
        "nemotron_sft_v3/science_v2/vendor": 8.223998093,
        "nemotron_sft_v3/swe_v1/r2e_gym": 2.592622304,
        "nemotron_sft_v3/swe_v2/agentless": 1.477754554,
        "nemotron_sft_v3/swe_v2/openhands_swe": 2.338105506,
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
        ("massive_function_calling", massive_chat_normalize_steps),
        ("nemotron-terminal", nemotron_terminal_chat_normalize_steps),
        ("numinamath-1.5", numinamath_v1_5_chat_normalize_steps),
        ("numinamath-tir", numinamath_tir_chat_normalize_steps),
        ("openthoughts4-code-glm-5.2-n4", openthoughts4_code_chat_normalize_steps),
        ("superior-reasoning", superior_reasoning_chat_normalize_steps),
        ("swe-rebench-openhands", swe_rebench_openhands_chat_normalize_steps),
        ("swe-zero-12m", swe_zero_12m_chat_normalize_steps),
        ("synthetic-1", synthetic1_chat_normalize_steps),
        ("synthetic-misconceptions-conversations", synthetic_misconceptions_chat_normalize_steps),
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
    # This chat-only source has 3,341,347,579 completion tokens in its pinned
    # manifest. The rough weight excludes repeated prompts.
    token_counts["openthoughts4-code-glm-5.2-n4"] = 3.341347579
    # Initial sharding estimates; token-store preparation measures the actual mixture sizes.
    token_counts["agenttrove-glm53-compactions"] = 0.25
    token_counts["wildchat-glm53-format-completions"] = 0.01
    token_counts["synthetic-misconceptions-conversations"] = 0.002
    return {
        name: DatakitChatSource(
            name=name,
            chat_steps=factory(),
            rough_token_count_b=token_counts[name],
        )
        for name, factory in rows
    }
