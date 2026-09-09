# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical registry of Datakit sources that retain structured conversations."""

from collections.abc import Callable
from functools import cache

from marin.datakit.chat import DatakitChatSource
from marin.datakit.download.agenttrove import agenttrove_chat_normalize_steps
from marin.datakit.download.coderforge import coderforge_chat_normalize_steps
from marin.datakit.download.davinci_dev import davinci_dev_env_native_chat_normalize_steps
from marin.datakit.download.glm_kernelgym_rollouts import glm_kernelgym_rollouts_chat_normalize_steps
from marin.datakit.download.gpt_oss_rollouts import gpt_oss_rollouts_chat_normalize_steps
from marin.datakit.download.massive import massive_chat_normalize_steps
from marin.datakit.download.nemotron_terminal import nemotron_terminal_chat_normalize_steps
from marin.datakit.download.nemotron_v2 import nemotron_sft_chat_normalize_steps
from marin.datakit.download.numinamath_tir import numinamath_tir_chat_normalize_steps
from marin.datakit.download.numinamath_v1_5 import numinamath_v1_5_chat_normalize_steps
from marin.datakit.download.openthoughts4_code import openthoughts4_code_chat_normalize_steps
from marin.datakit.download.penfever_rollouts import penfever_rollouts_chat_normalize_steps
from marin.datakit.download.superior_reasoning import superior_reasoning_chat_normalize_steps
from marin.datakit.download.swe_rebench_openhands import swe_rebench_openhands_chat_normalize_steps
from marin.datakit.download.swe_zero_12m import swe_zero_12m_chat_normalize_steps
from marin.datakit.download.synthetic1 import synthetic1_chat_normalize_steps
from marin.datakit.sources import all_sources
from marin.execution.step_spec import StepSpec

_EXCLUDED_CHAT_SOURCES = frozenset(
    {
        # These OpenCode traces omit the user request from every conversation, so
        # they cannot form valid training examples without inventing prompt text.
        "penfever-traces/qwen35-122b-131k-opencode/nemotron-gym-agent-workplace-v2",
    }
)
_ChatSourceRow = tuple[str, Callable[[], tuple[StepSpec, ...]]]


@cache
def all_sft_sources() -> dict[str, DatakitChatSource]:
    """Return sources whose canonical artifact contains Harmony messages."""
    penfever_steps = cache(penfever_rollouts_chat_normalize_steps)
    nemotron_steps = cache(nemotron_sft_chat_normalize_steps)
    rows: list[_ChatSourceRow] = [
        ("agenttrove", agenttrove_chat_normalize_steps),
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

    text_sources = all_sources()
    return {
        name: DatakitChatSource(
            name=name,
            normalize_steps=factory(),
            rough_token_count_b=text_sources[name].rough_token_count_b,
        )
        for name, factory in rows
    }
