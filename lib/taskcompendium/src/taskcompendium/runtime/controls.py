# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scripted responses and expected rewards for opt-in runtime checks."""

from dataclasses import dataclass

from taskcompendium.models import AssistantMessage, AssistantToolCalls, ConversationToolCall


@dataclass(frozen=True)
class Control:
    name: str
    responses: tuple[AssistantMessage, ...]
    expected_reward: float


def tool_turn(name: str, arguments: dict) -> AssistantToolCalls:
    return AssistantToolCalls(calls=(ConversationToolCall(call_id=f"control-{name}", name=name, arguments=arguments),))
