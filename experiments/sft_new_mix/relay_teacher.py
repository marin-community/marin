# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve relay source-turn supervision through the existing Terminus renderer."""

from dataclasses import dataclass

from marin.datakit.chat_render import render_marin_chat
from marin.datakit.download.calibforge_relay import THINK_TOKENS
from marin.datakit.download.rollout_transforms import openai_chat_messages
from marin.datakit.download.terminus import TerminusParseMode, terminus_protocol_messages


@dataclass(frozen=True)
class TeacherTranscript:
    text: str
    trained_assistant_turns: tuple[bool, ...]


def render_teacher_transcript(source_messages: list[dict]) -> TeacherTranscript:
    """Return a complete transcript with one training flag per assistant turn.

    Consecutive student and teacher turns retain separate supervision boundaries.
    """
    conversation = []
    flags = []
    for message in source_messages:
        if message["role"] == "system":
            if message["content"].strip() != "Reasoning: /think":
                raise ValueError("Unexpected relay system instruction")
            continue
        if not isinstance(message["trained"], bool):
            raise ValueError("Relay training flags must be explicit booleans")
        conversation.append({"role": message["role"], "content": message["content"]})
        flags.append(message["trained"])
    parsed = terminus_protocol_messages(conversation, THINK_TOKENS, mode=TerminusParseMode.REPAIR)
    if parsed is None:
        raise ValueError("Invalid relay Terminus protocol")
    if len(parsed) != len(conversation):
        raise ValueError("Terminus parser changed source-turn cardinality")
    pieces = []
    assistant_flags = []
    for original, message, trained in zip(conversation, parsed, flags, strict=True):
        if original["role"] != message["role"]:
            raise ValueError("Terminus parser changed source-turn roles")
        pieces.append(render_marin_chat(openai_chat_messages([message]), bos_token=""))
        if message["role"] == "assistant":
            assistant_flags.append(trained)
        elif trained:
            raise ValueError("Relay marks a non-assistant source turn as trained")
    if not any(assistant_flags):
        raise ValueError("Relay episode has no trained teacher turn")
    return TeacherTranscript("".join(pieces), tuple(assistant_flags))
