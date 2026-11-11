# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model wire protocols required by installed harnesses."""

from enum import StrEnum


class ModelAPI(StrEnum):
    OPENAI_CHAT = "openai-chat-completions"
    OPENAI_RESPONSES = "openai-responses"
    ANTHROPIC_MESSAGES = "anthropic-messages"
