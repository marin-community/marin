# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render OpenAI tool histories without changing their wire representation."""

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


def chat_template_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Decode tool-call argument JSON for chat templates, preserving the original messages."""
    rendered = deepcopy([dict(message) for message in messages])
    for message in rendered:
        for call in message.get("tool_calls") or []:
            function = call["function"]
            if isinstance(function["arguments"], str):
                function["arguments"] = json.loads(function["arguments"])
    return rendered
