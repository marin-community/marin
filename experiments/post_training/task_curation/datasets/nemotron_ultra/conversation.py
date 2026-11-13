# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read the reply a Nemotron Ultra grade script scores.

Shipped as ``/tests/conversation.py`` beside the grade scripts that score the chat conversation the
grader writes to ``/tests/conversation.json``; they import it after putting ``/tests`` on their path.
"""

import json
from pathlib import Path

from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text


def terminal_message() -> dict:
    """The final assistant message, its text reduced to the final answer."""
    message = json.loads(Path("/tests/conversation.json").read_text())[-1]
    if message["role"] != "assistant":
        raise ValueError(f"The conversation ends with a {message['role']} message, not a reply")
    return {**message, "content": final_answer_text(message.get("content") or "")}
