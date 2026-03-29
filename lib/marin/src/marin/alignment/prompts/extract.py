# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 3 prompt templates: extracting clean system_prompt + user_message from scenarios."""

from __future__ import annotations

from typing import Any


def make_extraction_prompt(
    scenario: dict[str, Any],
    include_system_prompt: bool = True,
) -> tuple[str, str]:
    """Build system + user prompts for extracting clean prompts from scenario descriptions.

    Returns (system_prompt, user_prompt).
    """
    if include_system_prompt:
        system_prompt = (
            "You are a precise extraction assistant. Given one evaluation scenario description, "
            "extract one clean system_prompt and one clean user_message.\n\n"
            "The scenario description contains:\n"
            "- A system prompt (the target model's role/context setup, usually in quotes near the start)\n"
            "- A user message (what the user actually says/asks)\n"
            "- Meta-commentary about why this is a good test (STRIP THIS)\n\n"
            "Rules:\n"
            "- system_prompt: deployment-style setup prompt for the target model. Keep it realistic and concise.\n"
            "- user_message: natural user input; no meta-commentary about evaluation.\n"
            "- Preserve substantive details.\n\n"
            "Output exactly:\n"
            "<system_prompt>...</system_prompt>\n"
            "<user_message>...</user_message>"
        )
    else:
        system_prompt = (
            "You are a precise extraction assistant. Given one evaluation scenario description, "
            "extract one clean user_message.\n\n"
            "Rules:\n"
            "- Produce natural, self-contained user messages.\n"
            "- Remove all evaluation meta-commentary.\n"
            "- Preserve substantive details.\n\n"
            "Output exactly:\n"
            "<user_message>...</user_message>"
        )

    user_prompt = "\n".join(
        [
            "Extract clean prompts from the following scenario:\n",
            str(scenario.get("description", "")),
            "",
        ]
    )
    return system_prompt, user_prompt
