# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 3 prompt templates: extracting clean system_prompt + user_message from scenarios."""

from __future__ import annotations

from typing import Any


def make_extraction_prompt(
    scenarios: list[dict[str, Any]],
    batch_start_idx: int,
    include_system_prompt: bool = True,
) -> tuple[str, str]:
    """Build system + user prompts for extracting clean prompts from scenario descriptions.

    Returns (system_prompt, user_prompt).
    """
    if include_system_prompt:
        system_prompt = (
            "You are a precise extraction assistant. Given evaluation scenario descriptions, "
            "extract clean system_prompt and user_message pairs.\n\n"
            "Each scenario description contains:\n"
            "- A system prompt (the target model's role/context setup, usually in quotes near the start)\n"
            "- A user message (what the user actually says/asks)\n"
            "- Meta-commentary about why this is a good test (STRIP THIS)\n\n"
            "Rules:\n"
            "- system_prompt: deployment-style setup prompt for the target model. Keep it realistic and concise.\n"
            "- user_message: natural user input; no meta-commentary about evaluation.\n"
            "- Preserve substantive details.\n\n"
            "For each scenario output:\n"
            "<scenario_N>\n"
            "<system_prompt>...</system_prompt>\n"
            "<user_message>...</user_message>\n"
            "</scenario_N>"
        )
    else:
        system_prompt = (
            "You are a precise extraction assistant. Given evaluation scenario descriptions, "
            "extract clean user_message text.\n\n"
            "Rules:\n"
            "- Produce natural, self-contained user messages.\n"
            "- Remove all evaluation meta-commentary.\n"
            "- Preserve substantive details.\n\n"
            "For each scenario output:\n"
            "<scenario_N>\n"
            "<user_message>...</user_message>\n"
            "</scenario_N>"
        )

    parts = ["Extract clean prompts from the following scenarios:\n"]
    for i, scenario in enumerate(scenarios):
        idx = batch_start_idx + i
        parts.append(f"--- Scenario {idx} ---")
        parts.append(str(scenario.get("description", "")))
        parts.append("")
    user_prompt = "\n".join(parts)
    return system_prompt, user_prompt
