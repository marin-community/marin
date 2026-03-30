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
            "- Preserve substantive details.\n"
            "- SELF-CONTAINED: The user_message must work as a standalone message. If the scenario "
            "describes an attached document, uploaded file, pasted text, or linked resource, you MUST "
            "inline a brief representative excerpt (2-5 sentences of plausible content) directly in "
            "the user_message instead of referencing an absent attachment. For example, replace "
            "'see the attached PDF' with the actual content the user would paste.\n"
            "- COMPLETE: The user_message must be a complete, natural utterance. End with proper "
            "punctuation. Do not truncate mid-sentence or mid-thought.\n"
            "- NO PLACEHOLDERS: Replace any bracketed tokens like [Name], [redacted], [INSERT] with "
            "realistic, contextually appropriate values.\n"
            "- NO DANGLING REFERENCES: Do not reference URLs, links, prior messages, 'the text above', "
            "or 'see attached' unless the referenced content is fully present in the message itself.\n\n"
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
            "- Preserve substantive details.\n"
            "- SELF-CONTAINED: If the scenario describes an attached document, uploaded file, or linked "
            "resource, inline a brief representative excerpt directly in the user_message instead of "
            "referencing an absent attachment.\n"
            "- COMPLETE: End with proper punctuation. Do not truncate mid-sentence.\n"
            "- NO PLACEHOLDERS: Replace bracketed tokens like [Name] with realistic values.\n"
            "- NO DANGLING REFERENCES: Do not reference absent URLs, links, or attachments.\n\n"
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
