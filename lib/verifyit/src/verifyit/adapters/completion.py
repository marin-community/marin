# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reasoning and end-of-turn markers in completion text, and reasoning-trace removal."""


REASONING_END_MARKERS = ("</think>", "<|end_think|>")
END_OF_TURN = (
    "<|im_end|>",
    "<|eot_id|>",
    "<|end_of_text|>",
    "<|endoftext|>",
    "</s>",
    "\nYou are an AI assistant",
    "\nQuestion:",
    "\nQ:",
    "\n[Question]",
    "\nUser:",
    "\nuser\n",
    "\nAssistant:",
)


def final_text(text: str) -> str:
    """Remove a preceding inline reasoning trace using the last end marker."""
    boundary = max(
        (index + len(marker) for marker in REASONING_END_MARKERS if (index := text.rfind(marker)) >= 0),
        default=0,
    )
    return text[boundary:]
