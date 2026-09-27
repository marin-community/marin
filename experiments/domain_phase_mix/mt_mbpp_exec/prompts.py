# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Signature-disclosing MT-MBPP prompts.

The native MT-MBPP prompt (OLMo 3's BPB task) is three solved examples and the target task, each a one-line task
description followed by a fenced code block. The executable variant adds one line after every task description,
before its code fence, stating the signature of the function the tests call. Removing those lines recovers the native
prompt exactly, which is how the evaluation runner checks the prompts against the native request set.
"""

from collections.abc import Sequence

SIGNATURE_LINE = "Function signature: `{signature}`"
SHOTS = 3


def fence(language: str) -> str:
    return f"\n```{language}\n"


def with_signatures(context: str, language: str, signatures: Sequence[str]) -> str:
    """Insert one signature line before each of the prompt's opening code fences (the examples', then the target's)."""
    parts = context.split(fence(language))
    if len(parts) != SHOTS + 2 or len(signatures) != SHOTS + 1:
        raise ValueError(f"Expected {SHOTS + 1} {language} code fences and signatures")
    if any("`" in s or "\n" in s for s in signatures):
        raise ValueError("Signatures must be single-line and free of backticks")
    prompt = parts[0]
    for signature, part in zip(signatures, parts[1:], strict=True):
        prompt += "\n" + SIGNATURE_LINE.format(signature=signature) + fence(language) + part
    return prompt


def shot_codes(context: str, language: str) -> list[str]:
    """The three example solutions in a native prompt, in order."""
    parts = context.split(fence(language))
    return [part.split("\n```\n", 1)[0] for part in parts[1 : SHOTS + 1]]
