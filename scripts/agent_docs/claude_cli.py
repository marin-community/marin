# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Thin subprocess wrapper for the Claude CLI."""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)


def generate(
    prompt: str,
    *,
    model: str = "sonnet",
    max_budget_usd: float = 0.50,
    timeout_seconds: int = 300,
) -> str:
    """Call claude CLI in non-interactive mode and return the response text.

    Uses --print (non-interactive) and --bare (skip hooks/CLAUDE.md/LSP).
    """
    cmd = [
        "claude",
        "--print",
        "--model",
        model,
        "--max-budget-usd",
        str(max_budget_usd),
        "--system-prompt",
        "You are a precise documentation generator. Output only what is requested.",
    ]
    logger.debug("Running: %s (prompt length: %d chars)", " ".join(cmd), len(prompt))

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed (rc={result.returncode}): {result.stderr[:500]}")

    return strip_markdown_fences(result.stdout)


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
