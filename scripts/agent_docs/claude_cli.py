# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Thin subprocess wrapper for the Claude CLI."""

from __future__ import annotations

import logging
import subprocess
import time

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def generate(
    prompt: str,
    *,
    model: str = "sonnet",
    max_budget_usd: float = 0.50,
    timeout_seconds: int = 600,
) -> str:
    """Call claude CLI in non-interactive mode and return the response text.

    Retries up to MAX_RETRIES times on transient failures (empty stderr).
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

    for attempt in range(1, MAX_RETRIES + 2):
        t0 = time.monotonic()
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        elapsed = time.monotonic() - t0

        if result.returncode == 0:
            output = strip_markdown_fences(result.stdout)
            logger.debug(
                "claude %s: %.1fs, %d chars prompt → %d chars output",
                model,
                elapsed,
                len(prompt),
                len(output),
            )
            return output

        # Transient failure (empty stderr) — retry
        if not result.stderr.strip() and attempt <= MAX_RETRIES:
            logger.warning(
                "claude CLI returned rc=%d with empty stderr (attempt %d/%d, %.1fs), retrying...",
                result.returncode,
                attempt,
                MAX_RETRIES + 1,
                elapsed,
            )
            time.sleep(2 * attempt)
            continue

        raise RuntimeError(
            f"claude CLI failed (rc={result.returncode}) after {attempt} attempt(s):\n"
            f"  stderr: {result.stderr[:500]}\n"
            f"  stdout: {result.stdout[:500]}\n"
            f"  prompt length: {len(prompt)} chars\n"
            f"  cmd: {' '.join(cmd)}"
        )

    raise RuntimeError("unreachable")


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
