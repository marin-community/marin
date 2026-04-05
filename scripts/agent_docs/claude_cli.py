# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Thin subprocess wrapper for the Claude CLI."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def _run_claude(
    prompt: str,
    *,
    model: str,
    system_prompt: str,
    max_budget_usd: float,
    timeout_seconds: int,
    output_format: str = "text",
) -> subprocess.CompletedProcess[str]:
    """Run the claude CLI with retries on transient failures.

    Returns the successful CompletedProcess, or raises on persistent failure.
    """
    cmd = [
        "claude",
        "--print",
        "--model",
        model,
        "--max-budget-usd",
        str(max_budget_usd),
        "--output-format",
        output_format,
        "--system-prompt",
        system_prompt,
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
            logger.debug(
                "claude %s: %.1fs, %d chars prompt → %d chars output",
                model,
                elapsed,
                len(prompt),
                len(result.stdout),
            )
            return result

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


def generate(
    prompt: str,
    *,
    model: str = "sonnet",
    system_prompt: str = "You are a precise documentation generator. Output only what is requested.",
    max_budget_usd: float = 0.50,
    timeout_seconds: int = 600,
) -> str:
    """Call claude CLI in non-interactive mode and return the response text."""
    result = _run_claude(
        prompt,
        model=model,
        system_prompt=system_prompt,
        max_budget_usd=max_budget_usd,
        timeout_seconds=timeout_seconds,
        output_format="text",
    )
    return strip_markdown_fences(result.stdout)


def generate_json(
    prompt: str,
    *,
    model: str = "sonnet",
    system_prompt: str = "You are a precise documentation generator. Output only what is requested.",
    max_budget_usd: float = 0.50,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Call claude CLI with --output-format json and return the parsed response.

    The returned dict has at minimum a "result" key with the model's text,
    plus "total_cost_usd" and "usage" metadata from the CLI.
    """
    result = _run_claude(
        prompt,
        model=model,
        system_prompt=system_prompt,
        max_budget_usd=max_budget_usd,
        timeout_seconds=timeout_seconds,
        output_format="json",
    )
    return json.loads(result.stdout)


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
