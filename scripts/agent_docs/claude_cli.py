# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Thin subprocess wrapper for the Claude CLI."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

# Env keys stripped before launching the CLI. Removing the API key makes claude
# authenticate via OAuth (the flat-rate subscription plan) instead of metered
# per-token API billing; dropping CLAUDE_CODE_* starts a clean session rather
# than inheriting the calling agent's. Without this, nested calls would bill
# against whatever the parent process has set — potentially the expensive API.
_STRIPPED_ENV_KEYS = frozenset({"ANTHROPIC_API_KEY"})
_STRIPPED_ENV_PREFIXES = ("CLAUDE_CODE_",)


def _subscription_env() -> dict[str, str]:
    """Return ``os.environ`` minus the API token and CLAUDE_CODE_* session vars.

    The generator and eval call claude in a tight loop, so they must run on the
    subscription plan, not the API. Stripping ANTHROPIC_API_KEY forces OAuth;
    the CLI then authenticates from ``~/.claude`` credentials.
    """
    return {
        k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV_KEYS and not k.startswith(_STRIPPED_ENV_PREFIXES)
    }


def _run_claude(
    prompt: str,
    *,
    model: str,
    system_prompt: str,
    max_budget_usd: float,
    timeout_seconds: int,
    output_format: str = "text",
    disable_tools: bool = False,
    allowed_tools: list[str] | None = None,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the claude CLI with retries on transient failures.

    Returns the successful CompletedProcess, or raises on persistent failure.

    Tool access (mutually exclusive, in priority order):
    - ``disable_tools=True``: empty allowed-tools set — no bash, file reads, or
      grep. The eval coder uses this so its output is attributable to the docs.
    - ``allowed_tools=[...]``: only these tools (e.g. ``["Read", "Grep",
      "Glob"]``) — used by the agentic generator to explore real source read-only.
    - neither: the CLI default toolset.

    ``cwd`` sets the working directory, so a tool-using agent reads the repo.
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
    if disable_tools:
        # Empty allowed-tools value -> the model gets no tools at all.
        cmd += ["--allowed-tools", ""]
    elif allowed_tools is not None:
        cmd += ["--allowed-tools", " ".join(allowed_tools)]

    for attempt in range(1, MAX_RETRIES + 2):
        t0 = time.monotonic()
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=_subscription_env(),
            cwd=cwd,
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
    disable_tools: bool = False,
) -> dict[str, Any]:
    """Call claude CLI with --output-format json and return the parsed response.

    The returned dict has at minimum a "result" key with the model's text,
    plus "total_cost_usd" and "usage" metadata from the CLI.

    Pass ``disable_tools=True`` to run the model with no tools (no bash, file
    reads, or grep) — used by the eval coder so its output depends only on the
    documentation in the prompt.
    """
    result = _run_claude(
        prompt,
        model=model,
        system_prompt=system_prompt,
        max_budget_usd=max_budget_usd,
        timeout_seconds=timeout_seconds,
        output_format="json",
        disable_tools=disable_tools,
    )
    return json.loads(result.stdout)


_AGENTIC_SYSTEM_PROMPT = (
    "You are a precise documentation generator. Explore the source as needed, "
    "then output only the requested document."
)


def run_agent_json(
    prompt: str,
    *,
    model: str,
    allowed_tools: list[str],
    cwd: str,
    system_prompt: str,
    max_budget_usd: float,
    timeout_seconds: int = 1200,
) -> dict[str, Any]:
    """Run claude as a tool-using agent in ``cwd``; return the parsed JSON envelope.

    The agent gets only ``allowed_tools`` (e.g. read-only Read/Grep/Glob) and is
    bounded by ``max_budget_usd`` — the CLI halts it after a turn once the spend
    would exceed, which is how we cap exploration (there is no --max-turns flag).
    The envelope has at least ``result`` (final text), ``usage``, and
    ``total_cost_usd``.
    """
    result = _run_claude(
        prompt,
        model=model,
        system_prompt=system_prompt,
        max_budget_usd=max_budget_usd,
        timeout_seconds=timeout_seconds,
        output_format="json",
        allowed_tools=allowed_tools,
        cwd=cwd,
    )
    return json.loads(result.stdout)


def generate_with_tools(
    prompt: str,
    *,
    model: str,
    allowed_tools: list[str],
    cwd: str,
    system_prompt: str = _AGENTIC_SYSTEM_PROMPT,
    max_budget_usd: float = 2.0,
    timeout_seconds: int = 1200,
) -> tuple[str, float]:
    """Run claude with a restricted read-only toolset in ``cwd``.

    The agentic doc generator uses this: the agent explores real source with
    ``allowed_tools`` (e.g. Read/Grep/Glob) instead of being handed a digest.
    Returns ``(doc_text, cost_usd)``; cost is the CLI's API-equivalent figure,
    used to enforce the experiment's spend cap.
    """
    data = run_agent_json(
        prompt,
        model=model,
        allowed_tools=allowed_tools,
        cwd=cwd,
        system_prompt=system_prompt,
        max_budget_usd=max_budget_usd,
        timeout_seconds=timeout_seconds,
    )
    return strip_markdown_fences(data.get("result", "")), float(data.get("total_cost_usd", 0.0))


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
