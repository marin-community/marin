# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Multi-turn rollout generator using mini-swe-agent v1 format.

For each rollout we materialize a fresh per-PR worktree (a real checkout of
the repo at base_commit with test_patch applied), then drive a multi-turn
conversation with the LM where each assistant turn produces one bash command
and we run it against the worktree via the SWE-ZERO sandbox.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from openai import OpenAI

from experiments.swe_zero.data_loader import PRRecord
from experiments.swe_zero.scaffold import (
    SYSTEM_PROMPT,
    build_task_message,
    execute_in_worktree,
    extract_bash_command,
)
from experiments.swe_zero.worktree import materialize_worktree

logger = logging.getLogger(__name__)

MAX_TURNS = 30
MAX_OUTPUT_TOKENS = 1024
"""Hard cap on completion tokens per turn. Kept well below max_total_tokens
so input + output fit in the model's context window even after many turns
have grown the conversation."""
MAX_TOTAL_TOKENS = 8192
RESERVE_TOKENS = 256
"""Headroom subtracted from max_total_tokens when computing the per-turn
``max_tokens`` ceiling. Avoids 400 errors when input is close to the model's
context length."""
EXEC_TIMEOUT_SECONDS = 30.0
OBSERVATION_MAX_CHARS = 8000


@dataclass
class RolloutStep:
    """A single step in a rollout trace (mini-swe-agent v1 format)."""

    role: str  # "system" | "user" | "assistant"
    content: str = ""
    bash_command: str | None = None
    extra: dict | None = None

    def to_dict(self) -> dict:
        d: dict = {"role": self.role, "content": self.content}
        if self.bash_command is not None:
            d["bash_command"] = self.bash_command
        if self.extra is not None:
            d["extra"] = self.extra
        return d


@dataclass
class Rollout:
    """A complete multi-turn rollout trace in mini-swe-agent v1 format."""

    instance_id: str
    repo: str
    steps: list[RolloutStep] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    finished: bool = False
    error: str | None = None
    duration_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "trajectory_format": "mini-swe-agent-1",
            "messages": [s.to_dict() for s in self.steps],
            "info": {
                "exit_status": "Submitted" if self.finished else self.error or "incomplete",
                "model_stats": {
                    "instance_cost": 0.0,
                    "prompt_tokens": self.total_prompt_tokens,
                    "completion_tokens": self.total_completion_tokens,
                },
            },
            "duration_sec": self.duration_sec,
        }

    def actions_text(self) -> str:
        """Concatenation of all bash commands for diversity measurement."""
        return "\n".join(step.bash_command for step in self.steps if step.bash_command)


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token count estimate (4 chars ≈ 1 token)."""
    return len(json.dumps(messages)) // 4


def generate_rollout(
    client: OpenAI,
    model: str,
    pr: PRRecord,
    temperature: float = 1.0,
    max_turns: int = MAX_TURNS,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
) -> Rollout:
    """Generate a single execution-free rollout for a PR.

    Materializes a per-rollout worktree (real checkout at base_commit + test
    patch), drives a multi-turn conversation, and runs each bash command in
    the SWE-ZERO sandbox. Always cleans up the worktree on exit.
    """
    rollout = Rollout(instance_id=pr.instance_id, repo=pr.repo)
    start_time = time.monotonic()

    try:
        worktree = materialize_worktree(pr)
    except Exception as e:
        rollout.error = f"Failed to materialize worktree: {e}"
        rollout.duration_sec = time.monotonic() - start_time
        logger.error(rollout.error)
        return rollout

    try:
        system_msg = {"role": "system", "content": SYSTEM_PROMPT}
        user_msg = {"role": "user", "content": build_task_message(pr)}
        rollout.steps.append(RolloutStep(role="system", content=SYSTEM_PROMPT))
        rollout.steps.append(RolloutStep(role="user", content=user_msg["content"]))
        messages: list[dict] = [system_msg, user_msg]

        for turn in range(max_turns):
            input_tokens_estimate = _estimate_tokens(messages)
            budget_remaining = max_total_tokens - input_tokens_estimate - RESERVE_TOKENS
            if budget_remaining <= 64:
                logger.info(
                    "Token budget exhausted at turn %d for %s (input~%d)",
                    turn,
                    pr.instance_id,
                    input_tokens_estimate,
                )
                break
            max_tokens_this_turn = min(MAX_OUTPUT_TOKENS, budget_remaining)

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens_this_turn,
                )
            except Exception as e:
                rollout.error = f"API error at turn {turn}: {e}"
                logger.error(rollout.error)
                break

            choice = response.choices[0]
            assistant_text = choice.message.content or ""
            if response.usage:
                rollout.total_prompt_tokens += response.usage.prompt_tokens
                rollout.total_completion_tokens += response.usage.completion_tokens

            bash_cmd = extract_bash_command(assistant_text)
            rollout.steps.append(RolloutStep(role="assistant", content=assistant_text, bash_command=bash_cmd))
            messages.append({"role": "assistant", "content": assistant_text})

            if bash_cmd is None:
                err = "Please always provide EXACTLY ONE action in triple backticks."
                rollout.steps.append(RolloutStep(role="user", content=err))
                messages.append({"role": "user", "content": err})
                if choice.finish_reason == "stop":
                    break
                continue

            exec_result = execute_in_worktree(bash_cmd, worktree, timeout_seconds=EXEC_TIMEOUT_SECONDS)
            observation = exec_result.as_observation(max_chars=OBSERVATION_MAX_CHARS)

            # v1 submission detection: first non-empty line of bash output must be
            # COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT (or MINI_SWE_AGENT_FINAL_OUTPUT).
            first_line = observation.lstrip().splitlines()[0].strip() if observation.strip() else ""
            if first_line in ("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "MINI_SWE_AGENT_FINAL_OUTPUT"):
                rollout.finished = True
                break

            obs_text = f"Observation: {observation}" if observation else "Observation: "
            rollout.steps.append(RolloutStep(role="user", content=obs_text))
            messages.append({"role": "user", "content": obs_text})
    finally:
        worktree.cleanup()
        rollout.duration_sec = time.monotonic() - start_time

    return rollout


def generate_rollouts_for_pr(
    client: OpenAI,
    model: str,
    pr: PRRecord,
    n_rollouts: int = 10,
    temperature: float = 1.0,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
) -> list[Rollout]:
    """Generate multiple rollouts for a single PR."""
    rollouts = []
    for i in range(n_rollouts):
        logger.info("Generating rollout %d/%d for %s", i + 1, n_rollouts, pr.instance_id)
        rollout = generate_rollout(
            client=client,
            model=model,
            pr=pr,
            temperature=temperature,
            max_total_tokens=max_total_tokens,
        )
        rollouts.append(rollout)
        logger.info(
            "Rollout %d: %d steps, finished=%s, %.1fs",
            i + 1,
            len(rollout.steps),
            rollout.finished,
            rollout.duration_sec,
        )
    return rollouts
