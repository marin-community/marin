# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Multi-turn rollout generator using mini-swe-agent v2 format.

Generates execution-free agentic rollouts by orchestrating multi-turn
conversations where the model outputs bash commands and receives simulated
outputs, matching mini-swe-agent v2's THOUGHT + ```mswea_bash_command format.
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
    RepoSnapshot,
    build_task_message,
    extract_bash_command,
    simulate_bash,
)

logger = logging.getLogger(__name__)

MAX_TURNS = 30
MAX_TOKENS_PER_TURN = 4096
MAX_TOTAL_TOKENS = 8192


@dataclass
class RolloutStep:
    """A single step in a rollout trace (mini-swe-agent v2 format)."""

    role: str  # "system" | "user" | "assistant" | "observation"
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
    """A complete multi-turn rollout trace in mini-swe-agent v2 format."""

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
            "trajectory_format": "mini-swe-agent-1.1",
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
        parts = []
        for step in self.steps:
            if step.bash_command:
                parts.append(step.bash_command)
        return "\n".join(parts)


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token count estimate (4 chars ~ 1 token)."""
    text = json.dumps(messages)
    return len(text) // 4


def generate_rollout(
    client: OpenAI,
    model: str,
    pr: PRRecord,
    temperature: float = 1.0,
    max_turns: int = MAX_TURNS,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
) -> Rollout:
    """
    Generate a single execution-free rollout for a PR.

    Uses mini-swe-agent v2 format: model outputs THOUGHT + bash command,
    environment returns simulated bash output.
    """
    snapshot = RepoSnapshot.from_pr(pr)
    rollout = Rollout(instance_id=pr.instance_id, repo=pr.repo)
    start_time = time.monotonic()

    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    user_msg = {"role": "user", "content": build_task_message(pr)}

    rollout.steps.append(RolloutStep(role="system", content=SYSTEM_PROMPT))
    rollout.steps.append(RolloutStep(role="user", content=build_task_message(pr)))

    messages: list[dict] = [system_msg, user_msg]

    for turn in range(max_turns):
        if _estimate_tokens(messages) > max_total_tokens:
            logger.info("Token budget exceeded at turn %d for %s", turn, pr.instance_id)
            break

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=MAX_TOKENS_PER_TURN,
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

        # Extract bash command from response
        bash_cmd = extract_bash_command(assistant_text)

        rollout.steps.append(RolloutStep(role="assistant", content=assistant_text, bash_command=bash_cmd))
        messages.append({"role": "assistant", "content": assistant_text})

        if bash_cmd is None:
            # Model didn't produce a bash block — v1 would raise FormatError
            # and prompt for a single action. We send a format error message.
            err = "Please always provide EXACTLY ONE action in triple backticks."
            rollout.steps.append(RolloutStep(role="user", content=err))
            messages.append({"role": "user", "content": err})
            if choice.finish_reason == "stop":
                # Likely the model is done responding entirely
                break
            continue

        # Simulate the bash command
        observation = simulate_bash(bash_cmd, snapshot)

        # v1 submission detection: first line of the bash output must be
        # COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT (or MINI_SWE_AGENT_FINAL_OUTPUT)
        first_line = observation.lstrip().splitlines()[0].strip() if observation.strip() else ""
        if first_line in ("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "MINI_SWE_AGENT_FINAL_OUTPUT"):
            rollout.finished = True
            break

        # v1 truncates long outputs in the timeout_template; do similar here.
        if len(observation) > 10000:
            observation = observation[:5000] + "\n\n... (output truncated) ...\n\n" + observation[-2000:]

        # v1 observation format: "Observation: {{output}}"
        obs_text = f"Observation: {observation}" if observation else "Observation: "
        rollout.steps.append(RolloutStep(role="user", content=obs_text))
        messages.append({"role": "user", "content": obs_text})

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
