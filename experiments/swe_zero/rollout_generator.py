# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Multi-turn rollout generator for SWE-ZERO.

Generates execution-free agentic rollouts by orchestrating multi-turn
tool-calling conversations with a language model (via OpenAI-compatible API,
e.g. vLLM serving Gemma 4).
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
    TOOLS,
    RepoSnapshot,
    build_task_message,
    simulate_tool_response,
)

logger = logging.getLogger(__name__)

MAX_TURNS = 30
MAX_TOKENS_PER_TURN = 4096
MAX_TOTAL_TOKENS = 8192


@dataclass
class RolloutStep:
    """A single step in a rollout trace."""

    role: str  # "assistant" | "tool"
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d


@dataclass
class Rollout:
    """A complete multi-turn rollout trace."""

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
            "steps": [s.to_dict() for s in self.steps],
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "finished": self.finished,
            "error": self.error,
            "duration_sec": self.duration_sec,
        }

    def actions_text(self) -> str:
        """Concatenation of all actions (tool calls) for diversity measurement."""
        parts = []
        for step in self.steps:
            if step.tool_calls:
                for tc in step.tool_calls:
                    fn = tc.get("function", {})
                    parts.append(f"{fn.get('name', '')}({fn.get('arguments', '')})")
        return "\n".join(parts)


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token count estimate (4 chars ≈ 1 token)."""
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

    Orchestrates a multi-turn conversation where the model makes tool calls
    and receives simulated responses, until it calls `finish` or hits limits.
    """
    snapshot = RepoSnapshot.from_pr(pr)
    rollout = Rollout(instance_id=pr.instance_id, repo=pr.repo)
    start_time = time.monotonic()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_task_message(pr)},
    ]

    for turn in range(max_turns):
        # Check token budget
        if _estimate_tokens(messages) > max_total_tokens:
            logger.info("Token budget exceeded at turn %d for %s", turn, pr.instance_id)
            break

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=MAX_TOKENS_PER_TURN,
            )
        except Exception as e:
            rollout.error = f"API error at turn {turn}: {e}"
            logger.error(rollout.error)
            break

        choice = response.choices[0]
        message = choice.message

        # Track token usage
        if response.usage:
            rollout.total_prompt_tokens += response.usage.prompt_tokens
            rollout.total_completion_tokens += response.usage.completion_tokens

        # Record assistant message
        assistant_step = RolloutStep(role="assistant", content=message.content)
        if message.tool_calls:
            assistant_step.tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        rollout.steps.append(assistant_step)

        # Add assistant message to conversation
        msg_dict: dict = {"role": "assistant"}
        if message.content:
            msg_dict["content"] = message.content
        if message.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        messages.append(msg_dict)

        # Process tool calls
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                tool_response = simulate_tool_response(tc.function.name, args, snapshot)

                tool_step = RolloutStep(
                    role="tool",
                    content=tool_response,
                    tool_call_id=tc.id,
                    name=tc.function.name,
                )
                rollout.steps.append(tool_step)

                messages.append(
                    {
                        "role": "tool",
                        "content": tool_response,
                        "tool_call_id": tc.id,
                    }
                )

                # Check if the agent called finish
                if tc.function.name == "finish":
                    rollout.finished = True
                    break

            if rollout.finished:
                break
        else:
            # No tool calls and no finish — model just responded with text.
            # This can happen; continue to next turn.
            if choice.finish_reason == "stop":
                # Model stopped generating without tool calls — treat as done
                rollout.finished = True
                break

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
