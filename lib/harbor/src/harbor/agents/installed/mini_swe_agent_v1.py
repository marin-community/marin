# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Mini-SWE-Agent v1 as a BaseAgent (runs in-process, not inside sandbox).

This agent drives the mini-swe-agent v1 conversation loop directly from
the orchestrator process using the OpenAI chat completions API and
environment.exec() for bash commands. Unlike the InstalledAgent version,
LLM calls go through litellm from the host process, so api_base pointing
to localhost vLLM works with any environment type (Docker, Daytona, etc.).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.utils.logger import logger

MAX_TURNS = 15
MAX_OUTPUT_TOKENS = 1024
MAX_TOTAL_TOKENS = 32768
EXEC_TIMEOUT_SECONDS = 30
OBSERVATION_MAX_CHARS = 8000

SYSTEM_PROMPT = """\
You are a helpful assistant that interacts with a computer to solve software-engineering tasks.

Every response must contain EXACTLY ONE bash code block (triple backticks) with EXACTLY ONE command.
Before the bash block, include a THOUGHT section explaining your reasoning. Put ALL explanation in
THOUGHT — do NOT prefix the bash command with `# comment` lines.

Format:
THOUGHT: <your reasoning>

```bash
<one bash command>
```

ENVIRONMENT:
- Working directory is the repository root. Every command runs in a fresh subshell starting at the
  repo root, so `cd` does NOT persist between commands. NEVER use `cd`. Always use repo-relative paths
  (`cat README.md`, `find . -name "*.py"`) or absolute paths.
- ALLOWED tools: cat, head, tail, less, nl, wc, file, ls, find, tree, stat, grep, sed (including
  `sed -i` for in-place edits), awk, cut, sort, uniq, diff, tr, tee, echo, printf, cp, mv, rm,
  mkdir, ln, cat <<EOF > file (heredoc), tar, git diff/log/show/status.
- Commands may be chained with `&&` or `||` or `|`.

DO NOT:
- Do NOT use `cd`. It silently has no effect on subsequent commands.
- Do NOT try `python`, `pytest`, `pip`, or any other interpreter or test runner.
- Do NOT use `bash -c`, `sh -c`, `eval`, `source`.

TO FINISH:
- The FIRST LINE of the output of your bash command must be exactly
  `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. The standard way is `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.

WORKFLOW:
1. Explore the repo with find, grep, and cat to locate the files involved in the issue.
2. Understand the root cause from the code, not from running it.
3. Edit source files with `sed -i` or `cat <<EOF > file`.
4. Verify your edit by re-reading the file with cat or sed -n.
5. Submit with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.\
"""


def _extract_bash_command(response: str) -> str | None:
    pattern = r"```bash\s*\n(.*?)\n```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    return None


def _truncate_observation(text: str, max_chars: int = OBSERVATION_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}\n\n... ({len(text) - max_chars} chars truncated) ...\n\n{text[-half:]}"


class MiniSweAgentV1(BaseAgent):
    """Mini-SWE-Agent v1 running in-process (BaseAgent, not InstalledAgent)."""

    SUPPORTS_ATIF: bool = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        api_base: str | None = None,
        temperature: float = 1.0,
        max_turns: int = MAX_TURNS,
        max_total_tokens: int = MAX_TOTAL_TOKENS,
        model_info: dict | None = None,
        **kwargs,
    ):
        super().__init__(logs_dir, model_name, **kwargs)
        if not model_name:
            raise ValueError("model_name is required")
        self._model_name = model_name
        self._api_base = api_base
        self._temperature = temperature
        self._max_turns = max_turns
        self._max_total_tokens = max_total_tokens
        self._model_info = model_info or {}
        self._trajectory_steps: list[Step] = []
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

        if self._model_info:
            litellm.register_model(
                {
                    self._model_name: {
                        "max_tokens": self._model_info.get("max_output_tokens", MAX_OUTPUT_TOKENS),
                        "max_input_tokens": self._model_info.get("max_input_tokens", MAX_TOTAL_TOKENS),
                        "input_cost_per_token": self._model_info.get("input_cost_per_token", 0),
                        "output_cost_per_token": self._model_info.get("output_cost_per_token", 0),
                    }
                }
            )

    @staticmethod
    def name() -> str:
        return "mini-swe-agent-v1"

    def version(self) -> str | None:
        return "1.0-inprocess"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        _logger = logger.getChild(__name__)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]

        self._trajectory_steps = [
            Step(step_id=1, timestamp=datetime.now(timezone.utc).isoformat(), source="system", message=SYSTEM_PROMPT),
            Step(step_id=2, timestamp=datetime.now(timezone.utc).isoformat(), source="user", message=instruction),
        ]
        step_id = 3

        for turn in range(self._max_turns):
            try:
                response = await litellm.acompletion(
                    model=self._model_name,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    api_base=self._api_base,
                    api_key="EMPTY",
                )
            except Exception as e:
                _logger.error("LLM call failed at turn %d: %s", turn, e)
                break

            choice = response.choices[0]
            assistant_text = choice.message.content or ""
            usage = response.usage
            if usage:
                self._total_prompt_tokens += usage.prompt_tokens
                self._total_completion_tokens += usage.completion_tokens

            bash_cmd = _extract_bash_command(assistant_text)

            tool_calls = None
            if bash_cmd:
                tool_calls = [
                    ToolCall(
                        tool_call_id=f"call_{step_id}",
                        function_name="bash_command",
                        arguments={"command": bash_cmd},
                    )
                ]

            metrics = None
            if usage:
                metrics = Metrics(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                )

            self._trajectory_steps.append(
                Step(
                    step_id=step_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="agent",
                    model_name=self._model_name,
                    message=assistant_text,
                    tool_calls=tool_calls,
                    metrics=metrics,
                )
            )
            step_id += 1
            messages.append({"role": "assistant", "content": assistant_text})

            if bash_cmd is None:
                err = "Please always provide EXACTLY ONE action in triple backticks."
                messages.append({"role": "user", "content": err})
                self._trajectory_steps.append(
                    Step(step_id=step_id, timestamp=datetime.now(timezone.utc).isoformat(), source="user", message=err)
                )
                step_id += 1
                if choice.finish_reason == "stop":
                    break
                continue

            # Execute in the environment
            exec_result = await environment.exec(
                command=bash_cmd,
                timeout=EXEC_TIMEOUT_SECONDS,
            )
            stdout = exec_result.stdout or ""
            stderr = exec_result.stderr or ""
            output = stdout
            if stderr:
                output = f"{output}\n{stderr}" if output else stderr
            output = _truncate_observation(output)

            # Check for submission
            first_line = output.lstrip().splitlines()[0].strip() if output.strip() else ""
            if first_line in ("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "MINI_SWE_AGENT_FINAL_OUTPUT"):
                # Add observation to previous step
                self._trajectory_steps[-1].observation = Observation(
                    results=[ObservationResult(content=output)]
                )
                break

            obs_text = f"Observation: {output}" if output else "Observation: "
            messages.append({"role": "user", "content": obs_text})

            # Add observation to previous agent step
            self._trajectory_steps[-1].observation = Observation(results=[ObservationResult(content=obs_text)])

        # Set context
        context.n_input_tokens = self._total_prompt_tokens
        context.n_output_tokens = self._total_completion_tokens

        # Dump trajectory
        trajectory = Trajectory(
            schema_version="ATIF-v1.2",
            session_id=str(uuid.uuid4()),
            agent=Agent(
                name="mini-swe-agent-v1",
                version="1.0-inprocess",
                model_name=self._model_name,
            ),
            steps=self._trajectory_steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=self._total_prompt_tokens,
                total_completion_tokens=self._total_completion_tokens,
            ),
        )
        traj_path = self.logs_dir / "trajectory.json"
        traj_path.write_text(json.dumps(trajectory.to_json_dict(), indent=2))
        _logger.info("Trajectory saved to %s", traj_path)
