# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Mini-SWE-Agent v1 as a BaseAgent (in-process, not installed in sandbox).

Directly reuses the official mini-swe-agent v1 DefaultAgent class
(SWE-agent/mini-SWE-agent@v1) with Harbor-native adapters for the Model
and Environment interfaces. LLM calls go through litellm on the host
(reaching localhost vLLM), bash commands execute via Harbor's
environment.exec() (runs in the sandbox).

This avoids the networking issue with InstalledAgent + Daytona where the
mini CLI inside the sandbox can't reach the host's vLLM server.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
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

# ---------------------------------------------------------------------------
# Import mini-swe-agent v1's DefaultAgent and config
# ---------------------------------------------------------------------------
try:
    from minisweagent import Environment as MsweaEnvironment
    from minisweagent import Model as MsweaModel
    from minisweagent.agents.default import AgentConfig, DefaultAgent, Submitted, TerminatingException

    _HAS_MINISWEAGENT = True
except ImportError:
    _HAS_MINISWEAGENT = False


# ---------------------------------------------------------------------------
# Adapters: bridge mini-swe-agent's Model/Environment to Harbor's
# ---------------------------------------------------------------------------

if _HAS_MINISWEAGENT:

    class LitellmModelAdapter(MsweaModel):
        """Adapter: mini-swe-agent Model → litellm (host-side, reaches localhost vLLM)."""

        def __init__(self, model_name: str, api_base: str | None = None, temperature: float = 1.0, **kwargs):
            self.model_name = model_name
            self.api_base = api_base
            self.temperature = temperature
            self.model_kwargs = kwargs.get("model_kwargs", {})
            self._cost = 0.0
            self._n_calls = 0

        @property
        def cost(self) -> float:
            return self._cost

        @property
        def n_calls(self) -> int:
            return self._n_calls

        def get_template_vars(self) -> dict:
            return {}

        def query(self, messages: list[dict]) -> dict:
            """Synchronous query — mini-swe-agent v1 uses sync API."""
            cleaned = [{"role": m["role"], "content": m["content"]} for m in messages]
            response = litellm.completion(
                model=self.model_name,
                messages=cleaned,
                temperature=self.temperature,
                api_base=self.api_base,
                api_key="EMPTY",
                **self.model_kwargs,
            )
            self._n_calls += 1
            usage = response.usage
            if usage:
                cost = litellm.completion_cost(response) if hasattr(litellm, "completion_cost") else 0.0
                self._cost += cost
            content = response.choices[0].message.content or ""
            result: dict[str, Any] = {"content": content}
            if usage:
                result["extra"] = {
                    "response": {
                        "usage": {
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                        }
                    }
                }
            return result

    class HarborEnvironmentAdapter(MsweaEnvironment):
        """Adapter: mini-swe-agent Environment → Harbor's environment.exec()."""

        def __init__(self, harbor_env: BaseEnvironment, loop: asyncio.AbstractEventLoop):
            self._env = harbor_env
            self._loop = loop

        def get_template_vars(self) -> dict:
            import platform

            return {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
            }

        def execute(self, command: str, timeout: int = 30) -> dict:
            """Execute command in Harbor sandbox, return mini-swe-agent format."""
            future = asyncio.run_coroutine_threadsafe(self._env.exec(command=command, timeout=timeout), self._loop)
            result = future.result(timeout=timeout + 10)
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            output = stdout
            if stderr:
                output = f"{output}\n{stderr}" if output else stderr
            return {
                "output": output,
                "returncode": result.exit_code if hasattr(result, "exit_code") else 0,
            }


# ---------------------------------------------------------------------------
# Harbor BaseAgent wrapper
# ---------------------------------------------------------------------------


class MiniSweAgentV1(BaseAgent):
    """Mini-SWE-Agent v1 using the official DefaultAgent with Harbor adapters."""

    SUPPORTS_ATIF: bool = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        api_base: str | None = None,
        temperature: float = 1.0,
        max_turns: int = 15,
        model_info: dict | None = None,
        **kwargs,
    ):
        super().__init__(logs_dir, model_name, **kwargs)
        if not model_name:
            raise ValueError("model_name is required")
        if not _HAS_MINISWEAGENT:
            raise ImportError(
                "minisweagent not installed. Install with: "
                'uv pip install "git+https://github.com/SWE-agent/mini-SWE-agent.git@v1"'
            )
        self._model_name = model_name
        self._api_base = api_base
        self._temperature = temperature
        self._max_turns = max_turns
        self._model_info = model_info or {}

        if self._model_info:
            litellm.register_model(
                {
                    self._model_name: {
                        "max_tokens": self._model_info.get("max_output_tokens", 1024),
                        "max_input_tokens": self._model_info.get("max_input_tokens", 32768),
                        "input_cost_per_token": self._model_info.get("input_cost_per_token", 0),
                        "output_cost_per_token": self._model_info.get("output_cost_per_token", 0),
                    }
                }
            )

    @staticmethod
    def name() -> str:
        return "mini-swe-agent-v1"

    def version(self) -> str | None:
        return "1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        _logger = logger.getChild(__name__)
        loop = asyncio.get_event_loop()

        model = LitellmModelAdapter(
            model_name=self._model_name,
            api_base=self._api_base,
            temperature=self._temperature,
            model_kwargs={"drop_params": True},
        )
        env = HarborEnvironmentAdapter(environment, loop)

        agent = DefaultAgent(
            model,
            env,
            step_limit=self._max_turns,
        )

        # Run the official v1 agent loop (synchronous — run in thread)
        exit_status, exit_message = await asyncio.to_thread(agent.run, instruction)
        _logger.info("Agent finished: %s — %s", exit_status, exit_message[:100])

        # Extract metrics
        context.n_input_tokens = sum(
            (m.get("extra", {}).get("response", {}).get("usage", {}).get("prompt_tokens", 0)) for m in agent.messages
        )
        context.n_output_tokens = sum(
            (m.get("extra", {}).get("response", {}).get("usage", {}).get("completion_tokens", 0))
            for m in agent.messages
        )

        # Convert messages to ATIF trajectory
        steps: list[Step] = []
        step_id = 1
        for msg in agent.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                steps.append(
                    Step(
                        step_id=step_id, timestamp=datetime.now(timezone.utc).isoformat(), source="system",
                        message=content,
                    )
                )
            elif role == "user":
                if step_id <= 2:
                    steps.append(
                        Step(
                            step_id=step_id, timestamp=datetime.now(timezone.utc).isoformat(), source="user",
                            message=content,
                        )
                    )
                elif steps and steps[-1].source == "agent":
                    steps[-1].observation = Observation(results=[ObservationResult(content=content)])
                else:
                    steps.append(
                        Step(
                            step_id=step_id, timestamp=datetime.now(timezone.utc).isoformat(), source="user",
                            message=content,
                        )
                    )
            elif role == "assistant":
                import re

                bash_cmd = None
                matches = re.findall(r"```bash\s*\n(.*?)\n```", content, re.DOTALL)
                if matches:
                    bash_cmd = matches[0].strip()

                tool_calls = None
                if bash_cmd:
                    tool_calls = [
                        ToolCall(
                            tool_call_id=f"call_{step_id}",
                            function_name="bash_command",
                            arguments={"command": bash_cmd},
                        )
                    ]

                usage = msg.get("extra", {}).get("response", {}).get("usage", {})
                metrics = None
                if usage:
                    metrics = Metrics(
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                    )

                steps.append(
                    Step(
                        step_id=step_id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="agent",
                        model_name=self._model_name,
                        message=content,
                        tool_calls=tool_calls,
                        metrics=metrics,
                    )
                )
            step_id += 1

        trajectory = Trajectory(
            schema_version="ATIF-v1.2",
            session_id=str(uuid.uuid4()),
            agent=Agent(name="mini-swe-agent-v1", version="1.0", model_name=self._model_name),
            steps=steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=context.n_input_tokens or 0,
                total_completion_tokens=context.n_output_tokens or 0,
            ),
        )
        traj_path = self.logs_dir / "trajectory.json"
        traj_path.write_text(json.dumps(trajectory.to_json_dict(), indent=2))

        # Also save the raw mini-swe-agent messages for debugging
        raw_path = self.logs_dir / "mini-swe-agent.trajectory.json"
        raw_path.write_text(
            json.dumps(
                {
                    "messages": agent.messages,
                    "info": {
                        "exit_status": exit_status,
                        "exit_message": exit_message,
                        "model_stats": {"instance_cost": model.cost, "n_calls": model.n_calls},
                    },
                },
                indent=2,
                default=str,
            )
        )
        _logger.info("Trajectories saved to %s", self.logs_dir)
