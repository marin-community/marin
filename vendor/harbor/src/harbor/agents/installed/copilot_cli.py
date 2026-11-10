"""GitHub Copilot CLI agent implementation for Harbor."""

import json
import re
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    with_prompt_template,
)
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

_UNSUPPORTED_REASON = (
    "Copilot CLI is unsupported: GitHub credentials must remain on the host "
    "and no guest model-relay authentication path is supported; "
    "use mini-swe-agent or terminus-2"
)


class CopilotCli(BaseInstalledAgent):
    """GitHub Copilot CLI agent.

    Execution is blocked because its GitHub authentication cannot use the
    guest's declared model relay. Trajectory parsing remains available.
    """

    SUPPORTS_ATIF: bool = True

    _TRAJECTORY_FILENAME = "copilot-cli.jsonl"
    CLI_FLAGS = [
        CliFlag(
            "reasoning_effort",
            cli="--effort",
            type="enum",
            choices=["low", "medium", "high", "xhigh"],
            env_fallback="COPILOT_CLI_EFFORT",
        ),
    ]

    _RE_VERSION = re.compile(r"(\d+\.\d+\.\d+)")

    @staticmethod
    def name() -> str:
        """Return the unique name of the agent."""
        return AgentName.COPILOT_CLI.value

    def get_version_command(self) -> str | None:
        """Return the command to get the agent version."""
        return 'export PATH="$HOME/.local/bin:$PATH"; copilot --version'

    def parse_version(self, stdout: str) -> str:
        """Parse the agent version from the command output."""
        text = stdout.strip()
        match = self._RE_VERSION.search(text)
        return match.group(1) if match else text

    async def install(self, environment: BaseEnvironment) -> None:
        raise RuntimeError(_UNSUPPORTED_REASON)

    def _read_copilot_cli_jsonl(self, jsonl_path: Path) -> list[dict[str, Any]]:
        """Read and parse the Copilot CLI JSONL output file."""
        try:
            text = jsonl_path.read_text(encoding="utf-8")
        except OSError as ex:
            self.logger.debug(
                "Error reading Copilot CLI trajectory file %s: %s",
                jsonl_path,
                ex,
                exc_info=True,
            )
            return []

        if text.startswith("Error: No authentication information found"):
            self.logger.error("Copilot CLI authentication error:\n%s", text)
            return []

        raw_events: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw_events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        if not raw_events and text.strip():
            self.logger.debug(
                "Copilot CLI output file %s contained no valid JSON. "
                "Raw content (first 500 chars): %s",
                jsonl_path,
                text[:500],
            )

        return raw_events

    def _convert_jsonl_to_trajectory(self, jsonl_path: Path) -> Trajectory | None:
        """Convert Copilot CLI JSONL output to ATIF trajectory."""

        raw_events = self._read_copilot_cli_jsonl(jsonl_path)
        if not raw_events:
            return None

        step_id = 1
        steps: list[Step] = []
        total_input_tokens = 0
        total_output_tokens = 0

        for event in raw_events:
            event_type = event.get("type")
            timestamp = event.get("timestamp")

            # --- message ---
            if event_type == "message":
                role = event.get("role", "user")
                source = "agent" if role == "assistant" else "user"
                content = event.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    )

                step = Step(
                    step_id=step_id,
                    timestamp=timestamp,
                    source=source,
                    message=content,
                )

                model_name_val = event.get("model")
                if source == "agent" and model_name_val:
                    step.model_name = model_name_val

                steps.append(step)
                step_id += 1

            # --- tool_use ---
            elif event_type == "tool_use":
                tool_name = event.get("name", "")
                arguments = event.get("input", {})

                tool_call = ToolCall(
                    tool_call_id=event.get("id", ""),
                    function_name=tool_name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                )

                step = Step(
                    step_id=step_id,
                    timestamp=timestamp,
                    source="agent",
                    message=f"Executed {tool_name}",
                    tool_calls=[tool_call],
                )

                if model_name_val := event.get("model"):
                    step.model_name = model_name_val

                steps.append(step)
                step_id += 1

            # --- tool_result ---
            elif event_type == "tool_result":
                content = event.get("content")
                if isinstance(content, list):
                    content = "\n".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    )

                if steps and steps[-1].tool_calls:
                    steps[-1].observation = Observation(
                        results=[
                            ObservationResult(
                                source_call_id=event.get("tool_use_id") or None,
                                content=content or None,
                            )
                        ]
                    )
                    continue

                step = Step(
                    step_id=step_id,
                    timestamp=timestamp,
                    source="agent",
                    message=content or "Tool result",
                )

                steps.append(step)
                step_id += 1

            # --- usage ---
            elif event_type == "usage":
                input_tokens = event.get("input_tokens", 0)
                output_tokens = event.get("output_tokens", 0)

                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

                if steps and steps[-1].source == "agent":
                    steps[-1].metrics = Metrics(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                    )

        if not steps:
            return None

        default_model = self.model_alias
        for step in steps:
            if step.source == "agent" and step.model_name:
                default_model = step.model_name
                break

        return Trajectory(
            schema_version="ATIF-v1.6",
            session_id="copilot-cli",
            agent=Agent(
                name="copilot-cli",
                version=self.version() or "unknown",
                model_name=default_model,
            ),
            steps=steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=total_input_tokens or None,
                total_completion_tokens=total_output_tokens or None,
                total_steps=len(steps),
            ),
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        """
        After running the agent, parse the Copilot CLI JSONL output
        and populate the context with token counts.
        """
        jsonl_path = self.logs_dir / self._TRAJECTORY_FILENAME
        if not jsonl_path.exists():
            return

        try:
            trajectory = self._convert_jsonl_to_trajectory(jsonl_path)
        except Exception as e:
            self.logger.debug(
                "Error converting Copilot CLI trajectory from file %s: %s",
                jsonl_path,
                e,
                exc_info=True,
            )
            return

        if trajectory and trajectory.final_metrics:
            context.n_input_tokens, context.n_output_tokens = (
                trajectory.final_metrics.total_prompt_tokens or 0,
                trajectory.final_metrics.total_completion_tokens or 0,
            )

        if trajectory:
            atif_path = self.logs_dir / "trajectory.json"
            try:
                atif_path.write_text(
                    json.dumps(
                        trajectory.to_json_dict(),
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            except Exception as e:
                self.logger.debug(
                    "Error writing ATIF trajectory to file %s: %s",
                    atif_path,
                    e,
                    exc_info=True,
                )

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        raise RuntimeError(_UNSUPPORTED_REASON)
