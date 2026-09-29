# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test-only replay fixtures for Harbor's agent and verifier lifecycle."""

from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial

from taskcompendium.harbor.adapter import _record_submission
from taskcompendium.lowering import SPECIFICATION_FILE, read_specification
from taskcompendium.submission import conversation_messages


class ReplayAgent(BaseAgent):
    """Test-only agent that replays a fixed assistant message."""

    def __init__(self, *args, response: dict[str, Any], messages: list[dict[str, Any]], **kwargs):
        super().__init__(*args, **kwargs)
        self.response = response
        self.messages = messages

    @staticmethod
    def name() -> str:
        return "taskcompendium-replay"

    def version(self) -> str:
        return "0.1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        _record_submission(self.logs_dir, self.messages, self.response, context)


async def run_replay_trial(
    task_dir: Path,
    response: dict[str, Any],
    trials_dir: Path,
    trial_name: str,
) -> TrialResult:
    """Exercise Harbor grading with a fixed assistant message instead of a model."""
    specification = read_specification(task_dir / SPECIFICATION_FILE)
    config = TrialConfig.model_validate(
        {
            "task": {"path": str(task_dir.resolve())},
            "trials_dir": str(trials_dir.resolve()),
            "trial_name": trial_name,
            "environment": {"import_path": "taskcompendium.harbor.adapter:NoToolEnvironment"},
            "agent": {
                "import_path": f"{__name__}:ReplayAgent",
                "kwargs": {"response": response, "messages": conversation_messages(specification.context)},
            },
            "verifier": {"import_path": "taskcompendium.harbor.adapter:SemanticVerifier"},
        }
    )
    trial = await Trial.create(config)
    return await trial.run()
