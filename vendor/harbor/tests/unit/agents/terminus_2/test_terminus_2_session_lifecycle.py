"""A reused Terminus agent must open a terminal after phase cleanup."""

import json
import shutil
import tempfile

import pytest

from harbor.agents.terminus_2 import terminus_2
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.environments.daytona import DaytonaEnvironment
from harbor.llms.base import BaseLLM, LLMResponse
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from shellbox.machine import MachineSpec, PackageSpec
from tests.unit.environments.transport import ProcessMachine


class ScriptedLLM(BaseLLM):
    _api_base = "http://scripted.test/v1"

    def __init__(self):
        self.commands = iter(
            ["printf first > first.txt\n", "", "cat first.txt > second.txt\n", ""]
        )

    async def call(self, prompt: str, **kwargs) -> LLMResponse:
        command = next(self.commands)
        return LLMResponse(
            content=json.dumps(
                {
                    "analysis": "Write the phase output.",
                    "plan": "Complete the instruction.",
                    "commands": (
                        [{"keystrokes": command, "duration": 0.1}] if command else []
                    ),
                    "task_complete": True,
                }
            )
        )

    def get_model_context_limit(self) -> int:
        return 32768

    def get_model_output_limit(self) -> int:
        return 4096


async def test_run_after_cleanup_starts_a_new_terminal_and_retains_phase_files(
    tmp_path, monkeypatch
):
    if shutil.which("tmux") is None:
        pytest.skip("The terminal lifecycle regression requires installed tmux")

    async def token_count(url, payload):
        return 128

    monkeypatch.setattr(Terminus2, "_init_llm", lambda self, **kwargs: ScriptedLLM())
    monkeypatch.setattr(terminus_2, "_server_token_count", token_count)
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    adapter = DaytonaEnvironment(
        environment_dir=environment_dir,
        environment_name="session-lifecycle",
        session_id="session-lifecycle",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="test/image"),
        client_factory=lambda: pytest.fail("Unexpected provider access"),
    )
    with tempfile.TemporaryDirectory(prefix="harbor-tmux-") as socket_dir:
        machine = ProcessMachine(
            tmp_path / "guest",
            MachineSpec(
                PackageSpec(),
                workdir="/workspace",
                env={"TMUX_TMPDIR": socket_dir, "TMUX": ""},
            ),
        )
        adapter.machine = machine
        agent = Terminus2(logs_dir=tmp_path / "agent", model_name="test/terminal")
        try:
            await agent.setup(adapter)
            await agent.run("Write first.txt.", adapter, AgentContext())
            await agent.cleanup(adapter)
            assert machine.path("/workspace/first.txt").read_text() == "first"
            assert (
                await adapter.exec("tmux has-session -t terminus-2")
            ).return_code != 0

            await agent.run("Copy first.txt to second.txt.", adapter, AgentContext())
            await agent.cleanup(adapter)
            assert machine.path("/workspace/second.txt").read_text() == "first"
            assert (
                await adapter.exec("tmux has-session -t terminus-2")
            ).return_code != 0
        finally:
            await adapter.exec("tmux kill-server")
            await adapter.stop(delete=True)
