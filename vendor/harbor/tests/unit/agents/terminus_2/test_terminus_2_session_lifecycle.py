"""A reused Terminus agent must open a terminal after phase cleanup."""

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

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


class ForegroundLLM(ScriptedLLM):
    def __init__(self, foreground, background):
        self.command_sent = False
        self.foreground = foreground
        self.background = background

    async def call(self, prompt: str, **kwargs) -> LLMResponse:
        if self.command_sent:
            await asyncio.Event().wait()
        self.command_sent = True
        return LLMResponse(
            content=json.dumps(
                {
                    "analysis": "Start a service, then run a foreground command.",
                    "plan": "Leave the service available for grading.",
                    "commands": [
                        {
                            "keystrokes": (
                                f"{self.background} & echo $! > background.pid; "
                                f"{self.foreground}; "
                                "printf unexpected > after-timeout\n"
                            ),
                            "duration": 0.1,
                        }
                    ],
                    "task_complete": False,
                }
            )
        )


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
            # The keystroke duration does not wait for shell completion.
            ready = await adapter.exec(
                "until test -s first.txt; do sleep 0.01; done", timeout_sec=5
            )
            assert ready.return_code == 0
            await agent.cleanup(adapter)
            assert machine.path("/workspace/first.txt").read_text() == "first"
            assert (
                await adapter.exec("tmux has-session -t terminus-2")
            ).return_code != 0

            await agent.run("Copy first.txt to second.txt.", adapter, AgentContext())
            ready = await adapter.exec(
                "until test -s second.txt; do sleep 0.01; done", timeout_sec=5
            )
            assert ready.return_code == 0
            await agent.cleanup(adapter)
            assert machine.path("/workspace/second.txt").read_text() == "first"
            assert (
                await adapter.exec("tmux has-session -t terminus-2")
            ).return_code != 0
        finally:
            await adapter.exec("tmux kill-server")
            await adapter.stop(delete=True)


@pytest.mark.parametrize(
    "foreground_kind", ["external", "builtin", "zombie-descendant"]
)
async def test_cancelled_run_stops_foreground_job_and_preserves_background_service(
    tmp_path, monkeypatch, foreground_kind
):
    if shutil.which("tmux") is None:
        pytest.skip("The terminal lifecycle regression requires installed tmux")

    async def token_count(url, payload):
        return 128

    fixture_path = Path(__file__).parent / "fixtures/foreground_zombie.py"
    fixture = "foreground_zombie.py"
    foreground = {
        "external": "sh -c 'echo $$ > foreground.pid; exec sleep 3600'",
        "builtin": "echo $$ > foreground.pid; while :; do :; done",
        "zombie-descendant": (
            "until test -p join-group; do sleep 0.01; done; "
            f"python3 {fixture} foreground"
        ),
    }[foreground_kind]
    background = (
        f"python3 {fixture} holder"
        if foreground_kind == "zombie-descendant"
        else "sleep 3600"
    )
    monkeypatch.setattr(
        Terminus2,
        "_init_llm",
        lambda self, **kwargs: ForegroundLLM(foreground, background),
    )
    monkeypatch.setattr(terminus_2, "_server_token_count", token_count)
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    adapter = DaytonaEnvironment(
        environment_dir=environment_dir,
        environment_name="foreground-lifecycle",
        session_id="foreground-lifecycle",
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
        if foreground_kind == "zombie-descendant":
            shutil.copyfile(
                fixture_path, machine.path("/workspace/foreground_zombie.py")
            )
        agent = Terminus2(logs_dir=tmp_path / "agent", model_name="test/terminal")
        running = asyncio.create_task(
            agent.run("Run a service.", adapter, AgentContext())
        )
        try:
            ready = await adapter.exec(
                "until test -s foreground.pid; do sleep 0.01; done", timeout_sec=5
            )
            assert ready.return_code == 0
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(running, 5)

            if foreground_kind != "builtin":
                assert (
                    await adapter.exec("kill -0 $(cat foreground.pid)")
                ).return_code != 0
            if foreground_kind == "zombie-descendant":
                descendant = int(machine.path("/workspace/descendant.pid").read_text())
                assert (
                    Path(f"/proc/{descendant}/stat")
                    .read_text()
                    .split(") ", 1)[1]
                    .split()[0]
                    == "Z"
                )
            assert (
                await adapter.exec("kill -0 $(cat background.pid)")
            ).return_code == 0
            assert (
                await adapter.exec("tmux has-session -t terminus-2")
            ).return_code == 0
            assert not machine.path("/workspace/after-timeout").exists()
            await adapter.exec(
                "tmux send-keys -t terminus-2 'printf recovered > resumed' Enter"
            )
            assert (
                await adapter.exec(
                    "until test -s resumed; do sleep 0.01; done", timeout_sec=5
                )
            ).return_code == 0
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
            try:
                if foreground_kind == "zombie-descendant":
                    await adapter.exec(
                        "if test -s background.pid; then kill -TERM $(cat background.pid); "
                        "until test -f descendant-reaped; do sleep 0.01; done; fi",
                        timeout_sec=5,
                    )
            finally:
                await agent.cleanup(adapter)
                await adapter.exec("tmux kill-server")
                await adapter.stop(delete=True)
