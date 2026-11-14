"""Installed harnesses share process ownership through Daytona's execution boundary."""

import asyncio
import json
import logging
import os
import shlex
import shutil
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.agents.installed.codex import Codex
from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.agents.installed.opencode import OpenCode
from harbor.agents.installed.pi import Pi
from harbor.agents.terminus_2 import Terminus2
from harbor.agents.terminus_kira import TerminusKira
from harbor.environments.base import AgentProcessCleanupError
from harbor.environments.daytona import DaytonaEnvironment
from harbor.llms.base import BaseLLM, LLMResponse
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
from harbor.models.trial.paths import TrialPaths
from harbor.trial.errors import AgentTimeoutError
from harbor.trial.single_step import SingleStepTrial
from shellbox.machine import MachineSpec, PackageSpec

from tests.unit.environments.transport import ProcessMachine


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper and procfs contract")
async def test_forged_guest_completion_cannot_authorize_verification(tmp_path, monkeypatch):
    monkeypatch.setattr("harbor.environments.base.CLEANUP_TIMEOUT", 1)
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    environment = DaytonaEnvironment(
        environment_dir=environment_dir,
        environment_name="forged-completion",
        session_id="forged-completion",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="test/image"),
        client_factory=lambda: pytest.fail("Unexpected provider access"),
    )
    environment.machine = ProcessMachine(tmp_path / "guest", MachineSpec(PackageSpec()))
    scope = await environment.start_process_scope("sleep 60", env={})
    control = Path(scope.directory)
    try:
        (control / "done").write_text("complete")
        with pytest.raises(AgentProcessCleanupError):
            await environment.kill_processes()
        assert (control / "pid").exists()
    finally:
        (control / "done").unlink(missing_ok=True)
        await environment.kill_processes()
        await environment.machine.close()


@pytest.mark.skipif(
    sys.platform != "linux", reason="Installed harnesses execute on Linux"
)
@pytest.mark.parametrize("agent_type", [Pi, MiniSweAgent, Codex, OpenCode])
@pytest.mark.parametrize("ending", ["complete", "timeout", "cancel"])
async def test_installed_harness_stops_detached_tools_before_verification(
    tmp_path, monkeypatch, agent_type, ending
):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture")
    monkeypatch.setenv("MSWEA_API_KEY", "fixture")
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.delenv("CODEX_FORCE_AUTH_JSON", raising=False)
    for directory in ("bin", "logs", ".nvm", ".local/bin"):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / ".nvm/nvm.sh").write_text("nvm() { :; }\n")
    (tmp_path / ".local/bin/env").write_text(":\n")
    (tmp_path / "bin/node").write_text("#!/bin/sh\nexec cat\n")
    (tmp_path / "bin/node").chmod(0o755)
    os.mkfifo(tmp_path / "block")
    script = "#!" + sys.executable + "\n" + """
import os, signal
from pathlib import Path
home = Path(os.environ['HOME'])
r, w = os.pipe()
if os.fork() == 0:
    os.close(r)
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (home / 'child.pid').write_text(str(os.getpid()))
    os.write(w, b'1')
    os.close(w)
    null = os.open('/dev/null', os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(null, fd)
    with (home / 'block').open() as fifo:
        fifo.read()
    (home / 'late').touch()
    os._exit(0)
os.close(w)
os.read(r, 1)
print('READY', flush=True)
"""
    if ending != "complete":
        script += "signal.pause()\n"
    executable = tmp_path / "bin" / agent_type.name()
    executable.write_text(script)
    executable.chmod(0o755)
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    environment = DaytonaEnvironment(
        environment_dir=environment_dir,
        environment_name="lifecycle",
        session_id="lifecycle",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="test/image"),
        persistent_env={
            "HOME": str(tmp_path),
            "PATH": f"{tmp_path}/bin:/usr/local/bin:/usr/bin:/bin",
        },
        client_factory=lambda: pytest.fail("Unexpected provider access"),
    )
    environment.machine = ProcessMachine(
        tmp_path / "guest",
        MachineSpec(PackageSpec(), workdir="/workspace"),
    )
    environment.machine.path("/logs/agent").mkdir(parents=True)
    environment.agent_timeout_sec = 2 if ending == "timeout" else None
    agent = agent_type(
        logs_dir=tmp_path / "logs",
        model_name="openai/test-model",
        extra_env={"OPENAI_API_KEY": "EMPTY", "MSWEA_API_KEY": "EMPTY"},
    )
    task = asyncio.create_task(agent.run("Run a tool", environment, AgentContext()))
    try:
        async with asyncio.timeout(10):
            while not (tmp_path / "child.pid").exists():
                # The tool publishes its PID after detaching and before blocking.
                await asyncio.sleep(0)
        if ending == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif ending == "timeout":
            with pytest.raises(AgentTimeoutError):
                await task
        else:
            await task
        pid = int((tmp_path / "child.pid").read_text())
        assert not Path(f"/proc/{pid}").exists()
        assert not (tmp_path / "late").exists()
        verifier = await asyncio.wait_for(environment.exec("printf verifier-ready"), 5)
        assert (verifier.return_code, verifier.stdout) == (0, "verifier-ready")
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await environment.machine.close()


@pytest.mark.skipif(
    sys.platform != "linux", reason="Linux subreaper and real tmux contract"
)
@pytest.mark.parametrize("agent_type", [Terminus2, TerminusKira])
@pytest.mark.parametrize("ending", ["complete", "timeout", "cancel"])
async def test_terminus_phase_preserves_background_tools_then_reaps_them(
    tmp_path, monkeypatch, agent_type, ending
):
    if shutil.which("tmux") is None:
        pytest.skip("tmux is not installed")
    ready, release = asyncio.Event(), asyncio.Event()
    tool = tmp_path / "tool.py"
    tool.write_text(
        "import os,signal\nfrom pathlib import Path\n"
        f"root=Path({str(tmp_path)!r})\n"
        "if os.fork()==0:\n"
        " os.setsid()\n signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        " (root/'child.pid').write_text(str(os.getpid()))\n"
        " with (root/'release').open() as fifo: fifo.read()\n"
        " (root/'late').touch()\n"
    )
    os.mkfifo(tmp_path / "release")

    class Model(BaseLLM):
        def __init__(self, **kwargs):
            self.first = True

        def get_model_context_limit(self):
            return 32768

        def get_model_output_limit(self):
            return 4096

        async def call(self, prompt, **kwargs):
            if self.first:
                self.first = False
                payload = {
                    "analysis": "run",
                    "plan": "run",
                    "commands": [
                        {
                            "keystrokes": f"python3 {shlex.quote(str(tool))}\n",
                            "duration": 0.1,
                        }
                    ],
                    "task_complete": False,
                }
                name = "execute_commands"
            else:
                async with asyncio.timeout(10):
                    while not (tmp_path / "child.pid").exists():
                        await asyncio.sleep(0)
                ready.set()
                await release.wait()
                payload = {
                    "analysis": "done",
                    "plan": "done",
                    "commands": [],
                    "task_complete": True,
                }
                name = "task_complete"
            if agent_type is TerminusKira:
                return LLMResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": uuid.uuid4().hex,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(payload),
                            },
                        }
                    ],
                )
            return LLMResponse(content=json.dumps(payload))

    monkeypatch.setattr("harbor.agents.terminus_2.terminus_2.OpenAILLM", Model)
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    environment = DaytonaEnvironment(
        environment_dir=environment_dir,
        environment_name="phase",
        session_id="phase",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="test/image"),
        persistent_env={"HOME": str(tmp_path), "PATH": os.environ["PATH"]},
        client_factory=lambda: pytest.fail("Unexpected provider access"),
    )
    environment.machine = ProcessMachine(
        tmp_path / "guest", MachineSpec(PackageSpec(), workdir="/workspace")
    )
    environment.machine.path("/logs/agent").mkdir(parents=True)
    agent = agent_type(
        logs_dir=tmp_path / "logs",
        model_name="openai/test",
        max_turns=4,
        tokenizer_encoding="cl100k_base",
    )
    await agent.setup(environment)
    trial = object.__new__(SingleStepTrial)
    trial.agent = agent
    trial.agent_environment = environment
    trial.config = TrialConfig(
        agent=AgentConfig(name=agent_type.name(), model_name="openai/test"),
        task=TaskConfig(path=tmp_path),
        trial_name="phase",
        trial_cleanup_grace_sec=15,
    )
    trial.task = SimpleNamespace(name="phase")
    trial._hooks = defaultdict(list)
    trial._result = SimpleNamespace()
    trial._first_phase_timeout = None
    trial._pending_cleanup_tasks = set()
    trial.logger = logging.getLogger(__name__)
    target = SimpleNamespace(agent_result=None, agent_execution=None)
    pending = asyncio.create_task(
        trial._run_agent_phase(
            target=target,
            instruction="Run a tool",
            timeout_sec=2 if ending == "timeout" else None,
            user=None,
        )
    )
    try:
        waiting = asyncio.create_task(ready.wait())
        try:
            done, _ = await asyncio.wait(
                {waiting, pending}, timeout=10, return_when=asyncio.FIRST_COMPLETED
            )
            if pending in done:
                await pending
            assert waiting in done, "The agent did not reach its second model call"
        finally:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
        pid = int((tmp_path / "child.pid").read_text())
        assert Path(f"/proc/{pid}").exists()
        assert await agent._session.is_session_alive()
        await environment.exec("printf retained > answer")
        if ending == "complete":
            release.set()
            await pending
        elif ending == "cancel":
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            with pytest.raises(AgentTimeoutError):
                await pending
        assert not Path(f"/proc/{pid}").exists()
        assert not (tmp_path / "late").exists()
        verified = await environment.exec("cat answer")
        assert verified.stdout == "retained"
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await environment.kill_processes()
        await environment.machine.close()
