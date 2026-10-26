"""Installed harnesses share process ownership through Daytona's execution boundary."""

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from harbor.agents.installed.codex import Codex
from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.agents.installed.opencode import OpenCode
from harbor.agents.installed.pi import Pi
from harbor.environments.daytona import DaytonaEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from harbor.trial.errors import AgentTimeoutError


class LocalSessions:
    """Daytona SDK boundary with real commands and a queue per session."""

    def __init__(self, home: Path):
        self.home = home
        self.sessions = {}
        self.commands = {}
        self.ready = asyncio.Event()

    async def create_session(self, session):
        self.sessions[session] = asyncio.Lock()

    async def delete_session(self, session):
        del self.sessions[session]

    async def execute_session_command(self, session, request, timeout=None):
        task = asyncio.create_task(self.execute(session, request.command))
        command_id = str(len(self.commands))
        self.commands[command_id] = task
        if request.run_async:
            return SimpleNamespace(cmd_id=command_id)
        return SimpleNamespace(exit_code=(await task).return_code)

    async def execute(self, session, command):
        async with self.sessions[session]:
            process = await asyncio.create_subprocess_exec(
                "/bin/bash",
                "-c",
                command.replace("/logs/agent", str(self.home / "logs")),
                env={"HOME": str(self.home), "PATH": f"{self.home}/bin:/usr/bin:/bin"},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            errors = asyncio.create_task(process.stderr.read())
            lines = []
            async for line in process.stdout:
                lines.append(line)
                if line.strip() == b"READY":
                    self.ready.set()
            return SimpleNamespace(
                return_code=await process.wait(),
                stdout=b"".join(lines).decode(),
                stderr=(await errors).decode(),
            )

    async def get_session_command(self, session, command):
        task = self.commands[command]
        return SimpleNamespace(
            id=command, exit_code=task.result().return_code if task.done() else None
        )

    async def get_session_command_logs(self, session, command):
        return self.commands[command].result()


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
    script = """#!/usr/bin/python3
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
    transport = LocalSessions(tmp_path)
    environment = DaytonaEnvironment(
        environment_dir=tmp_path,
        environment_name="lifecycle",
        session_id="lifecycle",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="fixture"),
    )
    environment._sandbox = SimpleNamespace(id="fixture", process=transport)
    environment.agent_timeout_sec = 2 if ending == "timeout" else None
    agent = agent_type(logs_dir=tmp_path / "logs", model_name="openai/test-model")
    task = asyncio.create_task(agent.run("Run a tool", environment, AgentContext()))
    try:
        await asyncio.wait_for(transport.ready.wait(), 10)
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
        await asyncio.gather(*transport.commands.values())
