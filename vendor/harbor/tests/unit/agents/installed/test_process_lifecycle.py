"""Installed harnesses share process ownership through Daytona's execution boundary."""

import asyncio
import os
import sys
from pathlib import Path

import pytest

from harbor.agents.installed.codex import Codex
from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.agents.installed.opencode import OpenCode
from harbor.agents.installed.pi import Pi
from harbor.environments.daytona import DaytonaEnvironment
from harbor.models.agent.context import AgentContext
from harbor.trial.errors import AgentTimeoutError
from shellbox.machine import MachineSpec, PackageSpec
from tests.unit.environments.transport import ProcessMachine


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
    environment = object.__new__(DaytonaEnvironment)
    environment.machine = ProcessMachine(
        tmp_path / "guest",
        MachineSpec(PackageSpec(), workdir="/workspace"),
    )
    environment.machine.path("/logs/agent").mkdir(parents=True)
    environment.session_id = "lifecycle"
    environment.default_user = None
    environment._persistent_env = {
        "HOME": str(tmp_path),
        "PATH": f"{tmp_path}/bin:/usr/local/bin:/usr/bin:/bin",
    }
    environment.agent_timeout_sec = 2 if ending == "timeout" else None
    agent = agent_type(
        logs_dir=tmp_path / "logs", model_name="openai/test-model",
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
