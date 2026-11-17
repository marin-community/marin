import asyncio
from pathlib import Path

import pytest

from harbor.environments.base import ExecResult
from harbor.environments.setup import run_environment_setup
from harbor.trial.errors import EnvironmentSetupError
from harbor_config.models.task.config import TaskConfig


class LocalFixtureEnvironment:
    def __init__(self, root: Path):
        self.root = root

    async def exec(self, command: str, *, timeout_sec: float, user: str) -> ExecResult:
        assert user == "root"
        async with asyncio.timeout(timeout_sec):
            process = await asyncio.create_subprocess_exec(
                "/bin/sh", "-c", command, cwd=self.root,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
        return ExecResult(return_code=process.returncode, stdout=stdout.decode(), stderr=stderr.decode())


@pytest.mark.asyncio
async def test_released_setup_hook_initializes_fixture_before_agent_and_keeps_failures_infra(tmp_path):
    fixtures = tmp_path / "setup_files"
    fixtures.mkdir()
    (fixtures / "calibforge-initialize.sh").write_text("mkdir -p app; printf public-fixture > app/input.txt\n")
    cfg = TaskConfig.model_validate_toml('''
[environment.setup]
command = "sh setup_files/calibforge-initialize.sh"
timeout_sec = 180.0
''')
    assert cfg.environment.setup is not None
    await run_environment_setup(LocalFixtureEnvironment(tmp_path), cfg.environment.setup)
    assert (tmp_path / "app/input.txt").read_text() == "public-fixture"
    (fixtures / "calibforge-initialize.sh").write_text("echo broken-fixture >&2; exit 7\n")
    with pytest.raises(EnvironmentSetupError, match="broken-fixture"):
        await run_environment_setup(LocalFixtureEnvironment(tmp_path), cfg.environment.setup)
