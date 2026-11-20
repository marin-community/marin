import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.installed.mini_swe_agent import MiniSweAgent


@pytest.mark.asyncio
async def test_failed_install_preserves_error_tail_without_protected_credentials(tmp_path):
    secret = "synthetic-protected-credential"
    agent = MiniSweAgent(logs_dir=tmp_path, protected_values=frozenset({secret}))
    stderr = "package installation preamble\n" * 100 + secret + "\nterminal dependency failure"
    environment = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=2, stdout="", stderr=stderr)))
    with pytest.raises(NonZeroAgentExitCodeError) as caught:
        await agent.exec_as_agent(environment, command="false")
    evidence = json.loads(next((tmp_path / "command-failures").glob("*.json")).read_text())
    assert evidence["stderr"].endswith("<redacted>\nterminal dependency failure")
    assert evidence["stderr"].startswith("package installation preamble")
    assert "terminal dependency failure" in str(caught.value)
    assert secret not in str(caught.value)
    assert secret not in json.dumps(evidence)


@pytest.mark.asyncio
async def test_evidence_write_failure_keeps_the_command_exit_as_primary_error(tmp_path, monkeypatch):
    agent = MiniSweAgent(logs_dir=tmp_path)
    environment = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=2, stdout="", stderr="dependency failed")))

    def cannot_write(*args, **kwargs):
        raise PermissionError("read-only evidence directory")

    monkeypatch.setattr(type(tmp_path), "write_text", cannot_write)
    with pytest.raises(NonZeroAgentExitCodeError) as caught:
        await agent.exec_as_agent(environment, command="false")
    assert isinstance(caught.value.__cause__, PermissionError)
    assert caught.value.result.return_code == 2
    assert "dependency failed" in str(caught.value)
