# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_sotopia_agent(monkeypatch: pytest.MonkeyPatch) -> tuple[type[object], type[object]]:
    class BaseInstalledAgent:
        def __init__(self, *, logs_dir: Path) -> None:
            self.logs_dir = logs_dir

    class BaseEnvironment:
        pass

    class AgentContext:
        def __init__(self) -> None:
            self.metadata: dict[str, object] = {}

        def is_empty(self) -> bool:
            return not self.metadata

    modules = {
        "harbor": ModuleType("harbor"),
        "harbor.agents": ModuleType("harbor.agents"),
        "harbor.agents.installed": ModuleType("harbor.agents.installed"),
        "harbor.agents.installed.base": ModuleType("harbor.agents.installed.base"),
        "harbor.environments": ModuleType("harbor.environments"),
        "harbor.environments.base": ModuleType("harbor.environments.base"),
        "harbor.models": ModuleType("harbor.models"),
        "harbor.models.agent": ModuleType("harbor.models.agent"),
        "harbor.models.agent.context": ModuleType("harbor.models.agent.context"),
    }
    modules["harbor.agents.installed.base"].BaseInstalledAgent = BaseInstalledAgent  # type: ignore[attr-defined]
    modules["harbor.environments.base"].BaseEnvironment = BaseEnvironment  # type: ignore[attr-defined]
    modules["harbor.models.agent.context"].AgentContext = AgentContext  # type: ignore[attr-defined]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).parents[2] / "lib/marin/src/marin/evaluation/harbor/sotopia_agent.py"
    spec = importlib.util.spec_from_file_location("marin_sotopia_agent_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load SOTOPIA agent from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SotopiaAgent, AgentContext


def test_agent_allows_missing_episode_summary_after_failed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sotopia_agent, context_type = _load_sotopia_agent(monkeypatch)
    context = context_type()

    sotopia_agent(logs_dir=tmp_path).populate_context_post_run(context)

    assert context.is_empty()
