# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harbor.agents.installed.sotopia import SotopiaAgent


@pytest.mark.asyncio
async def test_agent_install_restages_task_config(tmp_path: Path) -> None:
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    task_config = environment_dir / "task_config.json"
    task_config.write_text('{"environment_id": "env"}\n')
    environment = SimpleNamespace(
        environment_dir=environment_dir,
        upload_file=AsyncMock(),
    )

    await SotopiaAgent(logs_dir=tmp_path).install(environment)

    environment.upload_file.assert_any_await(
        task_config,
        "/opt/sotopia/task_config.json",
    )
