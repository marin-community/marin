# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle behavior across Harbor's simulator and Docker process boundaries."""

import asyncio
import json
import os
import sys
from pathlib import Path

import msgspec
import pytest
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from taskcompendium.execution import HarborTaskBinding
from taskcompendium.execution import ShellSimEnvironment as ShellSimRequirement
from taskcompendium.harbor.environments import ShellSimEnvironment, TaskDockerEnvironment
from taskcompendium.shellsim import SHELLSIM_REVISION, ShellSimError, ShellSimTimeout


async def test_shellsim_exec_timeout_reaps_child_and_rejects_later_actions(tmp_path: Path):
    # Respond to bootstrap requests, then stall on the agent command. A real
    # child exercises the transport deadline and process cleanup together.
    script = tmp_path / "bridge"
    pid_file = tmp_path / "child.pid"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, os, signal, sys\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    if request.get('command') == 'hang':\n"
        "        signal.pause()\n"
        f"    result = {{'revision': {SHELLSIM_REVISION!r}}}\n"
        "    if request['op'] == 'exec':\n"
        "        result = {'stdout': '', 'stderr': '', 'return_code': 0, 'stop_reason': None,\n"
        "                  'usage': dict(cpu_used=0, memory_current=0, memory_peak=0,\n"
        "                                disk_current=0, disk_peak=0, output_bytes=0)}\n"
        "    print(json.dumps({'ok': True, 'result': result}), flush=True)\n"
    )
    script.chmod(0o755)
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (tmp_path / "binding.json").write_bytes(msgspec.json.encode(HarborTaskBinding(ShellSimRequirement())))
    environment = ShellSimEnvironment(
        environment_dir=environment_dir,
        environment_name="timeout",
        session_id="timeout",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(workdir="/app"),
        bridge_path=str(script),
    )
    await environment.start(force_build=False)
    assert environment.session is not None
    environment.session.timeout = 1
    try:
        with pytest.raises(ShellSimTimeout):
            await asyncio.wait_for(environment.exec("hang", timeout_sec=0.05), timeout=0.5)
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)
        with pytest.raises(ShellSimError):
            await environment.exec("echo gone")
    finally:
        await environment.stop(delete=True)


@pytest.mark.parametrize(
    "delete,keep_containers,container_exists,volume_exists",
    [(False, False, False, True), (True, False, False, False), (False, True, True, True), (True, True, True, True)],
)
async def test_docker_stop_preserves_requested_resources_and_shared_image(
    tmp_path: Path, monkeypatch, delete, keep_containers, container_exists, volume_exists
):
    # Fake only the Docker CLI. Persist its resource state so assertions observe
    # the cleanup effect through the real Harbor subprocess boundary.
    state_path = tmp_path / "docker-state.json"
    state_path.write_text(json.dumps(dict(running=True, container=True, volume=True, image=True)))
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[2:]\n"
        "while args and args[0].startswith('-'):\n"
        "    args = args[2:]\n"
        "path = Path(os.environ['TEST_DOCKER_STATE'])\n"
        "state = json.loads(path.read_text())\n"
        "if args[0] in ('stop', 'down'):\n"
        "    state['running'] = False\n"
        "if args[0] == 'down':\n"
        "    state['container'] = False\n"
        "    if '--volumes' in args:\n"
        "        state['volume'] = False\n"
        "    if '--rmi' in args:\n"
        "        state['image'] = False\n"
        "path.write_text(json.dumps(state))\n"
    )
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("TEST_DOCKER_STATE", str(state_path))
    environment = TaskDockerEnvironment(
        environment_dir=tmp_path,
        environment_name="cleanup",
        session_id="cleanup",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="sha256:" + "a" * 64, workdir="/app"),
        keep_containers=keep_containers,
    )
    await environment.stop(delete=delete)
    assert json.loads(state_path.read_text()) == dict(
        running=False, container=container_exists, volume=volume_exists, image=True
    )
