# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retained nl2bash tasks executed by ShellSim and graded by the original checker."""

import json
from pathlib import Path

import pytest

from taskcompendium.harbor.runner import run_trial
from taskcompendium.importers.tasktrove import read_archive
from taskcompendium.importers.tasktrove_shell import import_task
from taskcompendium.lowering import export_task
from taskcompendium.models import ChatWithTools, ContainerRuntime, ExecutionConfig, FinalState, Protocol, Rejected

pytestmark = pytest.mark.docker
FIXTURE = Path(__file__).parent / "fixtures/shell/script-row-16158.tar.gz"


@pytest.mark.parametrize("attempt,reward", [("good", 1.0), ("bad", 0.0), ("empty", 0.0), ("shadow", 0.0)])
async def test_real_nl2bash_shellsim_with_original_checker(tmp_path, runtime_image, bridge, attempt, reward):
    archive = read_archive(FIXTURE.read_bytes(), "16158", "shell-cmd")
    spec = import_task(archive, verifier_runtime=ContainerRuntime(runtime_image))
    assert not isinstance(spec, Rejected)
    # The pinned simulator copies source/. as a directory. Assemble the named
    # fixture explicitly, as allowed by the original task requirements.
    commands = ["bash /setup_files/setup_seeds.sh", "cp /setup_files/seeds/file /workspace/file"]
    if attempt == "good":
        commands.append(
            "{ grep -o . file | tr A-Z a-z | sort | uniq -c | sort -nr; } > /output/command_capture.txt 2>&1"
        )
    elif attempt == "shadow":
        commands += [
            "mkdir -p __external__/output",
            "{ grep -o . file | tr A-Z a-z | sort | uniq -c | sort -nr; } "
            "> __external__/output/command_capture.txt 2>&1",
        ]
    elif attempt == "bad":
        commands.append("echo wrong > /output/command_capture.txt")
    else:
        commands.append(": > /output/command_capture.txt")
    task = export_task(
        spec,
        Protocol("shell", ChatWithTools(), FinalState((".", "/output/command_capture.txt"))),
        ExecutionConfig("replay", spec.environment),
        tmp_path / "task",
        agent_kwargs={"commands": commands},
        environment_kwargs={"bridge_path": bridge},
    )
    execution = json.loads((task / "execution.json").read_text())
    result = await run_trial(task, execution, tmp_path / "trials", "shell")
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": reward}
    transcript = json.loads((tmp_path / "trials/shell/agent/transcript.json").read_text())
    observations = [entry["content"] for entry in transcript if entry["role"] == "tool"]
    assert all(observation["return_code"] == 0 for observation in observations), observations
