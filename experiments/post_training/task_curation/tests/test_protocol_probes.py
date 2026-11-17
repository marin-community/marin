# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from rolloutengine.harness.fetch import FetchProvider
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.test_support import (
    SHELL,
    ReplayModel,
    engine,
    machine_runtime,
    rollout_spec,
    tool_call,
    tool_observations,
)
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend
from taskcompendium.fetch import FETCH_ACTION_INTERFACE
from taskcompendium.models import CommandSemantics, EnvironmentRequirements

from experiments.post_training.task_curation.datasets.protocol_probes import (
    NONCE_FILE,
    PROBE_URL,
    ProtocolCheck,
    probe_nonce,
    probe_tasks,
)

pytestmark = pytest.mark.asyncio

FIXTURE_IMAGE = "fixture@sha256:" + "0" * 64
ENVIRONMENT = EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=FIXTURE_IMAGE)
TASKS = {task.id: task for task in probe_tasks(grader_environment=ENVIRONMENT, variants=1)}
FETCH_HARNESS = {SHELL: ShellHarness(command_timeout=None, tools={FETCH_ACTION_INTERFACE: FetchProvider()})}


def reply(content: str) -> dict:
    return {"role": "assistant", "content": content}


TOOL_TASK = TASKS[f"{ProtocolCheck.TOOL_EXECUTION}-0"]


@pytest.mark.parametrize(
    "turns,reward",
    [
        pytest.param(
            [
                tool_call("run_command", {"command": f"cat /{NONCE_FILE}"}, call_id="run"),
                reply(probe_nonce(TOOL_TASK.id)),
            ],
            1.0,
            id="executed",
        ),
        pytest.param([reply(f"$ cat /{NONCE_FILE}\n0123456789abcdef")], 0.0, id="narrated"),
    ],
)
async def test_tool_probe_rewards_only_an_executed_command(turns, reward):
    model = ReplayModel(turns)

    graded = await engine(model, {Backend.SHELLSIM: ShellSimMachineFactory()}).run_task(
        rollout_spec(TOOL_TASK, machine=machine_runtime(backend=Backend.SHELLSIM))
    )

    assert graded.grade.reward == reward


async def test_url_probe_serves_declared_page_as_observation():
    task = TASKS[f"{ProtocolCheck.URL_LOADING}-0"]
    answer = probe_nonce(task.id)
    model = ReplayModel([tool_call("fetch_url", {"url": PROBE_URL}, call_id="fetch"), reply(answer)])

    graded = await engine(model, harnesses=FETCH_HARNESS).run_task(rollout_spec(task))

    [served] = tool_observations(graded)
    assert served == {"url": PROBE_URL, "status": 200, "body": answer}
    assert graded.grade.reward == 1.0
