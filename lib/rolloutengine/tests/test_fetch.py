# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from taskcompendium.fetch import FETCH_ACTION_INTERFACE, FETCH_TOOL, HttpPage, NetworkFailure, fetch_requirement
from taskcompendium.models import EnvironmentRequirements

from rolloutengine.harness.fetch import FetchProvider
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.test_support import (
    SHELL,
    ReplayModel,
    arithmetic_task,
    engine,
    rollout_spec,
    tool_call,
    tool_observations,
)

PAGE_URL = "https://fetch-fixture.invalid/answer"
FETCH_HARNESS = {SHELL: ShellHarness(command_timeout=None, tools={FETCH_ACTION_INTERFACE: FetchProvider()})}

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    "page,observation",
    [
        pytest.param(
            HttpPage(status=503, body="Unavailable"),
            {"url": PAGE_URL, "status": 503, "body": "Unavailable"},
            id="http_error",
        ),
        pytest.param(
            NetworkFailure(error="Connection reset"),
            {"url": PAGE_URL, "error": "Connection reset"},
            id="network_failure",
        ),
    ],
)
async def test_fetch_provider_reports_declared_failures_as_observations(page, observation):
    task = arithmetic_task(
        environment_requirements=EnvironmentRequirements(tool_providers={"web": fetch_requirement({PAGE_URL: page})}),
        interaction_tools=(FETCH_TOOL,),
    )
    model = ReplayModel(
        [tool_call(FETCH_TOOL.name, {"url": PAGE_URL}, call_id="fetch"), {"role": "assistant", "content": "No answer"}]
    )

    graded = await engine(model, harnesses=FETCH_HARNESS).run_task(rollout_spec(task))

    assert tool_observations(graded) == [observation]
    assert graded.grade.reward == 0.0


async def test_fetch_provider_answers_unknown_urls_and_bad_arguments_as_observations():
    task = arithmetic_task(
        environment_requirements=EnvironmentRequirements(
            tool_providers={"web": fetch_requirement({PAGE_URL: HttpPage(status=200, body="12")})}
        ),
        interaction_tools=(FETCH_TOOL,),
    )
    model = ReplayModel(
        [
            tool_call(FETCH_TOOL.name, {"address": PAGE_URL}, call_id="bad"),
            tool_call(FETCH_TOOL.name, {"url": "https://elsewhere.invalid"}, call_id="unknown"),
            tool_call(FETCH_TOOL.name, {"url": PAGE_URL}, call_id="declared"),
            {"role": "assistant", "content": "12"},
        ]
    )

    graded = await engine(model, harnesses=FETCH_HARNESS).run_task(rollout_spec(task, max_turns=4))

    observations = tool_observations(graded)
    assert [observation.get("status") for observation in observations] == [None, 404, 200]
    assert observations[2]["body"] == "12"
    assert graded.grade.reward == 1.0
