# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live: author a builder program with GLM-5.3 and run it to a TaskSpec whose positive control passes.

Takes the ``parallel_key`` fixture (skips without the Parallel key file) for the template's
research step. Artifacts go to ``<evidence_root>/build/live-test/<timestamp>/``, where ``evidence_root``
is the fixture in ``tests/conftest.py``.
"""

import time
from pathlib import Path

import httpx
import pytest
from rolloutengine.contracts import ModelRequest, ModelTurn
from rolloutengine.engine import ShellboxRolloutEngine
from taskcompendium.grading_result import Outcome

from taskforge.build.author import author
from taskforge.build.run import item_id_for, run_build
from taskforge.build.sdk import BuildServices
from taskforge.build.step import CacheStatus
from taskforge.build.template import standard
from taskforge.ledger.jsonl import JsonlLedger
from taskforge.llm.client import GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.web import web_tools
from taskforge.sandbox.factories import MachineHost, machine_factories
from taskforge.spec.controls import ControlKind
from taskforge.validate.controls import control_turns


def scripted(turns: tuple[dict, ...]):
    """A model that serves ``turns`` in order, with token ids that extend each request's prefix."""

    async def model(request: ModelRequest) -> ModelTurn:
        served = sum(message["role"] == "assistant" for message in request.messages)
        message = turns[served]
        prompt = (*request.prefix_token_ids, 2 * served + 1)
        stop = "tool_calls" if "tool_calls" in message else "stop"
        return ModelTurn(message, prompt, (2 * served + 2,), None, stop)

    return model


@pytest.fixture
def evidence_dir(evidence_root: Path) -> Path:
    return evidence_root / "build" / "live-test"


@pytest.mark.live_glm
@pytest.mark.timeout(5400)
async def test_authored_program_builds_a_task_its_positive_control_passes(
    glm_settings, parallel_key, image_cache, proposal, evidence_dir
):
    run_dir = evidence_dir / time.strftime("%Y%m%d-%H%M%S")
    factories = machine_factories(MachineHost.LAPTOP, controller_url=None, image_cache=image_cache)
    endpoint = GlmEndpoint(base_url=glm_settings.base_url, token=glm_settings.token, pool=Pool.HIGH)
    async with GlmClient(endpoint) as client, httpx.AsyncClient() as http:
        services = BuildServices(
            client=client,
            policy=LLMPolicy(),
            factories=factories,
            ledger=JsonlLedger(run_dir / "ledger"),
            web_tools=web_tools(http, parallel_key.value),
        )
        item_dir = run_dir / item_id_for(proposal)
        program = await author(proposal, standard, item_dir, services, item_id_for(proposal))
        draft = await run_build(program, proposal, item_dir, run_dir / "cache", services)
        again = await run_build(program, proposal, item_dir, run_dir / "cache", services)

    assert {record.status for record in again.provenance.steps} == {CacheStatus.HIT}
    assert again.task == draft.task

    # Replay the positive control's whole transcript (shell turns included) through RolloutEngine.
    positive = next(c for c in draft.controls if c.kind == ControlKind.POSITIVE)
    turns = control_turns(positive)
    engine = ShellboxRolloutEngine(
        scripted(turns),
        factories,
        max_turns=len(turns),
        command_timeout=60,
        cleanup_timeout=60,
        convention=draft.convention,
    )
    rollout = await engine.run(draft.task, execution=draft.execution)
    assert rollout.grade.status == Outcome.GRADED
    assert positive.expect.reward_min is not None and rollout.grade.reward >= positive.expect.reward_min
