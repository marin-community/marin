# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Externally timed trials retain verification without bypassing the attempt watchdog."""

import asyncio

import pytest

from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.core import ActorDeadlineOwner
from rolloutengine.harness.external import AgentHarness
from rolloutengine.models import GradedRolloutInterrupted, Prompt, RolloutOperation
from rolloutengine.test_support import ReplayModel, arithmetic_task, rollout_spec


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner,attempt_timeout,verification_finishes",
    [
        (ActorDeadlineOwner.ENGINE, 5, False),
        (ActorDeadlineOwner.HARNESS, 5, True),
        (ActorDeadlineOwner.HARNESS, 0.02, False),
    ],
)
async def test_harness_owned_actor_timeout_keeps_verification_and_attempt_watchdog(
    owner, attempt_timeout, verification_finishes
):
    verified = False

    async def trial(context, complete):
        nonlocal verified
        await complete(Prompt(({"role": "user", "content": "What is six plus six?"},), {}))
        # The external trial exhausts its own actor phase, then verifies. A
        # duplicate recorder clock must not cancel that separate verifier phase.
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0):
                await asyncio.Event().wait()
        verifier_done = asyncio.Event()
        release = asyncio.get_running_loop().call_later(0.05, verifier_done.set)
        try:
            await verifier_done.wait()
            verified = True
        finally:
            release.cancel()

    spec = rollout_spec(
        arithmetic_task(),
        harness="trial",
        total_turn_timeout=0.01,
        attempt_timeout=attempt_timeout,
    )
    engine = RolloutEngine({}, {"trial": AgentHarness(trial)}, actor_deadline_owners={"trial": owner})
    model = ReplayModel([{"role": "assistant", "content": "12"}])
    if owner is ActorDeadlineOwner.HARNESS and attempt_timeout < 0.05:
        with pytest.raises(GradedRolloutInterrupted) as interrupted:
            await engine.run_task(spec, model.complete)
        assert interrupted.value.operation is RolloutOperation.ATTEMPT
    else:
        result = await engine.run_task(spec, model.complete)
        assert result.grade.reward == 1
        assert result.rollout.conversations[0].response_token_ids == (20,)
    assert verified is verification_finishes
