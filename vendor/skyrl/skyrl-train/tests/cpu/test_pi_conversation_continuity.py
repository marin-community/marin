import copy
import json
from pathlib import Path

import pytest
from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, TrialResult
from harbor.models.verifier.result import VerifierResult
from rolloutengine.harness.external import AgentHarness
from rolloutengine.models import Prompt
from rolloutengine.recorder import harness_rollout
from rolloutengine.test_support import ReplayModel, arithmetic_task, rollout_limits

from skyrl_train.rollouts.harbor_trial_session import HarborTrialHarness


@pytest.mark.asyncio
@pytest.mark.parametrize("harness_name", ["harbor", "agent"])
@pytest.mark.parametrize("compact", [False, True])
async def test_real_pi_roundtrip_preserves_all_turns_and_compaction_starts_a_conversation(harness_name, compact):
    # Captured server/Pi assistant messages from Pi 0.87.0, including its JSON
    # reserialization and absent/null/empty optional-field differences.
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/pi-0.87-roundtrip.json").read_text())
    model = ReplayModel([copy.deepcopy(fixture["server_assistant"]) for _ in range(60)])

    async def run(complete):
        messages = ({"role": "user", "content": "Inspect the workspace."},)
        for index in range(60):
            await complete(Prompt(messages, {}))
            if compact and index == 29:
                messages = ({"role": "user", "content": "Intentional compacted summary."},)
            else:
                messages = (
                    *messages,
                    copy.deepcopy(fixture["pi_assistant"]),
                    {"role": "tool", "tool_call_id": fixture["pi_assistant"]["tool_calls"][0]["id"], "content": "ok"},
                )
        return TrialResult(
            task_name="swe",
            trial_name="pi-roundtrip",
            trial_uri="/tmp/pi-roundtrip",
            task_id=TaskConfig(path=Path("/tmp/task")).get_task_id(),
            task_checksum="fixture",
            agent_info=AgentInfo(name="pi", version="0.87.0"),
            config=TrialConfig(task=TaskConfig(path=Path("/tmp/task")), agent=AgentConfig(name="pi")),
            verifier_result=VerifierResult(rewards={"reward": 1.0}),
        )

    async def agent(context, complete):
        await run(complete)

    harness = HarborTrialHarness(run, "reward") if harness_name == "harbor" else AgentHarness(agent)
    result = await harness_rollout(harness, arithmetic_task(), model.complete, limits=rollout_limits(max_turns=60))
    assert len(result.conversations) == (2 if compact else 1)
    assert [len(conversation.steps) for conversation in result.conversations] == ([30, 30] if compact else [60])
    for conversation in result.conversations:
        generated = sum(len(step.turn.response_token_ids) for step in conversation.steps)
        assert sum(conversation.loss_mask) == generated
    assert sum(sum(conversation.loss_mask) for conversation in result.conversations) == 60
    assert sum(request.continuation is None for request in model.requests) == (2 if compact else 1)
