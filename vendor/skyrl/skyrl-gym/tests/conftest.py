import json

import pytest
import pytest_asyncio
from rolloutengine.models import ModelTurn
from skyrl_gym.nemotron_tasks import NemotronTaskSession
from skyrl_gym.rollout_specs import rollout_spec
from skyrl_gym.source_task import source_task
from taskcompendium.models import Source

SESSION_LIMITS = {
    "task_session": "nemotron_ultra",
    "max_turns": 50,
    "max_total_tokens": None,
    "model_turn_timeout": None,
    "tool_turn_timeout": None,
    "total_turn_timeout": None,
    "attempt_timeout": None,
    "verifier_timeout": None,
    "cleanup_timeout": 5,
}


@pytest.fixture
def model_turn():
    def make(text, *, stop_reason="stop", message=None):
        message = {"role": "assistant", "content": text} if message is None else message
        return ModelTurn(message, (10, 11), (20, 21), (-0.5, -0.5), stop_reason, text=text, metadata={})

    return make


@pytest_asyncio.fixture
async def nemotron_session():
    """Source-row sessions for agents whose grading needs no task machine."""
    sessions = []

    async def make(agent, record, config=None):
        ultra = {"route": "task_session", "agent": agent, "record_json": json.dumps(record), "request_json": "{}"}
        task = source_task(
            [{"role": "user", "content": "Public task question"}],
            {"extra_info": {"nemotron_ultra": ultra}},
            {} if config is None else config,
            Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        )
        session = NemotronTaskSession(rollout_spec(task, SESSION_LIMITS), None)
        sessions.append(session)
        await session.prepare()
        return session

    try:
        yield make
    finally:
        for session in sessions:
            await session.close()
