# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from ops_workflow.loom import AgentPrompt, AgentSessionRequest, StubAgentGateway


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_stub_gateway_preserves_session_identity_across_process_restarts():
    first_gateway = StubAgentGateway(completion_delay=0)
    first = await first_gateway.create_session(
        AgentSessionRequest(
            name="ops-case-case-a",
            title="Case A",
            goal="Inspect case A",
            case_id="case-a",
            turn_id="turn-a",
        )
    )

    restarted_gateway = StubAgentGateway(completion_delay=0)
    second = await restarted_gateway.create_session(
        AgentSessionRequest(
            name="ops-case-case-b",
            title="Case B",
            goal="Inspect case B",
            case_id="case-b",
            turn_id="turn-b",
        )
    )
    resumed_turn = await restarted_gateway.prompt(
        first.id,
        AgentPrompt(
            text="Recheck case A",
            actor="operator@example.com",
            case_id="case-a",
            turn_id="turn-a-follow-up",
        ),
    )

    assert first.id != second.id
    assert resumed_turn == 0
    artifact = await restarted_gateway.artifact(first.id, "ops-result")
    assert json.loads(artifact.content)["ops_turn_id"] == "turn-a-follow-up"
