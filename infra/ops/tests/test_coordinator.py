# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from ops_workflow.coordinator import (
    ResultValidationError,
    SchedulerConfig,
    launch_claimed_turn,
    next_turn_to_claim,
    reconcile_running_turn,
)
from ops_workflow.runner import FakeAgentRunner, SessionSpec
from ops_workflow.turn import AgentTurn, TurnKind, TurnState

NOW = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)
CONFIG = SchedulerConfig(lease_duration=timedelta(minutes=10), turn_timeout=timedelta(minutes=20))
SESSION_SPEC = SessionSpec(
    name="ops-case-case-1-session-a",
    case_id="case-1",
    title="DNS warning",
    repo_revision="deadbeef",
)


def queued_turn(turn_id: str, *, priority: int, kind: TurnKind = TurnKind.AUTOMATIC) -> AgentTurn:
    return AgentTurn(
        id=turn_id,
        session_id=f"session-{turn_id}",
        kind=kind,
        state=TurnState.QUEUED,
        priority=priority,
        requested_by="system" if kind == TurnKind.AUTOMATIC else "operator@example.com",
        client_request_id=f"request-{turn_id}",
        prompt=f"Investigate {turn_id}",
        available_at=NOW,
        created_at=NOW,
    )


def serialized_result(case_id: str, turn_id: str) -> bytes:
    return json.dumps(
        {
            "schema_version": 1,
            "case_id": case_id,
            "ops_turn_id": turn_id,
            "outcome": "no_action",
            "summary": "The warning is benign and no action is required.",
            "evidence": [{"claim": "Pod remained ready", "source": "kubectl get pod"}],
            "action_taken": "none",
            "recommended_next_step": "Continue monitoring.",
        }
    ).encode()


def test_next_turn_to_claim_prioritizes_manual_work_without_preempting_active_turn() -> None:
    automatic = queued_turn("automatic", priority=10)
    manual = queued_turn("manual", priority=100, kind=TurnKind.QUESTION)

    claimed = next_turn_to_claim((automatic, manual), now=NOW, lease_owner="scheduler-1", config=CONFIG)

    assert claimed is not None
    assert claimed.id == "manual"
    assert claimed.state == TurnState.LAUNCHING
    assert claimed.deadline_at == NOW + CONFIG.turn_timeout
    assert next_turn_to_claim((automatic, claimed), now=NOW, lease_owner="scheduler-2", config=CONFIG) is None


def test_fake_runner_lost_ack_retry_adopts_same_session_and_turn() -> None:
    async def run() -> None:
        runner = FakeAgentRunner()
        claimed = next_turn_to_claim(
            (queued_turn("turn-1", priority=10),),
            now=NOW,
            lease_owner="scheduler-1",
            config=CONFIG,
        )
        assert claimed is not None

        first = await launch_claimed_turn(
            claimed,
            runner=runner,
            session_spec=SESSION_SPEC,
            evidence=b"{}",
            now=NOW,
            config=CONFIG,
        )
        retry = await launch_claimed_turn(
            claimed,
            runner=runner,
            session_spec=SESSION_SPEC,
            evidence=b"{}",
            now=NOW,
            config=CONFIG,
        )

        assert retry.session == first.session
        assert retry.turn.loom_turn_number == first.turn.loom_turn_number == 1

    asyncio.run(run())


def test_reconcile_running_turn_accepts_only_matching_structured_result() -> None:
    async def run() -> None:
        runner = FakeAgentRunner()
        claimed = next_turn_to_claim(
            (queued_turn("turn-1", priority=10),),
            now=NOW,
            lease_owner="scheduler-1",
            config=CONFIG,
        )
        assert claimed is not None
        launched = await launch_claimed_turn(
            claimed,
            runner=runner,
            session_spec=SESSION_SPEC,
            evidence=b"{}",
            now=NOW,
            config=CONFIG,
        )
        assert launched.turn.loom_turn_number is not None
        await runner.complete_turn(
            launched.session.id,
            launched.turn.loom_turn_number,
            serialized_result("case-1", launched.turn.id),
        )

        completed = await reconcile_running_turn(
            launched.turn,
            runner=runner,
            runner_session_id=launched.session.id,
            case_id="case-1",
            now=NOW + timedelta(minutes=1),
            config=CONFIG,
        )

        assert completed.state == TurnState.SUCCEEDED
        assert completed.result is not None
        assert completed.result.outcome == "no_action"

    asyncio.run(run())


def test_reconcile_running_turn_rejects_result_for_another_turn() -> None:
    async def run() -> None:
        runner = FakeAgentRunner()
        claimed = next_turn_to_claim(
            (queued_turn("turn-1", priority=10),),
            now=NOW,
            lease_owner="scheduler-1",
            config=CONFIG,
        )
        assert claimed is not None
        launched = await launch_claimed_turn(
            claimed,
            runner=runner,
            session_spec=SESSION_SPEC,
            evidence=b"{}",
            now=NOW,
            config=CONFIG,
        )
        assert launched.turn.loom_turn_number is not None
        await runner.complete_turn(
            launched.session.id,
            launched.turn.loom_turn_number,
            serialized_result("case-1", "different-turn"),
        )

        with pytest.raises(ResultValidationError):
            await reconcile_running_turn(
                launched.turn,
                runner=runner,
                runner_session_id=launched.session.id,
                case_id="case-1",
                now=NOW + timedelta(minutes=1),
                config=CONFIG,
            )

    asyncio.run(run())


def test_reconcile_running_turn_interrupts_exact_turn_at_durable_deadline() -> None:
    async def run() -> None:
        runner = FakeAgentRunner()
        claimed = next_turn_to_claim(
            (queued_turn("turn-1", priority=10),),
            now=NOW,
            lease_owner="scheduler-1",
            config=CONFIG,
        )
        assert claimed is not None
        launched = await launch_claimed_turn(
            claimed,
            runner=runner,
            session_spec=SESSION_SPEC,
            evidence=b"{}",
            now=NOW,
            config=CONFIG,
        )

        interrupted = await reconcile_running_turn(
            launched.turn,
            runner=runner,
            runner_session_id=launched.session.id,
            case_id="case-1",
            now=NOW + CONFIG.turn_timeout,
            config=CONFIG,
        )

        assert interrupted.state == TurnState.INTERRUPTED
        assert interrupted.completed_at == NOW + CONFIG.turn_timeout

    asyncio.run(run())
