# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure scheduling choices and runner reconciliation boundaries."""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from ops_workflow.runner import MAX_EVIDENCE_BYTES, MAX_RESULT_BYTES, AgentRunner, SessionRef, SessionSpec
from ops_workflow.state import CaseOutcome
from ops_workflow.turn import (
    ACTIVE_TURN_STATES,
    AgentTurn,
    OpsResult,
    ResultEvidence,
    TurnState,
    acknowledge_turn,
    claim_turn,
    complete_turn,
    interrupt_turn,
    renew_turn_lease,
)

MAX_PROMPT_BYTES = 32 * 1024
RESULT_SCHEMA_VERSION = 1
NO_ACTION_TAKEN = "none"


class ResultValidationError(ValueError):
    """The runner result artifact violates the ops result contract."""


@dataclass(frozen=True)
class SchedulerConfig:
    lease_duration: timedelta = timedelta(minutes=10)
    turn_timeout: timedelta = timedelta(minutes=20)


@dataclass(frozen=True)
class LaunchedTurn:
    turn: AgentTurn
    session: SessionRef


def next_turn_to_claim(
    turns: Iterable[AgentTurn],
    *,
    now: datetime,
    lease_owner: str,
    config: SchedulerConfig,
) -> AgentTurn | None:
    """Claim the highest-priority eligible turn when the global slot is free."""

    turn_list = tuple(turns)
    if any(turn.state in ACTIVE_TURN_STATES for turn in turn_list):
        return None
    eligible = [turn for turn in turn_list if turn.state == TurnState.QUEUED and turn.available_at <= now]
    if not eligible:
        return None
    selected = min(eligible, key=lambda turn: (-turn.priority, turn.available_at, turn.created_at, turn.id))
    return claim_turn(
        selected,
        now=now,
        lease_owner=lease_owner,
        lease_duration=config.lease_duration,
        turn_timeout=config.turn_timeout,
    )


async def launch_claimed_turn(
    turn: AgentTurn,
    *,
    runner: AgentRunner,
    session_spec: SessionSpec,
    evidence: bytes,
    now: datetime,
    config: SchedulerConfig,
) -> LaunchedTurn:
    """Launch a previously persisted claim and record its exact turn number."""

    if turn.state != TurnState.LAUNCHING:
        raise ValueError("turn must be persisted as launching before runner I/O")
    if len(evidence) > MAX_EVIDENCE_BYTES:
        raise ValueError(f"evidence exceeds {MAX_EVIDENCE_BYTES} bytes")
    if len(turn.prompt.encode()) > MAX_PROMPT_BYTES:
        raise ValueError(f"prompt exceeds {MAX_PROMPT_BYTES} bytes")

    session = await runner.ensure_session(session_spec)
    await runner.upload_evidence(session.id, evidence)
    acknowledgement = await runner.start_turn(session.id, turn.client_request_id, turn.prompt)
    launched = acknowledge_turn(
        turn,
        loom_turn_number=acknowledgement.number,
        now=now,
        lease_duration=config.lease_duration,
    )
    return LaunchedTurn(turn=launched, session=session)


async def reconcile_running_turn(
    turn: AgentTurn,
    *,
    runner: AgentRunner,
    runner_session_id: str,
    case_id: str,
    now: datetime,
    config: SchedulerConfig,
) -> AgentTurn:
    """Renew, interrupt, or complete one exact running turn."""

    if turn.state != TurnState.RUNNING or turn.loom_turn_number is None:
        raise ValueError("turn must be running with a Loom turn number")
    if turn.deadline_at is None:
        raise ValueError("running turn is missing its durable deadline")
    if now >= turn.deadline_at:
        await runner.interrupt_turn(runner_session_id, turn.loom_turn_number)
        return interrupt_turn(turn, now=now, error="turn deadline exceeded")

    snapshot = await runner.turn_snapshot(runner_session_id, turn.loom_turn_number)
    if snapshot.active:
        return renew_turn_lease(turn, now=now, lease_duration=config.lease_duration)
    if not snapshot.terminal:
        raise ResultValidationError("runner reported neither active nor terminal")

    artifact = await runner.result_artifact(runner_session_id)
    result = validated_result(artifact.content, case_id=case_id, turn_id=turn.id)
    return complete_turn(turn, now=now, result=result)


def validated_result(raw_result: bytes, *, case_id: str, turn_id: str) -> OpsResult:
    """Validate the bounded result artifact for one ops turn."""

    if len(raw_result) > MAX_RESULT_BYTES:
        raise ResultValidationError(f"result exceeds {MAX_RESULT_BYTES} bytes")
    try:
        result = json.loads(raw_result)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResultValidationError("result must be UTF-8 JSON") from error
    if not isinstance(result, dict):
        raise ResultValidationError("result root must be an object")
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ResultValidationError("result schema_version must be 1")
    if result.get("case_id") != case_id or result.get("ops_turn_id") != turn_id:
        raise ResultValidationError("result identifiers do not match the active turn")
    try:
        outcome = CaseOutcome(result.get("outcome"))
    except ValueError as error:
        raise ResultValidationError("result outcome is invalid") from error
    if result.get("action_taken") != NO_ACTION_TAKEN:
        raise ResultValidationError("v1 result action_taken must be none")
    summary = _required_string(result, "summary")
    recommended_next_step = _required_string(result, "recommended_next_step")
    raw_evidence = result.get("evidence")
    if not isinstance(raw_evidence, list):
        raise ResultValidationError("result evidence must be an array")
    evidence: list[ResultEvidence] = []
    for index, item in enumerate(raw_evidence):
        if not isinstance(item, dict):
            raise ResultValidationError(f"result evidence[{index}] must be an object")
        evidence.append(
            ResultEvidence(
                claim=_required_string(item, "claim", prefix=f"evidence[{index}]"),
                source=_required_string(item, "source", prefix=f"evidence[{index}]"),
            )
        )
    return OpsResult(
        schema_version=RESULT_SCHEMA_VERSION,
        case_id=case_id,
        ops_turn_id=turn_id,
        outcome=outcome,
        summary=summary,
        evidence=tuple(evidence),
        action_taken=NO_ACTION_TAKEN,
        recommended_next_step=recommended_next_step,
    )


def _required_string(result: dict[str, object], field: str, *, prefix: str = "result") -> str:
    value = result.get(field)
    if not isinstance(value, str) or not value:
        raise ResultValidationError(f"{prefix} {field} must be a non-empty string")
    return value
