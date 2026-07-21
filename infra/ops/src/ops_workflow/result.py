# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validated result contract produced by the ops-expert agent."""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TypeVar, cast
from uuid import UUID

SCHEMA_VERSION = 2
MAX_SUMMARY_BYTES = 8_000
MAX_REASON_BYTES = 2_000
MAX_NEXT_STEP_BYTES = 4_000
MAX_EVIDENCE_ITEMS = 20
MAX_EVIDENCE_FIELD_BYTES = 2_000
EnumValue = TypeVar("EnumValue", bound=StrEnum)


class OpsOutcome(StrEnum):
    NO_ACTION = "no_action"
    ACTION_RECOMMENDED = "action_recommended"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class EscalationSeverity(StrEnum):
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True)
class EscalationRequest:
    severity: EscalationSeverity
    reason: str


@dataclass(frozen=True)
class ResultEvidence:
    claim: str
    source: str


@dataclass(frozen=True)
class OpsResult:
    schema_version: int
    case_id: str
    ops_turn_id: str
    outcome: OpsOutcome
    summary: str
    evidence: tuple[ResultEvidence, ...]
    action_taken: str
    recommended_next_step: str
    escalation: EscalationRequest | None

    def as_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


def parse_ops_result(content: str, *, case_id: str, turn_id: str) -> OpsResult:
    """Parse and validate the agent artifact against the active database turn."""

    try:
        raw = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("ops-result is not valid JSON") from error
    if not isinstance(raw, dict):
        raise ValueError("ops-result must be a JSON object")
    result = cast(Mapping[str, object], raw)
    schema_version = _integer(result, "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"ops-result schema_version must be {SCHEMA_VERSION}")
    parsed_case_id = _uuid(result, "case_id")
    parsed_turn_id = _uuid(result, "ops_turn_id")
    if parsed_case_id != case_id:
        raise ValueError("ops-result case_id does not match the active case")
    if parsed_turn_id != turn_id:
        raise ValueError("ops-result ops_turn_id does not match the active turn")

    outcome = _enum(result, "outcome", OpsOutcome)
    summary = _bounded_string(result, "summary", MAX_SUMMARY_BYTES)
    action_taken = _bounded_string(result, "action_taken", 64)
    if action_taken != "none":
        raise ValueError("ops-result action_taken must be 'none'")
    evidence = _evidence(result.get("evidence"))
    recommended_next_step = _bounded_string(result, "recommended_next_step", MAX_NEXT_STEP_BYTES)
    escalation = _escalation(result.get("escalation"), outcome=outcome)
    return OpsResult(
        schema_version=schema_version,
        case_id=parsed_case_id,
        ops_turn_id=parsed_turn_id,
        outcome=outcome,
        summary=summary,
        evidence=evidence,
        action_taken=action_taken,
        recommended_next_step=recommended_next_step,
        escalation=escalation,
    )


def _evidence(value: object) -> tuple[ResultEvidence, ...]:
    if not isinstance(value, list):
        raise ValueError("ops-result evidence must be an array")
    if len(value) > MAX_EVIDENCE_ITEMS:
        raise ValueError(f"ops-result evidence may contain at most {MAX_EVIDENCE_ITEMS} items")
    items: list[ResultEvidence] = []
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError("ops-result evidence items must be objects")
        item = cast(Mapping[str, object], raw)
        items.append(
            ResultEvidence(
                claim=_bounded_string(item, "claim", MAX_EVIDENCE_FIELD_BYTES),
                source=_bounded_string(item, "source", MAX_EVIDENCE_FIELD_BYTES),
            )
        )
    return tuple(items)


def _escalation(value: object, *, outcome: OpsOutcome) -> EscalationRequest | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("ops-result escalation must be an object or null")
    if outcome not in (OpsOutcome.ACTION_RECOMMENDED, OpsOutcome.BLOCKED):
        raise ValueError("only action_recommended or blocked results may request escalation")
    item = cast(Mapping[str, object], value)
    return EscalationRequest(
        severity=_enum(item, "severity", EscalationSeverity),
        reason=_bounded_string(item, "reason", MAX_REASON_BYTES),
    )


def _integer(value: Mapping[str, object], field: str) -> int:
    item = value.get(field)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"ops-result {field} must be an integer")
    return item


def _uuid(value: Mapping[str, object], field: str) -> str:
    item = _bounded_string(value, field, 64)
    try:
        return str(UUID(item))
    except ValueError as error:
        raise ValueError(f"ops-result {field} must be a UUID") from error


def _bounded_string(value: Mapping[str, object], field: str, max_bytes: int) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"ops-result {field} must be a non-empty string")
    item = item.strip()
    if len(item.encode()) > max_bytes:
        raise ValueError(f"ops-result {field} exceeds {max_bytes} bytes")
    return item


def _enum(value: Mapping[str, object], field: str, enum_type: type[EnumValue]) -> EnumValue:
    item = _bounded_string(value, field, 64)
    try:
        return enum_type(item)
    except ValueError as error:
        allowed = ", ".join(member.value for member in enum_type)
        raise ValueError(f"ops-result {field} must be one of: {allowed}") from error
