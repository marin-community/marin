# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select explicitly superseded derived-rule receipts without hiding unknown conflicts."""

import hashlib
import json
from functools import cache
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contract_registry import CONTRACT_SUPERSESSIONS

INPUT_FIELDS = (
    "dataset_id",
    "dataset_revision",
    "converter_version",
    "converter_revision",
    "measured_tasks",
    "task_specs_sha256",
)


class ContractSupersession(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str
    previous_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_uri: str
    selected_uri: str
    input: dict[str, str | int]
    summary_changes: dict[str, tuple[int, int]]
    reason: Literal["single-turn-rule-correction"]

    @model_validator(mode="after")
    def exact_scope(self):
        if set(self.input) != set(INPUT_FIELDS) or set(self.summary_changes) != {"single_turn_count"}:
            raise ValueError("Derived-rule correction must bind all input fields and only single_turn_count")
        if self.previous_receipt_sha256 == self.selected_receipt_sha256:
            raise ValueError("A supersession must identify two distinct receipts")
        return self


def receipt_identity(receipt: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def contract_registry_key() -> str:
    return "contract-selection:" + json.loads(CONTRACT_SUPERSESSIONS)["version"]


@cache
def contract_supersessions() -> tuple[ContractSupersession, ...]:
    payload = json.loads(CONTRACT_SUPERSESSIONS)
    if payload["schema_version"] != 1:
        raise ValueError("Unknown contract-supersession schema")
    rules = tuple(ContractSupersession.model_validate(row) for row in payload["supersessions"])
    identities = [(rule.source_id, rule.previous_receipt_sha256) for rule in rules]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate derived-rule supersession")
    return rules


def select_contract_receipts(source_id: str, matches: list[dict]) -> list[dict]:
    """Remove only exact, input-bound predecessor receipts; every other conflict fails."""
    indexed = {receipt_identity(receipt): receipt for receipt in matches}
    superseded = set()
    for rule in contract_supersessions():
        if rule.source_id != source_id:
            continue
        previous = indexed.get(rule.previous_receipt_sha256)
        selected = indexed.get(rule.selected_receipt_sha256)
        if previous is None or selected is None:
            continue
        if any(
            any(receipt["input"][field] != rule.input[field] for field in INPUT_FIELDS)
            for receipt in (previous, selected)
        ):
            raise ValueError("Superseded contract input differs from its authorization")
        if (previous["evidence_url"], selected["evidence_url"]) != (rule.previous_uri, rule.selected_uri):
            raise ValueError("Superseded contract evidence differs from its authorization")
        fields = set(previous["summary"]) | set(selected["summary"])
        changes = {
            field: (previous["summary"].get(field), selected["summary"].get(field))
            for field in fields
            if previous["summary"].get(field) != selected["summary"].get(field)
        }
        if changes != rule.summary_changes:
            raise ValueError("Superseded contract summary differs from its authorization")
        superseded.add(rule.previous_receipt_sha256)
    result = [receipt for receipt in matches if receipt_identity(receipt) not in superseded]
    if result and any(receipt["summary"] != result[0]["summary"] for receipt in result):
        raise ValueError(f"Conflicting derived contract summaries: {source_id}")
    return result
