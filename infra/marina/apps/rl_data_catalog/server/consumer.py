# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the current consumer policy to retained catalog declarations."""

import json
from functools import cache
from typing import Any

from finestore.rl.curation import (
    ActorCensusInput,
    ConsumerDecision,
    ConsumerNeed,
    ConsumerReceipt,
    ConsumerRuntimeCommit,
    consumer_blockers,
    consumer_decision_status,
)
from pydantic import TypeAdapter

from .consumer_evidence import CONSUMER_INPUTS, CONSUMER_RUNTIME_COMMIT

RUNTIME_PIN = TypeAdapter(ConsumerRuntimeCommit | None)


@cache
def packaged_consumer_inputs() -> dict[str, dict[str, Any]]:
    """Read immutable scope and receipt metadata shipped with this app revision."""
    return json.loads(CONSUMER_INPUTS)


def consumer_projection(declaration: dict[str, Any]) -> dict[str, Any]:
    """Show bespoke findings; only a current per-source user decision changes status."""
    current = packaged_consumer_inputs().get(declaration["id"], declaration)
    requirements = tuple(ConsumerNeed(need) for need in current.get("consumer_requirements") or ())
    if not requirements:
        return declaration
    runtime_pin = RUNTIME_PIN.validate_python(current.get("consumer_runtime_commit"))
    runtime_commit = runtime_pin or CONSUMER_RUNTIME_COMMIT
    raw = current.get("consumer_receipt")
    binding = declaration.get("actor_census_input")
    reasons = consumer_blockers(
        ConsumerReceipt.model_validate(raw) if raw is not None else None,
        declaration["id"],
        ActorCensusInput.model_validate(binding) if binding is not None else None,
        (declaration.get("within_source_dedup") or {}).get("unique_count", 0),
        requirements,
        runtime_commit=runtime_commit,
    )
    raw_decision = current.get("consumer_decision")
    decision = ConsumerDecision.model_validate(raw_decision) if raw_decision is not None else None
    approved = consumer_decision_status(
        decision,
        declaration["id"],
        ActorCensusInput.model_validate(binding) if binding is not None else None,
        reasons,
        runtime_commit=runtime_commit,
    )
    effective = {
        **declaration,
        "consumer_requirements": [need.value for need in requirements],
        "consumer_receipt": raw,
        "consumer_findings": [reason.value for reason in reasons],
        "consumer_decision": raw_decision if approved is not None else None,
    }
    effective.pop("consumer_runtime_commit", None)
    if runtime_pin is not None:
        effective["consumer_runtime_commit"] = runtime_pin
    if not reasons:
        return effective
    notes = declaration.get("notes", "")
    missing_reasons = [reason.value for reason in reasons if reason.value not in notes]
    return {
        **effective,
        "status": (
            approved
            if approved is not None and declaration["status"] in {"Available", "Pending"}
            else declaration["status"]
        ),
        "notes": " ".join(
            filter(
                None,
                (
                    declaration.get("notes", ""),
                    "Consumer advisory: " + ", ".join(missing_reasons) if missing_reasons else "",
                ),
            )
        ),
    }
