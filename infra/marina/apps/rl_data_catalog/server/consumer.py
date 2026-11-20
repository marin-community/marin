# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the current consumer policy to retained catalog declarations."""

import json
from functools import cache
from typing import Any

from finestore.rl.curation import ActorCensusInput, ConsumerNeed, ConsumerReceipt, consumer_blockers

from .consumer_evidence import CONSUMER_INPUTS, CONSUMER_RUNTIME_COMMIT


@cache
def packaged_consumer_inputs() -> dict[str, dict[str, Any]]:
    """Read immutable scope and receipt metadata shipped with this app revision."""
    return json.loads(CONSUMER_INPUTS)


def consumer_projection(declaration: dict[str, Any]) -> dict[str, Any]:
    """Gate only bespoke needs; standard source declarations pass through unchanged."""
    if declaration["status"] not in {"Available", "Pending"}:
        return declaration
    current = packaged_consumer_inputs().get(declaration["id"], declaration)
    requirements = tuple(ConsumerNeed(need) for need in current.get("consumer_requirements") or ())
    if not requirements:
        return declaration
    raw = current.get("consumer_receipt")
    binding = declaration.get("actor_census_input")
    reasons = consumer_blockers(
        ConsumerReceipt.model_validate(raw) if raw is not None else None,
        declaration["id"],
        ActorCensusInput.model_validate(binding) if binding is not None else None,
        (declaration.get("within_source_dedup") or {}).get("unique_count", 0),
        requirements,
        runtime_commit=CONSUMER_RUNTIME_COMMIT,
    )
    effective = {
        **declaration,
        "consumer_requirements": [need.value for need in requirements],
        "consumer_receipt": raw,
    }
    if not reasons:
        return effective
    notes = declaration.get("notes", "")
    missing_reasons = [reason.value for reason in reasons if reason.value not in notes]
    return {
        **effective,
        "status": "Pending" if reasons and declaration["status"] == "Available" else declaration["status"],
        "consumer_blockers": [reason.value for reason in reasons],
        "notes": " ".join(
            filter(
                None,
                (
                    declaration.get("notes", ""),
                    ", ".join(missing_reasons),
                ),
            )
        ),
    }
