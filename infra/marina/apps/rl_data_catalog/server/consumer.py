# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the current consumer policy to retained catalog declarations."""

from typing import Any

from finestore.rl.curation import ActorCensusInput, ConsumerReceipt, consumer_blockers


def consumer_projection(declaration: dict[str, Any]) -> dict[str, Any]:
    """Keep old Available rows pending until their current consumer check passes."""
    if declaration["status"] not in {"Available", "Pending"}:
        return declaration
    raw = declaration.get("consumer_receipt")
    binding = declaration.get("actor_census_input")
    reasons = consumer_blockers(
        ConsumerReceipt.model_validate(raw) if raw is not None else None,
        declaration["id"],
        ActorCensusInput.model_validate(binding) if binding is not None else None,
        (declaration.get("within_source_dedup") or {}).get("unique_count", 0),
    )
    if not reasons:
        return declaration
    notes = declaration.get("notes", "")
    missing_reasons = [reason.value for reason in reasons if reason.value not in notes]
    return {
        **declaration,
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
