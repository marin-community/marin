# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive the preview package's oracle inventory and source counts from the census.

The packaged applet carries two measurement documents: the full-population oracle
inventory ledger and the per-source count evidence. Both describe the release's own
populations, so they are derived from the sealed census receipt of the same release,
never carried from an earlier release.
"""

import json
from dataclasses import dataclass
from typing import Any

from infra.marina.apps.rl_data_catalog.prepare_preview import oracle_inventory_payload
from infra.marina.apps.rl_data_catalog.server.source_counts import (
    ACTORLESS_POLICY_SHA256,
    NO_ORACLE_POLICY_SHA256,
    SourceCounts,
)
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

INVENTORY_KIND = "full_population_oracle_inventory"
INVENTORY_SCHEMA_VERSION = 2


def ruling_citations(prior_counts: dict) -> tuple[dict, dict]:
    """The no-oracle and actorless ruling references the prior count evidence cites.

    The rulings are release-independent documents; the census applies the actorless
    exemption to every static audit, so every derived row cites both. The prior
    release must cite exactly one version of each, at the digests the count model pins.
    """
    policies = {json.dumps(row["policy"], sort_keys=True) for row in prior_counts["sources"]}
    actorless = {
        json.dumps(row["actorless_policy"], sort_keys=True)
        for row in prior_counts["sources"]
        if row.get("actorless_policy") is not None
    }
    if len(policies) != 1 or len(actorless) != 1:
        raise ValueError(f"prior count evidence cites {len(policies)} no-oracle and {len(actorless)} actorless rulings")
    policy, actorless_policy = json.loads(policies.pop()), json.loads(actorless.pop())
    if (policy["sha256"], actorless_policy["sha256"]) != (NO_ORACLE_POLICY_SHA256, ACTORLESS_POLICY_SHA256):
        raise ValueError("prior count evidence cites rulings other than the ones the count model pins")
    return policy, actorless_policy


@dataclass(frozen=True)
class PackageInputDocuments:
    """The derived documents as their exact sealed bytes."""

    oracle_inventory: bytes
    source_counts: bytes
    census_sources: int
    counted_sources: int
    unmeasured_sources: tuple[str, ...]


def _document_bytes(document: dict) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def derive_package_inputs(
    census_proof: dict, expected_source_count: int, citations: tuple[dict, dict]
) -> PackageInputDocuments:
    """Derive both documents from one sealed census proof; refuse any gap.

    Every census row with a measured population becomes one inventory row and one
    count row. A row without a population must be a source the release does not
    serve as Available; an Available source without one, a half-measured row, a
    duplicate, or a census that does not cover the plan's sources fails closed.
    """
    policy, actorless_policy = citations
    results = census_proof["results"]
    if len(results) != expected_source_count or census_proof["sources"] != expected_source_count:
        raise ValueError(
            f"the census covers {len(results)} sources (proof says {census_proof['sources']}),"
            f" the plan expects {expected_source_count}"
        )
    seen: set[str] = set()
    ledger_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    unmeasured: list[str] = []
    for result in sorted(results, key=lambda row: canonical_source_id(row["source_id"])):
        source_id = canonical_source_id(result["source_id"])
        if source_id in seen:
            raise ValueError(f"the census carries {source_id} twice")
        seen.add(source_id)
        declaration = result["declaration"]
        census_input = declaration.get("actor_census_input")
        inventory = result.get("inventory")
        if census_input is None and inventory is None:
            if declaration["status"] == "Available":
                raise ValueError(f"Available source {source_id} has no measured census population")
            unmeasured.append(source_id)
            continue
        if census_input is None or inventory is None:
            raise ValueError(f"census row {source_id} carries only one of its actor input and inventory")
        reference = result["inventory_reference"]
        ledger_rows.append(
            {
                "source_id": source_id,
                "current_input": census_input,
                "inventory": inventory,
                "inventory_manifest_uri": reference["uri"],
                "inventory_manifest_sha256": reference["sha256"],
            }
        )
        count_rows.append(
            {
                "source_id": source_id,
                "input": census_input,
                "intended_use": declaration["intended_use"],
                "oracle_tasks": result["oracle_tasks"],
                "static_blocked_tasks": result["static_blocked_tasks"],
                "static_reason_counts": result["static_reason_counts"] or {},
                "policy": policy,
                "actorless_policy": actorless_policy,
                "oracle_inventory": {"uri": reference["uri"], "sha256": reference["sha256"]},
                "static_audit": result["static_audit"],
            }
        )
    ledger = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "kind": INVENTORY_KIND,
        "sources": len(ledger_rows),
        "retained_tasks": sum(row["current_input"]["measured_tasks"] for row in ledger_rows),
        "rows": ledger_rows,
    }
    counts = SourceCounts.model_validate({"version": 1, "sources": count_rows})
    inventory_raw = _document_bytes(ledger)
    # The packaging translation is the consumer: it must accept the derived ledger.
    oracle_inventory_payload(inventory_raw)
    return PackageInputDocuments(
        oracle_inventory=inventory_raw,
        source_counts=_document_bytes(counts.model_dump(mode="json")),
        census_sources=len(results),
        counted_sources=len(count_rows),
        unmeasured_sources=tuple(unmeasured),
    )
