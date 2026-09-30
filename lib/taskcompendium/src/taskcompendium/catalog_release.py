# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct reviewed TaskTrove demonstration tasks from pinned catalog rows."""

import json
from collections.abc import Iterable
from typing import Any

from taskcompendium.mixed_release import AcceptedTaskRecord, SourceProof
from taskcompendium.models import SCHEMA_VERSION, TaskSpec

CATALOG_SCHEMA_VERSION = "0.13"
EMPTY_FINAL_TOOLS = {"functions": [], "tool_choice": None, "parallel_tool_calls": None}
LEDGER_CATALOG_FIELDS = (
    "source",
    "path",
    "route",
    "mode",
    "family",
    "converter",
    "template_id",
    "tags",
    "archive_sha256",
)
PROJECTION_ONLY_FIELDS = {"record_version", "submission_instruction", "source_category"}


def upgrade_catalog_task(payload: dict[str, Any]) -> TaskSpec:
    """Convert the explicit 0.13 empty terminal-tool contract to the current schema."""
    if payload["schema_version"] != CATALOG_SCHEMA_VERSION:
        raise ValueError("Expected a schema 0.13 catalog specification")
    if payload["final_tools"] != EMPTY_FINAL_TOOLS:
        raise ValueError("Catalog conversion requires empty final tools and no request policy")
    converted = {**payload, "schema_version": SCHEMA_VERSION, "final_tools": []}
    task = TaskSpec.model_validate(converted)
    if task.verifier.model_dump(mode="json") != payload["verifier"]:
        raise ValueError("Catalog conversion changed the source verifier")
    return task


def reconstruct_catalog_records(
    projections: Iterable[dict[str, Any]],
    catalog_rows: Iterable[dict[str, Any]],
    ledger_rows: Iterable[dict[str, Any]],
) -> tuple[AcceptedTaskRecord, ...]:
    """Require an exact accepted-ID join and field equality before schema conversion."""
    accepted: dict[str, dict[str, Any]] = {}
    for projection in projections:
        identity = projection["task"]["id"]
        if identity in accepted:
            raise ValueError("Duplicate accepted projection ID")
        accepted[identity] = projection
    catalog: dict[str, dict[str, Any]] = {}
    for row in catalog_rows:
        identity = row["id"]
        if identity not in accepted:
            continue
        if identity in catalog:
            raise ValueError("Duplicate selected catalog ID")
        catalog[identity] = row
    ledger: dict[str, dict[str, Any]] = {}
    for row in ledger_rows:
        identity = row["imported_id"]
        if identity not in accepted:
            continue
        if identity in ledger:
            raise ValueError("Duplicate selected ledger ID")
        if row["disposition"] != "imported" or row["input_split"] != "tasks":
            raise ValueError("Accepted task has no imported tasks-split ledger record")
        ledger[identity] = row
    if set(accepted) != set(catalog) or set(accepted) != set(ledger):
        raise ValueError("Accepted IDs do not exactly join the catalog and ledger")
    result = []
    for identity, projection in accepted.items():
        source = catalog[identity]
        intake = ledger[identity]
        if any(source[field] != intake[field] for field in LEDGER_CATALOG_FIELDS):
            raise ValueError("Catalog source identity differs from ingestion ledger")
        payload = json.loads(source["specification_json"])
        projected = projection["task"]
        proof = SourceProof.model_validate(projection["source_proof"])
        if payload["id"] != identity or payload["tags"] != source["tags"]:
            raise ValueError("Catalog task identity or ordered tags differ from catalog metadata")
        if any(value != payload[field] for field, value in projected.items() if field not in PROJECTION_ONLY_FIELDS):
            raise ValueError("Accepted projection differs from complete catalog task")
        if (
            payload["source"]["row"] != proof.source_row
            or proof.source_row != f"{source['source']}:{source['path']}"
            or proof.archive_path != source["path"]
            or proof.archive_sha256 != source["archive_sha256"]
            or proof.input_file != intake["input_file"]
            or proof.input_object_pin != intake["input_object_pin"]
            or payload["source"]["revision"] != intake["input_object_pin"]
        ):
            raise ValueError("Accepted source proof differs from catalog or ledger")
        result.append(
            AcceptedTaskRecord(
                task=upgrade_catalog_task(payload),
                source_category=projected["source_category"],
                source_proof=proof,
            )
        )
    return tuple(result)
