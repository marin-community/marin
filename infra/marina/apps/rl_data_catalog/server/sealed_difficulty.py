# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project a release-bound difficulty seal without promoting stale or partial cells."""

from collections import defaultdict

from finestore.rl.difficulty import DifficultyGap, MeasurementStatus
from sqlalchemy import text
from sqlalchemy.engine import Connection

from .content_access import content_is_withheld, content_marker
from .json_digest import json_sha256
from .source_identity import canonical_source_id

_LEDGER_GAP_ALIASES = {
    "COVERAGE_SHORTFALL": DifficultyGap.AWAITING_SCORED_COVERAGE,
    "NO_MEASUREMENT_ROW": DifficultyGap.NO_QUALIFIED_MEASUREMENT_RECORDED,
    "TASK_SPEC_FACTORY_PENDING": DifficultyGap.DAYTONA_TASK_SPEC_FACTORY_PENDING,
    "SERVING_RELEASE_MISMATCH": DifficultyGap.PRODUCTION_HARBOR_TRIALS_PENDING,
    "SERVING_PROBE_MISMATCH": DifficultyGap.PRODUCTION_HARBOR_TRIALS_PENDING,
    "SOURCE_QUEUE_VALIDATION_FAILED": DifficultyGap.PRODUCTION_HARBOR_TRIALS_PENDING,
    "CONTROLLER_ACTION_CANCELLED": DifficultyGap.PRODUCTION_HARBOR_TRIALS_PENDING,
    "CONTROLLER_ACTION_FAILED": DifficultyGap.PRODUCTION_HARBOR_TRIALS_PENDING,
}


def sealed_difficulty_records(connection: Connection, load_id: str) -> dict[str, dict]:
    records = defaultdict(lambda: {"cells": []})
    rows = connection.execute(
        text(
            "SELECT c.source_id,c.cell,s.snapshot_id,s.manifest_uri,s.manifest_sha256 "
            "FROM atlas_difficulty_seal_loads l JOIN atlas_difficulty_seals s USING(snapshot_id) "
            "JOIN atlas_difficulty_seal_cells c USING(snapshot_id) WHERE l.load_id=:load_id"
        ),
        {"load_id": load_id},
    ).mappings()
    for row in rows:
        record = records[canonical_source_id(row["source_id"])]
        record.update(snapshot_id=row["snapshot_id"], uri=row["manifest_uri"], sha256=row["manifest_sha256"])
        record["cells"].append(row["cell"])
    return dict(records)


def sealed_difficulty_models(declaration: dict, models: list[dict], record: dict | None) -> list[dict]:
    """Overlay only cells measured against this declaration's exact population."""
    if record is None:
        return models
    census = declaration.get("actor_census_input") or {}
    revision = json_sha256(
        {
            "release": declaration.get("current_release"),
            "converter_version": (declaration.get("pipeline") or {}).get("version"),
            "converter_revision": census.get("converter_revision"),
            "task_specs_sha256": census.get("task_specs_sha256"),
        }
    )
    cells = {cell["solver"]: cell for cell in record["cells"] if cell["registry_revision"] == revision}
    output = []
    for model in models:
        cell = cells.get(model["solver"])
        if cell is None:
            output.append(model)
            continue
        gaps = []
        for gap in cell["gaps"]:
            reason = gap["reason"]
            mapped = DifficultyGap.__members__.get(reason) or _LEDGER_GAP_ALIASES.get(reason)
            if mapped is None:
                raise ValueError(f"Unmapped sealed ledger gap: {reason}")
            gaps.append(mapped.value)
        output.append(
            {
                "solver": model["solver"],
                "protocol_id": cell["protocol_id"],
                "status": MeasurementStatus.MEASURED.value if cell["measured"] else MeasurementStatus.NOT_MEASURED.value,
                "reason": "; ".join(sorted(set(gaps))) or None,
                "n": cell["scored_attempts"],
                "solved": cell["solved_attempts"],
                "solve_rate": cell["pass_rate"],
                "difficulty": cell["difficulty"],
                "wilson_95": None,
                "measurement_id": record["snapshot_id"] + ":" + model["solver"],
                "sealed_snapshot_id": record["snapshot_id"],
            }
        )
    return output


def sealed_difficulty_summary(summary: dict, declaration: dict, record: dict | None) -> dict:
    if record is None:
        return summary
    models = sealed_difficulty_models(declaration, summary["models"], record)
    if content_is_withheld(declaration):
        models = [
            {**model, "reason": content_marker(declaration)["marker"]} if model.get("reason") else model
            for model in models
        ]
    return {
        **summary,
        "models": models,
        "sealed_snapshot": {key: record[key] for key in ("snapshot_id", "uri", "sha256")},
    }
