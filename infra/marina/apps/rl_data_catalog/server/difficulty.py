# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project revision-bound difficulty evidence without promoting historical results."""

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from finestore.rl.difficulty import PROTOCOL_ID, DifficultyMeasurement, MeasurementStatus, Solver
from sqlalchemy import text
from sqlalchemy.engine import Connection

from .source_identity import canonical_source_id


def difficulty_records(connection: Connection) -> dict[str, list[DifficultyMeasurement]]:
    records: dict[str, list[DifficultyMeasurement]] = {}
    for payload in connection.execute(text("SELECT measurement FROM atlas_difficulty")).scalars():
        record = DifficultyMeasurement.model_validate(payload)
        records.setdefault(canonical_source_id(record.source_id), []).append(record)
    return records


def difficulty_summary(declaration: dict[str, Any], records: Iterable[DifficultyMeasurement]) -> dict[str, Any]:
    """Choose the latest observation per solver for this exact release and converter."""
    current: dict[Solver, DifficultyMeasurement] = {}
    historical = []
    population = declaration.get("actor_census_input")
    for record in sorted(
        records, key=lambda item: (datetime.fromisoformat(item.created_at), item.measurement_id), reverse=True
    ):
        matches = (
            record.protocol_id == PROTOCOL_ID
            and record.release.model_dump(mode="json") == declaration["current_release"]
            and record.converter_version == declaration["pipeline"]["version"]
            and (
                record.sample is None
                or (
                    population is not None
                    and (
                        record.sample.task_specs_sha256,
                        record.sample.population_count,
                        record.sample.converter_revision,
                    )
                    == (population["task_specs_sha256"], population["measured_tasks"], population["converter_revision"])
                )
            )
        )
        if not matches or record.solver in current:
            historical.append(record.model_dump(mode="json", exclude={"sample": {"tasks"}}))
            continue
        current[record.solver] = record
    models = []
    for solver in Solver:
        if solver not in current:
            models.append(
                {
                    "solver": solver.value,
                    "status": MeasurementStatus.NOT_MEASURED.value,
                    "reason": "No difficulty v3 measurement for this pinned release and converter.",
                    "n": 0,
                    "solved": 0,
                    "solve_rate": None,
                    "wilson_95": None,
                }
            )
            continue
        record = current[solver]
        payload = record.model_dump(mode="json", exclude={"sample": {"tasks"}})
        payload["solve_rate"] = record.solved / record.n if record.status == MeasurementStatus.MEASURED else None
        models.append(payload)
    return {"protocol_id": PROTOCOL_ID, "models": models, "historical": historical}
