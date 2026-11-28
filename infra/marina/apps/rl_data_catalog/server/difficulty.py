# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project revision-bound difficulty evidence without promoting historical results."""

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from finestore.rl.difficulty import PROTOCOL_ID, DifficultyObservation, MeasurementStatus, Solver
from sqlalchemy import text
from sqlalchemy.engine import Connection

from .audit import wilson_interval
from .content_access import content_is_withheld, content_marker
from .source_identity import canonical_source_id, original_source_id


def difficulty_records(connection: Connection, source_id: str | None = None) -> dict[str, list[DifficultyObservation]]:
    records: dict[str, list[DifficultyObservation]] = {}
    query = "SELECT measurement #- '{sample,tasks}' FROM atlas_difficulty"
    parameters = {}
    if source_id is not None:
        query += " WHERE source_id IN (:source, :original)"
        parameters = {"source": canonical_source_id(source_id), "original": original_source_id(source_id)}
    for payload in connection.execute(text(query), parameters).scalars():
        record = DifficultyObservation.model_validate(payload)
        records.setdefault(canonical_source_id(record.source_id), []).append(record)
    return records


def matches_current_pin(record: DifficultyObservation, declaration: dict[str, Any]) -> bool:
    """Whether one observation covers the declaration's current release and converter.

    The sample clause applies only when the record carries a population sample:
    rows without one (historical summaries) match on the release and converter alone.
    """
    population = declaration.get("actor_census_input")
    return (
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


def split_observations(
    declaration: dict[str, Any], records: Iterable[DifficultyObservation]
) -> tuple[dict[Solver, DifficultyObservation], list[DifficultyObservation]]:
    """Split observations into the latest per-solver current-pin map and the historical remainder."""
    current: dict[Solver, DifficultyObservation] = {}
    historical = []
    for record in sorted(
        records, key=lambda item: (datetime.fromisoformat(item.created_at), item.measurement_id), reverse=True
    ):
        if not matches_current_pin(record, declaration) or record.solver in current:
            historical.append(record)
            continue
        current[record.solver] = record
    return current, historical


def current_observations(
    declaration: dict[str, Any], records: Iterable[DifficultyObservation]
) -> dict[Solver, DifficultyObservation]:
    """The latest observation per solver that covers this exact release and converter."""
    return split_observations(declaration, records)[0]


def raw_current_models(declaration: dict[str, Any], records: Iterable[DifficultyObservation]) -> list[dict[str, Any]]:
    """The un-redacted current-solver summaries, for classification that must read typed reasons."""
    current, _ = split_observations(declaration, records)
    return model_summaries(current)


def difficulty_summary(declaration: dict[str, Any], records: Iterable[DifficultyObservation]) -> dict[str, Any]:
    """Choose the latest observation per solver for this exact release and converter.

    A PRIVATE_ALWAYS source's served reasons are replaced by the private marker:
    its measurements stay counts and hashes, and a tampered or repurposed
    free-text reason can never carry its content.
    """
    current, historical = split_observations(declaration, records)
    models = model_summaries(current)
    historical = [record.model_dump(mode="json", exclude={"sample": {"tasks"}}) for record in historical]
    if content_is_withheld(declaration):
        models = [{**model, "reason": content_marker(declaration)["marker"]} for model in models]
        historical = [{**record, "reason": content_marker(declaration)["marker"]} for record in historical]
    return {
        "protocol_id": PROTOCOL_ID,
        "models": models,
        "historical": historical,
    }


def model_summaries(current: dict[Solver, DifficultyObservation]) -> list[dict[str, Any]]:
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
        if record.status == MeasurementStatus.MEASURED and record.n > 0 and record.wilson_95 is None:
            payload["wilson_95"] = difficulty_interval(record.solved, record.n)
        models.append(payload)
    return models


def difficulty_interval(solved: int, n: int) -> dict[str, float]:
    """The Wilson 95% interval for a measured difficulty cell."""
    low, high = wilson_interval(solved, n)
    return {"low": low, "high": high}


def difficulty_table_summary(declaration: dict[str, Any], records: Iterable[DifficultyObservation]) -> dict[str, Any]:
    """Keep current solver bars and their complete tooltip values; history stays in Evidence."""
    return table_difficulty(difficulty_summary(declaration, records))


def table_difficulty(summary: dict[str, Any]) -> dict[str, Any]:
    """The current solver values and provenance displayed by the table and dialog."""
    fields = {
        "solver",
        "status",
        "reason",
        "n",
        "solved",
        "solve_rate",
        "wilson_95",
        "protocol_id",
        "difficulty",
        "sealed_snapshot_id",
    }
    output = []
    for model in summary["models"]:
        row = {key: value for key, value in model.items() if key in fields}
        if model.get("model") is not None:
            row["model"] = {key: model["model"][key] for key in ["model", "revision"]}
        if model.get("sample") is not None:
            row["sample"] = {"pre_oracle_filter": model["sample"]["pre_oracle_filter"]}
        output.append(row)
    return {
        "protocol_id": summary["protocol_id"],
        "models": output,
        **({"sealed_snapshot": summary["sealed_snapshot"]} if "sealed_snapshot" in summary else {}),
    }
