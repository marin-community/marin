# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Browse staged difficulty attempts without runtime object-store access."""

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from finestore.rl.difficulty import DifficultyObservation
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from .content_access import content_is_withheld, content_marker
from .difficulty import matches_current_pin
from .evidence_content import unpack_content
from .source_identity import canonical_source_id, original_source_id


def register_rollout_routes(api: FastAPI, engine: Engine) -> None:
    def measurement(connection: Connection, source_id: str, measurement_id: str) -> tuple[dict, dict]:
        declaration = connection.execute(
            text(
                "SELECT s.declaration FROM atlas_sources s JOIN atlas_current c USING(load_id) "
                "WHERE c.name='latest' AND s.id IN (:source,:original)"
            ),
            {"source": canonical_source_id(source_id), "original": original_source_id(source_id)},
        ).scalar_one_or_none()
        if declaration is None:
            raise HTTPException(404, "Source not found")
        source = {"declaration": declaration}
        if content_is_withheld(declaration):
            return source, {}
        payload = connection.execute(
            text(
                "SELECT measurement FROM atlas_difficulty WHERE measurement_id=:id "
                "AND source_id IN (:source,:original)"
            ),
            {"id": measurement_id, "source": canonical_source_id(source_id), "original": original_source_id(source_id)},
        ).scalar_one_or_none()
        if payload is None:
            raise HTTPException(404, "Difficulty measurement not found for this source")
        return source, payload

    @api.get("/difficulty-attempts")
    def attempts(
        source_id: str, measurement_id: str, limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)
    ) -> dict[str, Any]:
        with engine.connect() as connection:
            source, payload = measurement(connection, source_id, measurement_id)
            if content_is_withheld(source["declaration"]):
                return content_marker(source["declaration"])
            observation = DifficultyObservation.model_validate(
                {
                    **payload,
                    "sample": (
                        {key: value for key, value in payload["sample"].items() if key != "tasks"}
                        if payload.get("sample")
                        else None
                    ),
                }
            )
            info = {
                "measurement": observation.model_dump(mode="json"),
                "current_pin": matches_current_pin(observation, source["declaration"]),
                "source": source["declaration"]["display_name"],
            }
            metadata = connection.execute(
                text("SELECT metadata FROM atlas_rollout_sets WHERE measurement_id=:id"), {"id": measurement_id}
            ).scalar_one_or_none()
            if metadata is None:
                return {**info, "status": "not_loaded", "items": [], "total": 0, "limit": limit, "offset": offset}
            items = (
                connection.execute(
                    text(
                        "SELECT summary FROM atlas_rollout_attempts WHERE measurement_id=:id "
                        "ORDER BY task_id,attempt_id LIMIT :limit OFFSET :offset"
                    ),
                    {"id": measurement_id, "limit": limit, "offset": offset},
                )
                .scalars()
                .all()
            )
            return {
                **info,
                "status": "loaded",
                "metadata": metadata,
                "items": items,
                "total": metadata["attempt_count"],
                "limit": limit,
                "offset": offset,
            }

    @api.get("/difficulty-rollout")
    def rollout(source_id: str, measurement_id: str, attempt_id: str) -> dict[str, Any]:
        with engine.connect() as connection:
            source, _ = measurement(connection, source_id, measurement_id)
            if content_is_withheld(source["declaration"]):
                return content_marker(source["declaration"])
            row = (
                connection.execute(
                    text(
                        "SELECT summary,detail FROM atlas_rollout_attempts "
                        "WHERE measurement_id=:measurement AND attempt_id=:attempt"
                    ),
                    {"measurement": measurement_id, "attempt": attempt_id},
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise HTTPException(404, "Attempt evidence is not loaded")
            return {"summary": row["summary"], "detail": unpack_content(row["detail"])}
