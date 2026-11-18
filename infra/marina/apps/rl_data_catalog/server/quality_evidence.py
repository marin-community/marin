# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explain the displayed QA level using recorded review inputs, never campaign guesses."""

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from .cohort_summaries import COHORT_SUMMARY_VERSION
from .quality import Quality
from .source_identity import canonical_source_id, original_source_id


class FullReviewReason(StrEnum):
    NO_RECORDED_FULL_REVIEW = "no_recorded_full_review"
    OTHER_INPUT = "full_review_other_input"
    INCOMPLETE = "full_review_incomplete"
    NOT_FINALIZED = "full_review_not_finalized"
    INPUT_UNVERIFIED = "full_review_input_unverified"


@dataclass(frozen=True)
class ReviewInput:
    dataset_id: str | None
    dataset_revision: str | None
    converter_version: str | None
    task_specs_sha256: str | None
    retained_tasks: int | None

    @property
    def complete(self) -> bool:
        return all(value is not None and value != "" for value in asdict(self).values())


def declaration_input(declaration: dict[str, Any]) -> ReviewInput:
    actor = declaration.get("actor_census_input") or {}
    return ReviewInput(
        declaration.get("dataset_id"),
        declaration.get("dataset_revision"),
        (declaration.get("pipeline") or {}).get("version"),
        actor.get("task_specs_sha256"),
        actor.get("measured_tasks"),
    )


def full_review_records(connection: Connection, source_id: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Read only retained FULL receipt metadata and precomputed counts, without review content."""
    rows = connection.execute(
        text(
            """
        SELECT q.source_id,q.receipt,s.summary FROM atlas_review_current q
        JOIN atlas_assessments a USING (load_id,pipeline_key)
        JOIN atlas_loads l USING (load_id)
        LEFT JOIN atlas_cohort_summaries s ON s.load_id=q.load_id AND s.pipeline_key=q.pipeline_key
            AND s.cohort='admitted' AND s.version=:version
        WHERE a.scope='full' AND (CAST(:source AS TEXT) IS NULL OR q.source_id IN (:source,:original))
        ORDER BY l.loaded_at DESC,q.load_id,q.source_id
    """
        ),
        {
            "version": COHORT_SUMMARY_VERSION,
            "source": source_id,
            "original": original_source_id(source_id) if source_id else None,
        },
    ).mappings()
    records: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        records.setdefault(canonical_source_id(row["source_id"]), []).append(dict(row))
    return records


def quality_explanation(
    declaration: dict[str, Any],
    level: Quality,
    full_reviews: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return factual reasons below FULL; missing input identity cannot prove a re-pin."""
    if level is Quality.FULL:
        return None
    qc = declaration.get("independent_qc") or {}
    result = declaration.get("quality_result") or {}
    sample = None
    if level is Quality.SAMPLE:
        sample = {
            "reviewed": result.get("reviewed", qc.get("reviewed")),
            "selected": result.get("sample_count"),
            "population": (declaration.get("within_source_dedup") or {}).get("unique_count"),
            "receipt": qc.get("evidence_url") or (declaration.get("oversight_evidence") or {}).get("receipt_uri"),
        }
    explanation: dict[str, Any] = {"code": FullReviewReason.NO_RECORDED_FULL_REVIEW, "sample": sample, "full": None}
    current = declaration_input(declaration)
    candidates = []
    binding = declaration.get("full_content_result_binding")
    if binding and canonical_source_id(binding["source_id"]) == canonical_source_id(declaration["id"]):
        bound_input = binding["input"]
        full_reviews = [
            {
                "receipt": {
                    "artifact": binding["review_receipt"],
                    "panel": {
                        **bound_input,
                        "retained_tasks": bound_input.get("measured_tasks"),
                    },
                },
                "summary": {"reviewed": binding["result"]["reviewed"]},
            },
            *full_reviews,
        ]
    for row in full_reviews:
        receipt = row["receipt"]
        panel = receipt["panel"]
        recorded = ReviewInput(**{key: panel.get(key) for key in asdict(current)})
        summary = row.get("summary") or {}
        evidence = {
            "input": asdict(recorded),
            "receipt": receipt["artifact"],
            "reviewed": summary.get("reviewed"),
            "population": recorded.retained_tasks,
        }
        if not current.complete or not recorded.complete:
            candidates.append((FullReviewReason.INPUT_UNVERIFIED, evidence))
        elif current != recorded:
            evidence["different_fields"] = [
                key for key, value in asdict(current).items() if value != asdict(recorded)[key]
            ]
            candidates.append((FullReviewReason.OTHER_INPUT, evidence))
        else:
            reviewed = evidence["reviewed"]
            code = (
                FullReviewReason.INCOMPLETE
                if reviewed is not None and recorded.retained_tasks is not None and reviewed < recorded.retained_tasks
                else FullReviewReason.NOT_FINALIZED
            )
            return {**explanation, "code": code, "full": evidence}
    if candidates:
        code, evidence = candidates[0]
        return {**explanation, "code": code, "full": evidence}
    return explanation
