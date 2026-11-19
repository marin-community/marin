# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manual human audit of reviewer judgments.

A signed-in Marina user draws a reproducible, seeded audit sample from one
source's stored reviews, records one judgment per task (agree/disagree plus a
corrected verdict and defects from the reviewer's own enums), and reads
reviewer-versus-human agreement with Wilson confidence intervals. Judgments are
advisory evidence; nothing here changes an admission or a quality label.
"""

import hashlib
import json
import math
from typing import Any

from fastapi import FastAPI, HTTPException
from finestore.rl.curation import Defect, Verdict
from pydantic import BaseModel, Field
from rigging.server_auth import VerifiedIdentity, get_verified_identity
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from .content_access import content_is_withheld, current_source_declaration
from .restrictions import (
    redact_text,
)
from .source_identity import canonical_source_id, original_source_id

MAX_AUDIT_NOTE_CHARACTERS = 500
MAX_AUDIT_SAMPLE_SIZE = 500
RANDOM_STRATEGY = "random"
STRATIFIED_STRATEGY = "stratified"
STRATEGIES = (RANDOM_STRATEGY, STRATIFIED_STRATEGY)
Z_95 = 1.96


class AuditJudgment(BaseModel):
    """One human judgment about one reviewed task."""

    task_id: str
    agree: bool
    corrected_verdict: Verdict | None = None
    corrected_defects: list[Defect] = []
    note: str = Field(default="", max_length=MAX_AUDIT_NOTE_CHARACTERS)


class AuditSampleRequest(BaseModel):
    source_id: str
    strategy: str
    seed: str = Field(min_length=1, max_length=128)
    sample_size: int = Field(ge=1, le=MAX_AUDIT_SAMPLE_SIZE)


def signed_in_user() -> str:
    """The caller's verified Marina identity, or an error for unauthenticated callers.

    Ambient loopback trust (the local stack's anonymous admin identity) also
    records judgments, attributed to that id; production IAP callers record
    theirs.
    """
    identity: VerifiedIdentity | None = get_verified_identity()
    if identity is None:
        raise HTTPException(401, "Sign in to Marina to record audit judgments")
    return identity.user_id


def stratum_key(verdict: str, defects: list[str]) -> str:
    """One stratum per distinct (verdict, defect set); defects order canonically."""
    return json.dumps([verdict, sorted(defects)], ensure_ascii=False)


def audit_order_key(seed: str, task_id: str) -> bytes:
    """Deterministic per-task sort key derived from the recorded seed."""
    return hashlib.sha256(f"{seed}:{task_id}".encode()).digest()


def draw_audit_sample(
    items: list[dict[str, Any]], *, strategy: str, seed: str, sample_size: int
) -> list[dict[str, Any]]:
    """Draw a reproducible audit sample from reviewed items.

    Every task gets a seeded hash ordering. The random strategy takes the first
    ``sample_size`` tasks; the stratified strategy allocates them across
    (verdict, defect set) strata proportionally, with at least one task per
    nonempty stratum when the budget allows, then takes each stratum's quota in
    the same seeded order. Selected items gain a ``stratum`` field.
    """
    ordered = sorted(items, key=lambda item: audit_order_key(seed, item["task_id"]))
    if strategy == RANDOM_STRATEGY:
        return [
            {**item, "stratum": stratum_key(item["verdict"], item.get("defects") or [])}
            for item in ordered[:sample_size]
        ]
    strata: dict[str, list[dict[str, Any]]] = {}
    for item in ordered:
        key = stratum_key(item["verdict"], item.get("defects") or [])
        strata.setdefault(key, []).append(item)
    if not strata:
        return []
    quotas = stratum_quotas([len(members) for members in strata.values()], sample_size)
    selected: list[dict[str, Any]] = []
    for (key, members), quota in zip(strata.items(), quotas, strict=True):
        for member in members[:quota]:
            selected.append({**member, "stratum": key})
    return selected


def stratum_quotas(stratum_sizes: list[int], sample_size: int) -> list[int]:
    """Largest-remainder proportional allocation, reserving one seat per stratum when affordable."""
    total = sum(stratum_sizes)
    if not stratum_sizes:
        return []
    if sample_size >= total:
        return list(stratum_sizes)
    exact = [sample_size * size / total for size in stratum_sizes]
    seats = [math.floor(value) for value in exact]
    remainders = [exact[index] - seats[index] for index in range(len(seats))]
    order = sorted(range(len(seats)), key=lambda index: (-remainders[index], index))
    for index in order[: sample_size - sum(seats)]:
        seats[index] += 1
    if sample_size >= len(seats):
        for index, seat in enumerate(seats):
            if seat == 0:
                donor = seats.index(max(seats))
                seats[donor] -= 1
                seats[index] = 1
    return seats


def wilson_interval(agreements: int, judged: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion; (0.0, 1.0) when nothing is judged."""
    if judged == 0:
        return (0.0, 1.0)
    share = agreements / judged
    denominator = 1 + z * z / judged
    center = (share + z * z / (2 * judged)) / denominator
    half = z * math.sqrt(share * (1 - share) / judged + z * z / (4 * judged * judged)) / denominator
    return (max(0.0, center - half), min(1.0, center + half))


def agreement_report(judgments: list[dict[str, Any]], reviewer: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Reviewer-versus-human agreement per verdict and per defect, with intervals.

    ``judgments`` rows carry ``task_id`` and ``agree``; ``reviewer`` maps task_id
    to the reviewer's stored verdict and defects. Unknown tasks are skipped.
    """
    judged = [row for row in judgments if row["task_id"] in reviewer]
    return {
        "overall": _agreement(judged),
        "per_verdict": {
            verdict.value: _agreement([row for row in judged if reviewer[row["task_id"]]["verdict"] == verdict.value])
            for verdict in Verdict
        },
        "per_defect": {
            defect.value: _agreement(
                [row for row in judged if defect.value in (reviewer[row["task_id"]]["defects"] or [])]
            )
            for defect in Defect
        },
    }


def _agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    agreements = sum(1 for row in rows if row["agree"])
    low, high = wilson_interval(agreements, len(rows))
    return {
        "judged": len(rows),
        "agree": agreements,
        "agreement": agreements / len(rows) if rows else None,
        "interval": [low, high],
    }


def assessment_header(connection: Connection, source_id: str) -> dict[str, Any] | None:
    """The source's selected assessment header, or None when it has none."""
    row = (
        connection.execute(
            text(
                """
            SELECT s.load_id, COALESCE(r.pipeline_key,s.assessment_key) AS pipeline_key FROM atlas_sources s
            JOIN atlas_current c USING (load_id)
            LEFT JOIN atlas_review_current r ON r.load_id=s.load_id AND r.source_id=s.id
            WHERE c.name = 'latest' AND s.id IN (:source_id, :original_source_id)
                AND COALESCE(r.pipeline_key,s.assessment_key) IS NOT NULL
        """
            ),
            {"source_id": source_id, "original_source_id": original_source_id(source_id)},
        )
        .mappings()
        .one_or_none()
    )
    return dict(row) if row is not None else None


def sampled_items(connection: Connection, load_id: str, pipeline_key: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        text(
            """
        SELECT task_id, verdict, defects FROM atlas_items
        WHERE load_id = :load_id AND pipeline_key = :pipeline_key AND status = 'reviewed'
        ORDER BY task_id
    """
        ),
        {"load_id": load_id, "pipeline_key": pipeline_key},
    ).mappings()
    return [
        {"task_id": row["task_id"], "verdict": row["verdict"], "defects": list(row["defects"] or [])} for row in rows
    ]


def sample_id_for(request: AuditSampleRequest, pipeline_key: str, load_id: str) -> str:
    identity = {
        "source": canonical_source_id(request.source_id),
        "pipeline_key": pipeline_key,
        "load_id": load_id,
        "strategy": request.strategy,
        "seed": request.seed,
        "sample_size": request.sample_size,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def register_audit_routes(api: FastAPI, engine: Engine) -> None:
    @api.post("/audit/samples")
    def create_sample(request: AuditSampleRequest) -> dict[str, Any]:
        user = signed_in_user()
        if request.strategy not in STRATEGIES:
            raise HTTPException(422, f"strategy must be one of {STRATEGIES}")
        with engine.begin() as connection:
            header = assessment_header(connection, request.source_id)
            if header is None:
                raise HTTPException(404, "Source has no stored review assessment")
            if content_is_withheld(current_source_declaration(connection, request.source_id)):
                raise HTTPException(
                    403,
                    "Auditing a PRIVATE_ALWAYS source would quote its content to reviewers; "
                    "its reviews stay marker-only",
                )
            sample_id = sample_id_for(request, header["pipeline_key"], header["load_id"])
            existing = connection.execute(
                text("SELECT 1 FROM atlas_audit_samples WHERE sample_id = :sample_id"),
                {"sample_id": sample_id},
            ).scalar()
            if existing:
                return {"sample_id": sample_id, "existing": True}
            items = sampled_items(connection, header["load_id"], header["pipeline_key"])
            if not items:
                raise HTTPException(409, "Source has no reviewed tasks to audit")
            sample = draw_audit_sample(
                [dict(item) for item in items],
                strategy=request.strategy,
                seed=request.seed,
                sample_size=request.sample_size,
            )
            strata: dict[str, int] = {}
            for item in sample:
                strata[item["stratum"]] = strata.get(item["stratum"], 0) + 1
            connection.execute(
                text(
                    """
                INSERT INTO atlas_audit_samples
                    (sample_id, source_id, pipeline_key, load_id, strategy, seed, sample_size, strata, created_by)
                VALUES (:sample_id, :source_id, :pipeline_key, :load_id, :strategy, :seed,
                    :sample_size, :strata, :created_by)
            """
                ),
                {
                    "sample_id": sample_id,
                    "source_id": canonical_source_id(request.source_id),
                    "pipeline_key": header["pipeline_key"],
                    "load_id": header["load_id"],
                    "strategy": request.strategy,
                    "seed": request.seed,
                    "sample_size": request.sample_size,
                    "strata": json.dumps(strata, ensure_ascii=False),
                    "created_by": user,
                },
            )
            for item in sample:
                connection.execute(
                    text(
                        """
                    INSERT INTO atlas_audit_sample_tasks (sample_id, task_id, verdict, defects)
                    VALUES (:sample_id, :task_id, :verdict, :defects)
                """
                    ),
                    {
                        "sample_id": sample_id,
                        "task_id": item["task_id"],
                        "verdict": item["verdict"],
                        "defects": json.dumps(sorted(item["defects"])),
                    },
                )
        return {"sample_id": sample_id, "existing": False}

    @api.get("/audit/samples")
    def list_samples(source_id: str) -> dict[str, Any]:
        user = signed_in_user()
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                SELECT s.sample_id, s.strategy, s.seed, s.sample_size, s.strata, s.created_by, s.created_at,
                    s.source_id, s.pipeline_key, s.load_id,
                    (SELECT COUNT(*) FROM atlas_audit_sample_tasks t WHERE t.sample_id = s.sample_id) AS tasks,
                    (SELECT COUNT(*) FROM atlas_audit_judgments j WHERE j.sample_id = s.sample_id
                        AND j.user_id = :user) AS judged_by_me
                FROM atlas_audit_samples s
                WHERE s.source_id IN (:source_id, :original_source_id) ORDER BY s.created_at DESC
            """
                ),
                {
                    "source_id": canonical_source_id(source_id),
                    "original_source_id": original_source_id(source_id),
                    "user": user,
                },
            ).mappings()
            samples = [dict(row) for row in rows]
        return {"samples": samples}

    @api.get("/audit/samples/{sample_id}/tasks")
    def sample_tasks(sample_id: str) -> dict[str, Any]:
        user = signed_in_user()
        with engine.connect() as connection:
            sample = _sample_row(connection, sample_id)
            withheld = content_is_withheld(current_source_declaration(connection, sample["source_id"]))
            tasks = connection.execute(
                text(
                    """
                SELECT t.task_id, t.verdict, t.defects,
                    j.agree, j.corrected_verdict, j.corrected_defects, j.note, j.user_id,
                    j.created_at, j.updated_at,
                    i.cohort, i.explanation, i.evidence_ids, i.confidence, i.reference_status,
                    i.defects AS reviewer_defects
                FROM atlas_audit_sample_tasks t
                LEFT JOIN atlas_audit_judgments j ON j.sample_id = t.sample_id AND j.task_id = t.task_id
                    AND j.user_id = :user
                JOIN atlas_items i ON i.load_id = :load_id AND i.pipeline_key = :pipeline_key
                    AND i.task_id = t.task_id
                WHERE t.sample_id = :sample_id ORDER BY t.task_id
            """
                ),
                {
                    "sample_id": sample_id,
                    "load_id": sample["load_id"],
                    "pipeline_key": sample["pipeline_key"],
                    "user": user,
                },
            ).mappings()
        return {"sample": sample, "tasks": [_task_view(row, user, withheld=withheld) for row in tasks]}

    @api.put("/audit/samples/{sample_id}/judgments")
    def record_judgment(sample_id: str, judgment: AuditJudgment) -> dict[str, Any]:
        user = signed_in_user()
        if not judgment.agree and judgment.corrected_verdict is None:
            raise HTTPException(422, "A disagreement needs a corrected verdict")
        with engine.begin() as connection:
            _sample_row(connection, sample_id)
            in_sample = connection.execute(
                text("SELECT 1 FROM atlas_audit_sample_tasks WHERE sample_id = :sample_id AND task_id = :task_id"),
                {"sample_id": sample_id, "task_id": judgment.task_id},
            ).scalar()
            if not in_sample:
                raise HTTPException(404, "Task is not part of this audit sample")
            connection.execute(
                text(
                    """
                INSERT INTO atlas_audit_judgments
                    (sample_id, task_id, user_id, agree, corrected_verdict, corrected_defects, note)
                VALUES (:sample_id, :task_id, :user_id, :agree, :corrected_verdict, :corrected_defects, :note)
                ON CONFLICT (sample_id, task_id, user_id) DO UPDATE SET
                    agree = EXCLUDED.agree, corrected_verdict = EXCLUDED.corrected_verdict,
                    corrected_defects = EXCLUDED.corrected_defects, note = EXCLUDED.note,
                    updated_at = NOW()
            """
                ),
                {
                    "sample_id": sample_id,
                    "task_id": judgment.task_id,
                    "user_id": user,
                    "agree": judgment.agree,
                    "corrected_verdict": judgment.corrected_verdict,
                    "corrected_defects": json.dumps(sorted(judgment.corrected_defects)),
                    "note": judgment.note or None,
                },
            )
        return {"sample_id": sample_id, "task_id": judgment.task_id, "user_id": user}

    @api.get("/audit/samples/{sample_id}/agreement")
    def sample_agreement(sample_id: str) -> dict[str, Any]:
        with engine.begin() as connection:
            _sample_row(connection, sample_id)
            judgments = [
                dict(row)
                for row in connection.execute(
                    text("SELECT task_id, agree FROM atlas_audit_judgments WHERE sample_id = :sample_id"),
                    {"sample_id": sample_id},
                ).mappings()
            ]
            reviewer = {
                row["task_id"]: {"verdict": row["verdict"], "defects": row["defects"] or []}
                for row in connection.execute(
                    text("SELECT task_id, verdict, defects FROM atlas_audit_sample_tasks WHERE sample_id = :sample_id"),
                    {"sample_id": sample_id},
                ).mappings()
            }
        return {"sample_id": sample_id, **agreement_report(judgments, reviewer)}


def _sample_row(connection: Connection, sample_id: str) -> dict[str, Any]:
    row = (
        connection.execute(
            text(
                "SELECT sample_id, source_id, pipeline_key, load_id, strategy, seed, sample_size, "
                "strata, created_by, created_at FROM atlas_audit_samples WHERE sample_id = :sample_id"
            ),
            {"sample_id": sample_id},
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise HTTPException(404, "Audit sample not found")
    return dict(row)


def _task_view(row: dict[str, Any], user: str, withheld: bool = False) -> dict[str, Any]:
    view = {key: row[key] for key in ("task_id", "verdict", "cohort", "explanation", "evidence_ids")}
    if withheld:
        # A private source's reviewer text quotes its content; the marker replaces it.
        view["explanation"] = redact_text(view["explanation"])
    view["reviewer_defects"] = row["defects"] or []
    view["reviewer_confidence"] = row["confidence"]
    view["reviewer_reference_status"] = row["reference_status"]
    if row["user_id"] is None:
        view["judgment"] = None
    else:
        note = redact_text(row["note"]) if withheld else row["note"]
        view["judgment"] = {
            "agree": row["agree"],
            "corrected_verdict": row["corrected_verdict"],
            "corrected_defects": row["corrected_defects"] or [],
            "note": note,
            "mine": row["user_id"] == user,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        }
    return view
