# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only release evidence and task-population accounting."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from taskcompendium.source_fingerprints import DuplicateReview

from .server.app import canonical_duplicate_group, canonical_duplicate_review
from .server.derived_contract import contract_projection
from .server.duplicate_folding import SourceStanding, fold_groups, partial_overlap_notes
from .server.filtration import filtration
from .server.identity_measurements import IDENTITY_REVIEWS_SQL
from .server.source_counts import SourceCounts

SqlQuery = Callable[[str, dict[str, Any]], dict[str, Any]]


def json_sha256(value: Any) -> str:
    """Hash JSON independently of object key order and whitespace."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class TaskTotals:
    population: int = 0
    admitted: int = 0
    train: int = 0
    evaluation: int = 0
    pending: int = 0
    excluded: int = 0
    unmeasured: int = 0


def task_totals(rows: list[dict[str, Any]]) -> TaskTotals:
    """Require a conserved task partition, including the admitted Use split."""
    values = asdict(TaskTotals())
    for row in rows:
        counts = row["filtration"]
        population = counts["population_count"]
        admitted, excluded = counts["admitted_count"], counts["excluded_count"]
        if not isinstance(population, int) or population < 0:
            raise ValueError(f"{row['id']}: missing or invalid task population")
        values["population"] += population
        if admitted is not None:
            if admitted < 0 or excluded is None or excluded < 0 or admitted + excluded != population:
                raise ValueError(f"{row['id']}: admission and exclusions do not partition the population")
            use = {"train": "train", "eval": "evaluation"}.get(row["intended_use"])
            if use is None:
                raise ValueError(f"{row['id']}: admitted tasks have no TRAIN/EVAL Use")
            values["admitted"] += admitted
            values[use] += admitted
            values["excluded"] += excluded
        else:
            values["pending" if counts["state"] == "pending_oracle" else "unmeasured"] += population
    return TaskTotals(**values)


def admission_text(totals: TaskTotals) -> str:
    """Format the numeric header contract from independently checked populations."""
    text = (
        f"{totals.admitted:,} tasks admitted (TRAIN {totals.train:,} · EVAL {totals.evaluation:,})"
        f" · {totals.pending:,} tasks pending oracle filtration · {totals.excluded:,} tasks excluded"
    )
    if totals.unmeasured:
        text += f" · {totals.unmeasured:,} tasks with admission unmeasured"
    return text


def projected_sources(data: dict[str, Any], counts: SourceCounts) -> dict[str, Any]:
    """Bind count expectations to catalog inputs, never to a previous release's totals."""
    observations = {row.source_id: row for row in counts.sources}
    rows = [{**row, "filtration": filtration(row, observations.get(row["id"]))} for row in data["sources"]]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("The catalog contains duplicate source IDs")
    task_totals(rows)
    return {**data, "sources": rows}


def projected_contract_inventory(data: dict[str, Any], query: SqlQuery) -> dict[str, Any]:
    receipts = query("SELECT source_id,receipt FROM atlas_derived_contracts ORDER BY receipt_id", {})["rows"]
    return {
        **data,
        "sources": [
            contract_projection(row, [record["receipt"] for record in receipts if record["source_id"] == row["id"]])
            for row in data["sources"]
        ],
    }


DUPLICATE_REVIEW_BATCH = 100


def projected_duplicate_inventory(data: dict[str, Any], query: SqlQuery, load_id: str) -> dict[str, Any]:
    """Recompute display groups from frozen reviews and current displayed standings."""
    reviews = []
    offset = 0
    while True:
        rows = query(
            "SELECT review FROM (" + IDENTITY_REVIEWS_SQL + ") selected "
            "ORDER BY review->'similarity'->>'left',review->'similarity'->>'right' LIMIT :limit OFFSET :offset",
            {"load_id": load_id, "limit": DUPLICATE_REVIEW_BATCH, "offset": offset},
        )["rows"]
        reviews.extend(DuplicateReview.model_validate(canonical_duplicate_review(row["review"])) for row in rows)
        if len(rows) < DUPLICATE_REVIEW_BATCH:
            break
        offset += len(rows)
    standings = {
        row["id"]: SourceStanding(row["within_source_dedup"]["unique_count"], row["quality"]) for row in data["sources"]
    }
    reviews = [
        review for review in reviews if review.similarity.left in standings and review.similarity.right in standings
    ]
    groups = fold_groups(reviews, standings)
    by_source = {member.source_id: group.model_dump(mode="json") for group in groups for member in group.members}
    notes = partial_overlap_notes(reviews, groups)
    return {
        **data,
        "sources": [
            {**row, "duplicate_group": by_source.get(row["id"]), "partial_overlaps": notes.get(row["id"], [])}
            for row in data["sources"]
        ],
    }


def stored_duplicate_groups(query: SqlQuery, load_id: str) -> list[dict[str, Any]]:
    rows = query(
        "SELECT DISTINCT duplicate_group FROM atlas_sources " "WHERE load_id=:load_id AND duplicate_group IS NOT NULL",
        {"load_id": load_id},
    )["rows"]
    return sorted(
        (canonical_duplicate_group(row["duplicate_group"]) for row in rows), key=lambda group: group["group_id"]
    )


def catalog_state(query: SqlQuery) -> dict[str, Any]:
    """Snapshot selected pointers and every retained load's assessment/packet contents."""
    current = query(
        "SELECT c.load_id,l.snapshot_id FROM atlas_current c JOIN atlas_loads l USING(load_id) " "WHERE c.name='latest'",
        {},
    )["rows"]
    if len(current) != 1:
        raise ValueError("Atlas has no unique selected catalog")
    archives = query(
        "SELECT l.load_id, "
        "(SELECT COUNT(*) FROM atlas_sources s WHERE s.load_id=l.load_id) AS sources, "
        "(SELECT md5(COALESCE(string_agg(md5(s.declaration::text),'' ORDER BY s.id),'')) "
        "FROM atlas_sources s WHERE s.load_id=l.load_id) AS source_checksum, "
        "(SELECT md5(COALESCE(string_agg(md5(jsonb_build_object('fingerprint',s.source_fingerprint,"
        "'group',s.duplicate_group)::text),'' ORDER BY s.id),'')) "
        "FROM atlas_sources s WHERE s.load_id=l.load_id) AS identity_source_checksum, "
        "(SELECT md5(COALESCE(string_agg(md5(to_jsonb(i)::text),'' ORDER BY i.survey_id),'')) "
        "FROM atlas_identity_surveys i WHERE i.load_id=l.load_id) AS identity_survey_checksum, "
        "(SELECT md5(COALESCE(string_agg(md5(r.review::text),'' ORDER BY r.left_source,r.right_source),'')) "
        "FROM atlas_duplicate_reviews r WHERE r.load_id=l.load_id) AS duplicate_review_checksum, "
        "(SELECT COUNT(*) FROM atlas_assessments a WHERE a.load_id=l.load_id) AS assessments, "
        "(SELECT md5(COALESCE(string_agg(md5(to_jsonb(a)::text),'' ORDER BY a.pipeline_key),'')) "
        "FROM atlas_assessments a WHERE a.load_id=l.load_id) AS assessment_checksum, "
        "(SELECT COUNT(*) FROM atlas_packets p WHERE p.load_id=l.load_id) AS packets, "
        "(SELECT md5(COALESCE(string_agg(md5(p.packet),'' ORDER BY p.pipeline_key,p.task_id),'')) "
        "FROM atlas_packets p WHERE p.load_id=l.load_id) AS packet_checksum "
        "FROM atlas_loads l ORDER BY l.load_id",
        {},
    )["rows"]
    assessments = query(
        "WITH records AS (SELECT DISTINCT pipeline_key,md5((to_jsonb(a)-'load_id')::text) AS digest "
        "FROM atlas_assessments a) SELECT pipeline_key,COUNT(*) AS records,"
        "md5(string_agg(digest,'' ORDER BY digest)) AS checksum "
        "FROM records GROUP BY pipeline_key ORDER BY pipeline_key",
        {},
    )["rows"]
    packets = query(
        "WITH records AS (SELECT DISTINCT pipeline_key,task_id,md5(packet) AS digest FROM atlas_packets) "
        "SELECT pipeline_key,COUNT(*) AS packets,md5(string_agg(digest,'' ORDER BY task_id,digest)) AS checksum "
        "FROM records GROUP BY pipeline_key ORDER BY pipeline_key",
        {},
    )["rows"]
    curriculum = query("SELECT * FROM curriculum_current ORDER BY name", {})["rows"]
    releases = query(
        "SELECT load_id,release_id,release_uri,md5(manifest::text) AS manifest_checksum "
        "FROM curriculum_loads ORDER BY load_id",
        {},
    )["rows"]
    identity_measurements = query(
        """SELECT m.load_id || '/' || m.measurement_id AS id,
        md5(m.metadata::text) AS metadata_checksum,
        (SELECT COUNT(*) FROM atlas_identity_measurement_reviews r
            WHERE r.load_id=m.load_id AND r.measurement_id=m.measurement_id) AS reviews,
        (SELECT md5(COALESCE(string_agg(md5(r.review::text),'' ORDER BY r.left_source,r.right_source),''))
            FROM atlas_identity_measurement_reviews r WHERE r.load_id=m.load_id
            AND r.measurement_id=m.measurement_id) AS reviews_checksum
        FROM atlas_identity_measurements m ORDER BY id""",
        {},
    )["rows"]
    identity_selection = query("SELECT * FROM atlas_identity_measurement_current ORDER BY load_id", {})["rows"]
    derived_contracts = query(
        (
            "SELECT source_id || '/' || receipt_id AS id,md5(receipt::text) AS checksum FRO"
            "M atlas_derived_contracts ORDER BY id"
        ),
        {},
    )["rows"]
    confidence_vectors = query(
        """SELECT l.load_id,
        (SELECT COUNT(*) FROM curriculum_confidence v WHERE v.load_id=l.load_id) AS vectors,
        (SELECT md5(COALESCE(string_agg(md5(to_jsonb(v)::text),'' ORDER BY v.scope_id,v.metric,
            v.subject_id,v.node_id),''))
         FROM curriculum_confidence v WHERE v.load_id=l.load_id) AS checksum
        FROM curriculum_loads l ORDER BY l.load_id""",
        {},
    )["rows"]
    end = query("SELECT load_id FROM atlas_current WHERE name='latest'", {})["rows"]
    if end != [{"load_id": current[0]["load_id"]}]:
        raise ValueError("Catalog selection changed while recording the baseline")
    return {
        "selected": current[0],
        "archives": archives,
        "assessments": assessments,
        "packets": packets,
        "curriculum": curriculum,
        "releases": releases,
        "identity_measurements": identity_measurements,
        "identity_selection": identity_selection,
        "derived_contracts": derived_contracts,
        "confidence_vectors": confidence_vectors,
    }


def verify_preservation(before: dict[str, Any], after: dict[str, Any], *, refresh_snapshot: str | None = None) -> None:
    """Allow an explicitly named new catalog, while preserving all prior evidence."""
    fields = [("releases", "load_id"), ("assessments", "pipeline_key"), ("packets", "pipeline_key")]
    for field, key in (("identity_measurements", "id"), ("derived_contracts", "id"), ("confidence_vectors", "load_id")):
        if field in before:
            fields.append((field, key))
    if refresh_snapshot is None:
        fields.append(("archives", "load_id"))
    for field, key_field in fields:
        previous = {row[key_field]: row for row in before[field]}
        current = {row[key_field]: row for row in after[field]}
        if any(current.get(key) != row for key, row in previous.items()):
            raise ValueError(f"Historical {field} were removed or changed")
    if refresh_snapshot is None:
        for field in ("selected", "curriculum", "identity_selection"):
            if field not in before:
                continue
            if before[field] != after[field]:
                raise ValueError(f"{field} changed during a code-only release")
    elif after["selected"]["snapshot_id"] != refresh_snapshot:
        raise ValueError("Selected catalog does not match the explicitly intended refresh")
