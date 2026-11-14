# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only release evidence and task-population accounting."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from .server.filtration import filtration
from .server.source_counts import SourceCounts

PREVIEW_ID = "c9742a15-fb46-4d56-bc88-aa916351ebd3"
PRODUCTION_ID = "fb11c931-5861-4878-8bb5-a964d652b45f"
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
    }


def verify_preservation(before: dict[str, Any], after: dict[str, Any], *, refresh_snapshot: str | None = None) -> None:
    """Allow an explicitly named new catalog, while preserving all prior evidence."""
    fields = [("releases", "load_id"), ("assessments", "pipeline_key"), ("packets", "pipeline_key")]
    if refresh_snapshot is None:
        fields.append(("archives", "load_id"))
    for field, key_field in fields:
        previous = {row[key_field]: row for row in before[field]}
        current = {row[key_field]: row for row in after[field]}
        if any(current.get(key) != row for key, row in previous.items()):
            raise ValueError(f"Historical {field} were removed or changed")
    if refresh_snapshot is None:
        for field in ("selected", "curriculum"):
            if before[field] != after[field]:
                raise ValueError(f"{field} changed during a code-only release")
    elif after["selected"]["snapshot_id"] != refresh_snapshot:
        raise ValueError("Selected catalog does not match the explicitly intended refresh")
