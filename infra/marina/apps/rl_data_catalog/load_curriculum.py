# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load compact curriculum statistics and review samples into the checked-in Atlas app."""

import json
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from itertools import batched
from typing import Any

import click
import pyarrow.parquet as pq
from finestore.rl.status_taxonomy import DataRestriction
from marina.client import DEFAULT_MARINA_URL, AppletQuery, applet_query
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.curriculum.models import AlignmentRelease, CurriculumCoverage, CurriculumStatistic

from experiments.post_training.task_curation.catalog_identity import canonical_declaration
from experiments.post_training.task_curation.retirement import retired_admission_source_ids
from infra.marina.apps.rl_data_catalog.confidence import confidence_rows
from infra.marina.apps.rl_data_catalog.load import (
    batches,
    ensure_curriculum_analyze_settings,
    ensure_tables,
    insert_rows,
    served_declaration_projection,
    sql_column_names,
    table_batch_bytes,
    table_batch_rows,
)
from infra.marina.apps.rl_data_catalog.role_projection import (
    CurriculumManifests,
    ManifestReference,
    manifest_bytes,
    manifest_documents,
    validated_role_projection,
)
from infra.marina.apps.rl_data_catalog.server.catalog_text import CATALOG_TEXT_CODEC, encode_catalog_value
from infra.marina.apps.rl_data_catalog.server.restrictions import declaration_restriction
from infra.marina.apps.rl_data_catalog.server.source_areas import materialize_source_areas
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

DELETE_BATCH_SIZE = 10_000
TASK_BATCH_ROWS = 256
TABLE_COLUMNS = {
    "sources": "source_id TEXT, declaration JSONB, family TEXT, alignment JSONB",
    "nodes": "subject_id TEXT, node_id TEXT, parent_id TEXT, kind TEXT, name TEXT, description TEXT",
    "ancestors": "subject_id TEXT, node_id TEXT, ancestor_id TEXT",
    "tasks": "source_id TEXT, task_id TEXT, content_hash TEXT, description_status TEXT, description TEXT, detail JSONB",
    "matches": (
        "source_id TEXT, task_id TEXT, subject_id TEXT, capability_id TEXT, facet TEXT, "
        "membership_similarity DOUBLE PRECISION, capability_similarity DOUBLE PRECISION"
    ),
    "statistics": "source_id TEXT, metric TEXT, subject_id TEXT, node_id TEXT, instances BIGINT",
    "confidence": "scope_id TEXT, metric TEXT, subject_id TEXT, node_id TEXT, instances BIGINT[]",
}


def ancestor_rows(nodes: list[dict]) -> Iterator[dict]:
    by_id = {(row["subject_id"], row["node_id"]): row for row in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("Curriculum contains duplicate node IDs")
    for row in nodes:
        current = row
        visited = set()
        while True:
            identifier = current["node_id"]
            if identifier in visited:
                raise ValueError("Curriculum hierarchy contains a cycle")
            visited.add(identifier)
            yield {"subject_id": row["subject_id"], "node_id": row["node_id"], "ancestor_id": identifier}
            if current["parent_id"] is None:
                break
            current = by_id[(row["subject_id"], current["parent_id"])]


def reconcile_population_counts(
    selected: dict[str, dict[str, Any]], full_counts: dict[str, Counter], task_count: int
) -> None:
    """Check coverage statistics against the release's matched sources and totals.

    Declared inventory sources carry no embedded tasks and contribute no statistics;
    they are exempt here because their count describes the pinned release, not a
    curriculum population. The total still sums every statistics source, so a
    statistics row appearing for a declared source fails the load.
    """
    for source, expected in selected.items():
        if expected["basis"] != "matched":
            continue
        actual = full_counts[source]
        if (
            actual["denominator"] != expected["task_count"]
            or actual["assigned"] + actual["subject_only"] + actual["out_of_scope"] != expected["ready_count"]
            or actual["insufficient_context"] != expected["insufficient_context_count"]
            or sum(
                actual[metric]
                for metric in ("assigned", "subject_only", "out_of_scope", "insufficient_context", "skipped")
            )
            != actual["denominator"]
        ):
            raise ValueError(f"Coverage population counts differ from release: {source}")
    if sum(counts["denominator"] for counts in full_counts.values()) != task_count:
        raise ValueError("Coverage population counts differ from release")


def release_tasks(paths: list[str]) -> Iterator[dict]:
    for path in paths:
        with StoragePath(path).open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    row["source_id"] = canonical_source_id(row["source_id"])
                    yield {
                        **{
                            key: row[key]
                            for key in ("source_id", "task_id", "content_hash", "description_status", "description")
                        },
                        "detail": row,
                    }


def preflight_task_text(paths: list[str]) -> int:
    """Check the exact task encoding and SQL transport limits without any writes."""
    count = 0
    for body in batches(
        (encode_catalog_value(row) for row in release_tasks(paths)),
        table_batch_bytes("curriculum_tasks"),
        table_batch_rows("curriculum_tasks"),
    ):
        count += len(json.loads(body))
    return count


def atlas_source_restrictions(query: AppletQuery, load_id: str) -> dict[str, dict[str, Any]]:
    """Read only policy inputs, preserving absent tags separately from explicit nulls."""
    rows = query(
        "SELECT id,jsonb_build_object('id',id) || CASE WHEN declaration ? 'data_restriction' "
        "THEN jsonb_build_object('data_restriction',declaration->'data_restriction') "
        "ELSE '{}'::jsonb END AS declaration FROM atlas_sources WHERE load_id=:load_id",
        {"load_id": load_id},
    )["rows"]
    return {row["id"]: row["declaration"] for row in rows}


def load_curriculum(
    release_uri: str,
    coverage_uri: str,
    query: AppletQuery,
    role_projection: ManifestReference | None = None,
    confidence: ManifestReference | None = None,
    resume_load_id: str | None = None,
    text_scan: ManifestReference | None = None,
) -> dict[str, Any]:
    """Load full-population statistics and bounded task examples, then select atomically."""
    started = time.monotonic()
    release = AlignmentRelease.model_validate_json(StoragePath(release_uri).read_text())
    release = release.model_copy(
        update={
            "sources": [
                row.model_copy(update={"source_id": canonical_source_id(row.source_id)}) for row in release.sources
            ],
            "inventory": [canonical_declaration(row) for row in release.inventory],
        }
    )
    retired = (
        {row.id for row in release.inventory} | {row.source_id for row in release.sources}
    ) & retired_admission_source_ids()
    if retired:
        raise ValueError(f"Retired sources cannot enter the selected curriculum: {sorted(retired)}")
    atlas_current = query("SELECT load_id FROM atlas_current WHERE name='latest'", {})["rows"]
    atlas_declarations: dict[str, dict[str, Any]] = {}
    if atlas_current:
        atlas_declarations = atlas_source_restrictions(query, atlas_current[0]["load_id"])
    private = sorted(
        {
            row.id
            for row in release.inventory
            if declaration_restriction(row.model_dump(mode="json")) is DataRestriction.PRIVATE_ALWAYS
            or declaration_restriction(atlas_declarations.get(row.id, {})) is DataRestriction.PRIVATE_ALWAYS
        }
    )
    if private:
        # A release carries each source's task previews and details: a PRIVATE_ALWAYS
        # source must never enter one. The catalog's authoritative tier (registry
        # override included) also refuses a release row whose stored tag was
        # flipped or dropped.
        raise ValueError(f"Curriculum release contains a PRIVATE_ALWAYS source: {private}")
    coverage = CurriculumCoverage.model_validate_json(StoragePath(coverage_uri).read_text())
    if (coverage.alignment_release_id, coverage.curriculum_key, coverage.task_count) != (
        release.release_id,
        release.curriculum_key,
        release.task_count,
    ):
        raise ValueError("Coverage statistics refer to a different alignment release")
    confidence_document = json.loads(manifest_bytes(confidence)) if confidence else None
    if confidence_document:
        release_reference, coverage_reference = confidence_document["release"], confidence_document["coverage"]
        if (release_reference["uri"], coverage_reference["uri"]) != (release_uri, coverage_uri):
            raise ValueError("Confidence references do not name the selected immutable manifests")
        manifest_documents(
            CurriculumManifests(
                release_uri=release_uri,
                release_sha256=release_reference["sha256"],
                release_id=release.release_id,
                coverage_uri=coverage_uri,
                coverage_sha256=coverage_reference["sha256"],
                coverage_id=coverage.coverage_id,
            )
        )
    vectors = list(confidence_rows(confidence_document, release, coverage)) if confidence_document else []
    if bool(coverage.confidence_levels) != bool(confidence_document):
        raise ValueError("A confidence coverage load requires its pinned sidecar")
    projection = (
        validated_role_projection(role_projection, release_uri, coverage_uri, query)
        if role_projection is not None
        else None
    )
    recorded_uses = {change.source_id: change.recorded_use for change in projection.changes} if projection else {}
    current = query(
        "SELECT l.load_id,l.release_id,l.release_uri,l.manifest->'confidence' AS confidence "
        "FROM curriculum_current c JOIN curriculum_loads l USING(load_id) WHERE c.name='latest'",
        {},
    )["rows"]
    if text_scan is not None:
        scan = json.loads(manifest_bytes(text_scan))
        if (
            scan["status"] != "PASS"
            or scan["codec"] != CATALOG_TEXT_CODEC
            or scan["artifacts"]["release"]["uri"] != release_uri
            or scan["artifacts"]["coverage"]["uri"] != coverage_uri
            or scan["task_count"] != release.task_count
            or scan["sample_task_count"] != coverage.sample_task_count
            or scan["source_count"] != len(release.sources)
            or scan["unsupported_identity_fields"] != 0
            or scan["all_rows_encoded_and_json_serialized"] is not True
            or scan["affected_rows_lossless_roundtrip"] is not True
            or (confidence is not None and scan["artifacts"]["confidence"] != confidence.model_dump(mode="json"))
            or (
                confidence_document is not None
                and (
                    scan["artifacts"]["release"]["sha256"] != confidence_document["release"]["sha256"]
                    or scan["artifacts"]["coverage"]["sha256"] != confidence_document["coverage"]["sha256"]
                )
            )
        ):
            raise ValueError("Catalog text preflight does not bind this complete curriculum")
    if current and (current[0]["release_id"], current[0]["release_uri"]) == (coverage.coverage_id, release_uri):
        previous_reference = (current[0]["confidence"] or {}).get("reference")
        if previous_reference != (confidence.model_dump(mode="json") if confidence else None):
            raise ValueError("The selected coverage ID already binds different confidence evidence")
        ensure_tables(query, ("curriculum_source_areas",))
        materialize_source_areas(lambda sql, bound: query(sql, bound).get("rows", []), current[0]["load_id"])
        return {"changed": False, "release_id": coverage.coverage_id}
    if preflight_task_text(coverage.sample_paths) != coverage.sample_task_count:
        raise ValueError("Text preflight sample counts differ from coverage release")
    load_id = resume_load_id or uuid.uuid4().hex
    if resume_load_id is not None:
        prior = query(
            "SELECT release_id,release_uri,manifest->>'release_id' AS alignment_id,"
            "manifest->'confidence'->'reference' AS confidence FROM curriculum_loads WHERE load_id=:load_id",
            {"load_id": load_id},
        )["rows"]
        if len(prior) != 1 or (
            prior[0]["release_id"],
            prior[0]["release_uri"],
            prior[0]["alignment_id"],
            prior[0]["confidence"],
        ) != (
            coverage.coverage_id,
            release_uri,
            release.release_id,
            confidence.model_dump(mode="json") if confidence else None,
        ):
            raise ValueError("Resume cursor belongs to a different immutable curriculum")
    requests = 1

    def execute(sql: str, parameters: dict[str, Any]) -> dict[str, Any]:
        nonlocal requests
        requests += 1
        return query(sql, parameters)

    def insert(table: str, rows: Iterable[dict]) -> int:
        updates = sql_column_names(TABLE_COLUMNS[table])
        return insert_rows(
            execute,
            f"curriculum_{table}",
            TABLE_COLUMNS[table],
            load_id,
            (encode_catalog_value(row) for row in rows),
            conflict_updates=updates if resume_load_id else (),
        )

    execute(
        "INSERT INTO curriculum_loads (load_id, release_id, release_uri, manifest) "
        "VALUES (:load_id, :release_id, :release_uri, CAST(:manifest AS JSONB))"
        + (" ON CONFLICT(load_id) DO UPDATE SET manifest=EXCLUDED.manifest" if resume_load_id else ""),
        {
            "load_id": load_id,
            "release_id": coverage.coverage_id,
            "release_uri": release_uri,
            "manifest": json.dumps(
                encode_catalog_value(
                    {
                        **release.model_dump(mode="json"),
                        "coverage": coverage.model_dump(mode="json"),
                        "role_projection": role_projection.model_dump(mode="json") if role_projection else None,
                        "confidence": (
                            {"reference": confidence.model_dump(mode="json"), "levels": confidence_document["levels"]}
                            if confidence and confidence_document
                            else None
                        ),
                        "catalog_text": {
                            "codec": CATALOG_TEXT_CODEC,
                            "scan": text_scan.model_dump(mode="json") if text_scan else None,
                        },
                    }
                )
            ),
        },
    )
    selected = {row.source_id: row.model_dump(mode="json") for row in release.sources}
    if len(selected) != len(release.sources) or selected.keys() - {row.id for row in release.inventory}:
        raise ValueError("Release source inventory is inconsistent")
    nodes = [row.model_dump(mode="json") for row in release.nodes]
    counts = {
        "sources": insert(
            "sources",
            (
                {
                    "source_id": row.id,
                    "declaration": served_declaration_projection(row.model_dump(mode="json")),
                    "family": row.family or row.origin,
                    "alignment": selected.get(row.id),
                }
                for row in release.inventory
            ),
        ),
        "nodes": insert("nodes", nodes),
        "ancestors": insert("ancestors", ancestor_rows(nodes)),
    }
    node_kinds = {(row["subject_id"], row["node_id"]): row["kind"] for row in nodes}

    def checked_tasks() -> Iterator[dict]:
        for row in release_tasks(coverage.sample_paths):
            detail = row["detail"]
            expected = selected.get(row["source_id"])
            subjects = (detail["matches"] or {}).get("subjects", [])
            if (
                expected is None
                or any(detail[key] != expected[key] for key in ("source_key", "population"))
                or detail["intended_use"] != recorded_uses.get(row["source_id"], expected["intended_use"])
                or (row["description_status"] == "ready") != bool(subjects)
                or row["description_status"] not in ("ready", "insufficient_context", "skipped")
            ):
                raise ValueError(f"Loaded task lineage or readiness is inconsistent: {row['task_id']}")
            for match in subjects:
                subject_id = match["subject_id"]
                if (
                    node_kinds.get((subject_id, "subject:" + subject_id)) != "subject"
                    or node_kinds.get((subject_id, match["capability_id"])) != "capability"
                ):
                    raise ValueError(f"Loaded task curriculum references are inconsistent: {row['task_id']}")
            yield row

    counts["tasks"] = 0
    counts["matches"] = 0
    for rows in batched(checked_tasks(), TASK_BATCH_ROWS):
        counts["tasks"] += insert("tasks", rows)
        counts["matches"] += insert(
            "matches",
            (
                {
                    "source_id": row["source_id"],
                    "task_id": row["task_id"],
                    "subject_id": match["subject_id"],
                    "capability_id": match["capability_id"],
                    "facet": match["facet"],
                    "membership_similarity": match["membership_similarity"],
                    "capability_similarity": match["candidates"][0]["similarity"],
                }
                for row in rows
                for match in (row["detail"]["matches"] or {}).get("subjects", [])
            ),
        )
    if counts["tasks"] != coverage.sample_task_count:
        raise ValueError("Loaded sample counts differ from coverage release")
    full_counts: dict[str, Counter] = defaultdict(Counter)

    def checked_statistics() -> Iterator[dict]:
        for path in coverage.paths:
            with StoragePath(path).open("rb") as stream:
                for batch in pq.ParquetFile(stream).iter_batches(batch_size=256):
                    for raw in batch.to_pylist():
                        row = CurriculumStatistic.model_validate(
                            {**raw, "source_id": canonical_source_id(raw["source_id"])}
                        )
                        if row.source_id not in selected:
                            raise ValueError(f"Statistics contain an unknown source: {row.source_id}")
                        if row.metric in ("placement", "capability"):
                            if (row.subject_id, row.node_id) not in node_kinds:
                                raise ValueError(f"Statistics contain an unknown curriculum area: {row.node_id}")
                        else:
                            full_counts[row.source_id][row.metric] += row.instances
                        yield row.model_dump(mode="json")

    if confidence:
        ensure_tables(execute, ("curriculum_confidence",))
    counts["statistics"] = insert("statistics", checked_statistics())
    reconcile_population_counts(selected, full_counts, release.task_count)
    if confidence:
        counts["confidence"] = insert("confidence", vectors)
    ensure_tables(execute, ("curriculum_source_areas",))
    materialize_source_areas(lambda sql, bound: execute(sql, bound).get("rows", []), load_id)
    ensure_curriculum_analyze_settings(execute)
    execute(
        "UPDATE curriculum_current SET previous_load_id=load_id, load_id=:load_id WHERE name='latest'",
        {"load_id": load_id},
    )
    return {
        "changed": True,
        "release_id": coverage.coverage_id,
        "population_tasks": release.task_count,
        "counts": counts,
        "sql_requests": requests,
        "seconds": time.monotonic() - started,
    }


@click.command(help=__doc__)
@click.option("--release", "release_uri", required=True, help="Immutable release.json; run in its storage region.")
@click.option("--coverage", "coverage_uri", required=True, help="Compact coverage.json with statistics and samples.")
@click.option("--confidence", type=(str, str), metavar="URI SHA256", help="Pinned measured five-level sidecar.")
@click.option("--resume-load-id", help="Resume an unselected load of these exact immutable artifacts.")
@click.option("--text-scan", type=(str, str), metavar="URI SHA256", help="Full-input lossless text preflight receipt.")
@click.option("--app-id", "applet_id", required=True)
@click.option("--url", "service_url", default=DEFAULT_MARINA_URL, show_default=True)
@click.option(
    "--role-projection",
    type=(str, str),
    metavar="URI SHA256",
    help="Explicit hashed metadata-only role-correction receipt; ordinary imports keep exact-role checks.",
)
def main(
    release_uri: str,
    coverage_uri: str,
    applet_id: str,
    service_url: str,
    role_projection: tuple[str, str] | None,
    confidence: tuple[str, str] | None,
    resume_load_id: str | None,
    text_scan: tuple[str, str] | None,
) -> None:
    configure_coreweave_s3()
    reference = ManifestReference(uri=role_projection[0], sha256=role_projection[1]) if role_projection else None
    confidence_reference = ManifestReference(uri=confidence[0], sha256=confidence[1]) if confidence else None
    text_scan_reference = ManifestReference(uri=text_scan[0], sha256=text_scan[1]) if text_scan else None
    click.echo(
        json.dumps(
            load_curriculum(
                release_uri,
                coverage_uri,
                applet_query(service_url, applet_id),
                reference,
                confidence_reference,
                resume_load_id=resume_load_id,
                text_scan=text_scan_reference,
            )
        )
    )


if __name__ == "__main__":
    main()
