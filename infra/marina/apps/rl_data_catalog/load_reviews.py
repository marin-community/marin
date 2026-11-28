# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage frozen SAMPLE panels and exact per-task reviewer request/response evidence."""

import dataclasses
import hashlib
import io
import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import click
import pyarrow.parquet as pq
from finestore.rl.curation import Assessment, AssessmentPopulation, CurationSummary, ModelReference
from marina.client import DEFAULT_MARINA_URL, AppletQuery, MarinaRequestError, applet_query
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.models import ReviewRecord

from experiments.post_training.task_curation.full_qc_artifacts import FullQCReview
from experiments.post_training.task_curation.review_quality import quality_result
from experiments.post_training.task_curation.sample_qc_artifacts import SampleQCReview
from infra.marina.apps.rl_data_catalog.load import (
    MAX_BATCH_BYTES,
    TABLE_COLUMNS,
    batches,
    ensure_tables,
    full_review_items,
)
from infra.marina.apps.rl_data_catalog.role_projection import ManifestReference, manifest_bytes
from infra.marina.apps.rl_data_catalog.server.cohort_summaries import materialize_cohort_summaries
from infra.marina.apps.rl_data_catalog.server.content_access import content_is_withheld
from infra.marina.apps.rl_data_catalog.server.evidence_content import pack_content
from infra.marina.apps.rl_data_catalog.server.restrictions import declaration_restriction
from infra.marina.apps.rl_data_catalog.server.review_content import task_request_evidence


def read_batch(uri: str) -> tuple[dict, dict]:
    raw = StoragePath(uri).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    pointer_uri = str(StoragePath(uri).parent.parent / "latest.json")
    pointer_path = StoragePath(pointer_uri)
    pointer_raw = pointer_path.read_bytes() if pointer_path.exists() else None
    pointer = json.loads(pointer_raw) if pointer_raw else None
    basis = "staging_readback"
    if pointer and pointer["path"] == uri:
        if pointer["sha256"] != digest:
            raise ValueError(f"Publisher batch digest differs from its original receipt: {uri}")
        basis = "publisher_pointer"
    # A later retry can advance the pointer. The frozen records select the
    # original UUID receipt, never that retry's output. Its typed observation
    # and task/runtime hashes are checked below; this new seal records exactly
    # which original bytes were read without claiming a historic publisher hash.
    if StoragePath(uri).read_bytes() != raw:
        raise ValueError(f"Original review receipt changed during staging: {uri}")
    return json.loads(raw), {
        "receipt": {"uri": uri, "sha256": digest},
        "hash_basis": basis,
        "pointer_at_staging": (
            {"uri": pointer_uri, "sha256": hashlib.sha256(pointer_raw).hexdigest(), "selected_attempt": pointer["path"]}
            if pointer_raw and pointer
            else None
        ),
    }


def reviewer_profiles(reviewer: dict) -> list[dict]:
    """Retain both named reviewer profiles in a sealed repair overlay."""
    if "model" in reviewer:
        return [reviewer]
    if reviewer["mode"] != "repair_overlay":
        raise ValueError("Unknown composite reviewer identity")
    return [*reviewer_profiles(reviewer["primary"]), *reviewer_profiles(reviewer["repair"])]


def stage_content_reviews(
    reference: ManifestReference, source: dict, workers: int, mapping_reference: ManifestReference | None = None
) -> dict[str, Any]:
    """Resolve the frozen declaration's panel, retaining its original input identity."""
    if content_is_withheld(source):
        raise ValueError("Withheld reviews remain in their private store; do not stage content")
    selected = (source.get("independent_qc") or {}).get("evidence_url")
    expected_uri = selected if selected and selected.endswith(".artifact.json") else f"{selected}/.artifact.json"
    if reference.uri != expected_uri:
        raise ValueError("Review artifact is not the evidence selected by the frozen declaration")
    artifact = json.loads(manifest_bytes(reference))
    result = artifact["result"]
    if "panel" in result:
        review = SampleQCReview.model_validate({**result, "path": reference.uri.rsplit("/", 1)[0]})
        panel = review.panel.model_dump(mode="json")
        scope = "sample"
    else:
        review = FullQCReview.model_validate({**result, "path": reference.uri.rsplit("/", 1)[0]})
        population = review.inputs.population
        panel = {
            "source_id": review.source_id,
            "dataset_id": population.dataset_id,
            "dataset_revision": population.dataset_revision,
            "converter_version": population.source_version,
            "task_specs_sha256": population.task_specs_sha256,
            "retained_tasks": review.inputs.retained_count,
        }
        scope = "full"
    if panel["source_id"] != source["id"]:
        raise ValueError("Panel belongs to another source")
    records_ref = ManifestReference(uri=review.records_path, sha256=review.records_sha256)
    rows = pq.read_table(io.BytesIO(manifest_bytes(records_ref))).to_pylist()
    records = [ReviewRecord.model_validate_json(row["review"]) for row in rows]
    if len(records) != review.attempted_count or len({row.task_id for row in records}) != len(records):
        raise ValueError("Review records do not cover the frozen panel exactly once")
    if quality_result(records) != review.result:
        raise ValueError("Panel observations disagree with the recorded content Result")
    mapping_entries = None
    if mapping_reference:
        mapping = json.loads(manifest_bytes(mapping_reference))
        if mapping["artifact"] != reference.model_dump(mode="json") or mapping["records"] != records_ref.model_dump(
            mode="json"
        ):
            raise ValueError("Original-batch mapping belongs to another sealed review")
        mapping_entries = {row["task_id"]: row for row in mapping["entries"]}
        if set(mapping_entries) != {row["task_id"] for row in rows}:
            raise ValueError("Original-batch mapping does not cover every sealed row")
        for row in rows:
            mapped = mapping_entries[row["task_id"]]
            if any(mapped[field] != row[field] for field in ("task_id", "task_sha256", "runtime_sha256")):
                raise ValueError("Original-batch mapping input identity differs")
            row["batch_receipt"] = mapped["receipt"]["uri"] if mapped["status"] == "mapped" else None
    else:
        allowed_batches = set(review.batch_receipts)
        if any(row["batch_receipt"] not in allowed_batches for row in rows):
            raise ValueError("Review record cites a batch outside the bound artifact")
    batch_uris = sorted({row["batch_receipt"] for row in rows if row["batch_receipt"]})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        batch_rows = dict(zip(batch_uris, pool.map(read_batch, batch_uris), strict=True))
    key = f"qc:{reference.sha256}"
    request_evidence = {uri: task_request_evidence(batch) for uri, (batch, _) in batch_rows.items()}
    raw_records: list[dict[str, Any]] = []
    for row, record in zip(rows, records, strict=True):
        if row["batch_receipt"] is None:
            assert mapping_entries is not None and mapping_reference is not None
            raw_records.append(
                {
                    "pipeline_key": key,
                    "task_id": record.task_id,
                    "reviewer": {"model": "Not mapped", "revision": None},
                    "record": {
                        "input": [],
                        "output": [],
                        "typed_observation": record.model_dump(mode="json"),
                        "unavailable": mapping_entries[record.task_id]["reason"],
                        "provenance": {
                            "mapping": mapping_reference.model_dump(mode="json"),
                            "mapping_status": "not_mapped",
                            "records": records_ref.model_dump(mode="json"),
                        },
                    },
                }
            )
            continue
        batch, provenance = batch_rows[row["batch_receipt"]]
        if mapping_entries:
            mapped = mapping_entries[record.task_id]
            if provenance["receipt"] != mapped["receipt"]:
                raise ValueError("Original batch changed after mapping was sealed")
            provenance.update(
                mapping=mapping_reference.model_dump(mode="json"),
                mapping_status="mapped",
                hash_basis=mapped["hash_basis"],
            )
        actual = next(review for review in batch["reviews"] if review["task_id"] == record.task_id)
        if ReviewRecord.model_validate_json(json.dumps(actual)) != record:
            raise ValueError(f"Raw batch judgment differs from its consolidated row: {record.task_id}")
        contract = next(contract for contract in batch["identity"]["contracts"] if contract["task_id"] == record.task_id)
        if mapping_entries:
            if contract != mapping_entries[record.task_id]["original_contract"]:
                raise ValueError("Original reviewer input identity changed after its mapping was sealed")
            provenance.update(
                lineage="Original-reviewer lineage; not current-pin QA",
                consolidation_contract={field: row[field] for field in ("task_id", "task_sha256", "runtime_sha256")},
            )
        elif any(contract[field] != row[field] for field in ("task_id", "task_sha256", "runtime_sha256")):
            raise ValueError("Batch input differs from the panel contract")
        raw_records.append(
            {
                "pipeline_key": key,
                "task_id": record.task_id,
                "reviewer": {
                    "model": batch["identity"]["reviewer"]["model"],
                    "revision": batch["identity"]["reviewer"].get("model_revision"),
                },
                "record": {
                    **request_evidence[row["batch_receipt"]].get(record.task_id, {"input": [], "output": []}),
                    "typed_observation": actual,
                    "provenance": {
                        **provenance,
                        "records": records_ref.model_dump(mode="json"),
                        "contract": contract,
                        "reviewer": batch["identity"]["reviewer"],
                    },
                },
            }
        )
    for raw in raw_records:
        if not raw["record"]["input"] and not raw["record"]["output"] and "unavailable" not in raw["record"]:
            raw["record"][
                "unavailable"
            ] = "The original receipt contains no captured model request or response for this task."
    reviewer = review.reviewer
    profiles = reviewer_profiles(reviewer)
    models = sorted({profile["model"] for profile in profiles})
    revisions = sorted({profile["model_revision"] for profile in profiles})
    assessment = Assessment(
        pipeline_key=key,
        source_id=source["id"],
        scope=scope,
        model=ModelReference(
            model=" / ".join(models) + (" (multiple profiles)" if len(profiles) > 1 else ""),
            revision=" / ".join(revisions),
            max_tokens=max(profile["max_tokens"] for profile in profiles),
            max_attempts=max(profile["max_attempts"] for profile in profiles),
            retry_max_tokens=max(profile["retry_max_tokens"] for profile in profiles),
            reviewer_spec_json=json.dumps(reviewer),
        ),
        rubric=json.dumps(review.rubric),
        curation=CurationSummary(
            status="complete",
            mode=scope,
            source_revision=panel["dataset_revision"],
            recipe_revision=None,
            raw_population_count=panel["retained_tasks"],
            raw_sample_count=len(records),
            processed_rows=len(records),
            unprocessed_rows=0,
            admission_counts={},
            verification=None,
            quality_result=review.result,
        ),
        curation_uri=reference.uri,
        traces=[],
    )
    populations = [
        AssessmentPopulation(
            pipeline_key=key, cohort="admitted", population=panel["retained_tasks"], sample_count=len(records)
        ),
        AssessmentPopulation(pipeline_key=key, cohort="excluded", population=0, sample_count=0),
    ]
    items = full_review_items(key, records)
    # SAMPLE summary packet semantics require recorded reviewer inputs; these
    # raw requests are the actual task evidence, without fabricating task packets.
    items = (
        [
            item.model_copy(update={"has_task": bool(raw["record"]["input"])})
            for item, raw in zip(items, raw_records, strict=True)
        ]
        if scope == "sample"
        else items
    )
    raw_messages_missing = sum(not row["record"]["input"] and not row["record"]["output"] for row in raw_records)
    shared_batches = {}
    batch_links = []
    for row in raw_records:
        uri = (row["record"]["provenance"].get("receipt") or {}).get("uri")
        if uri is None:
            continue
        batch, provenance = batch_rows[uri]
        batch_key = provenance["receipt"]["sha256"]
        if batch_key not in shared_batches:
            shared_batches[batch_key] = {
                "pipeline_key": key,
                "batch_key": batch_key,
                "record": pack_content({"attempts": batch["attempts"]}),
            }
        batch_links.append({"pipeline_key": key, "task_id": row["task_id"], "batch_key": batch_key})
        del row["record"]["input"], row["record"]["output"]
    current = source.get("actor_census_input") or {}
    current_pin = (
        panel["dataset_id"],
        panel["dataset_revision"],
        panel["converter_version"],
        panel["task_specs_sha256"],
        panel["retained_tasks"],
    ) == (
        source.get("dataset_id"),
        source.get("dataset_revision"),
        (source.get("pipeline") or {}).get("version"),
        current.get("task_specs_sha256"),
        current.get("measured_tasks"),
    )
    return {
        "data_restriction": declaration_restriction(source).value,
        "assessment": assessment.model_dump(mode="json"),
        "populations": [row.model_dump(mode="json") for row in populations],
        "items": [row.model_dump(mode="json") for row in items],
        "raw": raw_records,
        "batches": list(shared_batches.values()),
        "batch_links": batch_links,
        "receipt": {
            "artifact": reference.model_dump(mode="json"),
            "panel": panel,
            "current_pin": current_pin,
            "mapping": mapping_reference.model_dump(mode="json") if mapping_reference else None,
            "mapping_counts": {key: mapping[key] for key in ("mapped", "not_mapped")} if mapping_reference else None,
            "raw_messages_missing": raw_messages_missing,
            "linked_tasks": len(batch_links),
        },
    }


def matched_review_rows(query: AppletQuery, sql: str, load_id: str, rows: list[dict]) -> int:
    """Reduce deadline-bound reads while retaining the exact comparison for every row."""
    try:
        return query(sql, {"load": load_id, "rows": json.dumps(rows)})["rows"][0]["matched"]
    except MarinaRequestError as error:
        if error.status_code != 504 or len(rows) <= 1:
            raise
        logging.getLogger(__name__).warning("Splitting timed-out review readback: %d rows", len(rows))
        middle = len(rows) // 2
        return matched_review_rows(query, sql, load_id, rows[:middle]) + matched_review_rows(
            query, sql, load_id, rows[middle:]
        )


def panel_row_groups(staged: dict) -> list[tuple[str, list[dict], str]]:
    """The panel's per-table row groups exactly as the loader inserts them.

    Shared by the loading path and the replay's pre-replay record total, so the
    two can never disagree about what one panel replays.
    """
    key = staged["assessment"]["pipeline_key"]
    batch_rows = staged.get("batches", [])
    chunk_rows = []
    stored_batches = []
    for row in batch_rows:
        record = dict(row["record"])
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if len(json.dumps(encoded).encode()) + 3 > MAX_BATCH_BYTES:
            payload = record.pop("payload")
            chunks = [payload[start : start + 400_000] for start in range(0, len(payload), 400_000)]
            record["payload_chunks"] = len(chunks)
            chunk_rows.extend(
                {"pipeline_key": key, "batch_key": row["batch_key"], "chunk_index": index, "payload": part}
                for index, part in enumerate(chunks)
            )
        stored_batches.append({**row, "record": record, "chunk_count": record.get("payload_chunks", 0)})
    return [
        ("atlas_review_batches", stored_batches, "pipeline_key TEXT,batch_key TEXT,record JSONB,chunk_count INTEGER"),
        ("atlas_review_batch_chunks", chunk_rows, "pipeline_key TEXT,batch_key TEXT,chunk_index INTEGER,payload TEXT"),
        ("atlas_review_batch_tasks", staged.get("batch_links", []), "pipeline_key TEXT,task_id TEXT,batch_key TEXT"),
        ("atlas_assessments", [staged["assessment"]], TABLE_COLUMNS["assessments"]),
        ("atlas_populations", staged["populations"], TABLE_COLUMNS["populations"]),
        ("atlas_items", staged["items"], TABLE_COLUMNS["items"]),
        (
            "atlas_raw_reviews",
            [{**row, "record": pack_content(row["record"]), "reviewer": row.get("reviewer")} for row in staged["raw"]],
            "pipeline_key TEXT,task_id TEXT,record JSONB,reviewer JSONB",
        ),
    ]


REVIEW_TABLES = (
    "atlas_review_batches",
    "atlas_review_batch_chunks",
    "atlas_review_batch_tasks",
    "atlas_review_current",
    "atlas_raw_reviews",
)


@dataclasses.dataclass(frozen=True)
class PreparedReviewPanel:
    """One staged panel past its validation, ready to insert segment by segment."""

    source_id: str
    load_id: str
    key: str
    staged: dict
    groups: list[tuple[str, list[dict], str]]


def prepare_content_review(
    staged: dict, load_id: str, source_id: str, query: AppletQuery, *, ensure_tables_: bool = True
) -> PreparedReviewPanel:
    """Validate the staged panel against the load's selected source and mark chunk metadata.

    The parallel replay runs this once per panel before any segment inserts: the
    checks read the load's declarations, the panel-scoped chunk_count update touches
    only this panel's rows, and the insertable groups are computed from the sealed
    document alone. ``ensure_tables_=False`` skips the idempotent table DDL — the
    parallel driver has already run it once for the whole replay, and concurrent
    ``CREATE TABLE IF NOT EXISTS`` statements race.
    """
    declarations = query(
        "SELECT declaration FROM atlas_sources WHERE load_id=:load AND id=:source",
        {"load": load_id, "source": source_id},
    )["rows"]
    if len(declarations) != 1 or content_is_withheld(declarations[0]["declaration"]):
        raise ValueError("Review content may only be staged for an open selected source")
    if staged["data_restriction"] != declaration_restriction(declarations[0]["declaration"]).value:
        raise ValueError("Staged review restriction is missing or differs from its source")
    declaration = declarations[0]["declaration"]
    artifact = ManifestReference.model_validate(staged["receipt"]["artifact"])
    selected = (declaration.get("independent_qc") or {}).get("evidence_url")
    selected_uri = selected if selected and selected.endswith(".artifact.json") else f"{selected}/.artifact.json"
    key = f"qc:{artifact.sha256}"
    if artifact.uri != selected_uri or staged["assessment"]["source_id"] != source_id:
        raise ValueError("Staged review does not belong to the frozen source's selected evidence")
    if staged["assessment"]["pipeline_key"] != key or any(
        row["pipeline_key"] != key for field in ("populations", "items", "raw") for row in staged[field]
    ):
        raise ValueError("Staged review rows mix immutable pipeline identities")
    if {row["task_id"] for row in staged["items"]} != {row["task_id"] for row in staged["raw"]}:
        raise ValueError("Raw reviewer evidence does not cover exactly the staged task rows")
    batch_rows = staged.get("batches", [])
    links = staged.get("batch_links", [])
    batch_keys = {row["batch_key"] for row in batch_rows}
    if len(batch_keys) != len(batch_rows) or {row["batch_key"] for row in links} != batch_keys:
        raise ValueError("Shared original-batch identities do not match their task links")
    if len({row["task_id"] for row in links}) != len(links) or not {row["task_id"] for row in links}.issubset(
        {row["task_id"] for row in staged["raw"]}
    ):
        raise ValueError("Original-batch links contain duplicate or unknown reviewed task IDs")
    if any(row["pipeline_key"] != key for row in [*batch_rows, *links]):
        raise ValueError("Original reviewer batches belong to another immutable panel")
    if ensure_tables_:
        ensure_tables(query, REVIEW_TABLES)
    query("ALTER TABLE atlas_review_batches ADD COLUMN IF NOT EXISTS chunk_count INTEGER NOT NULL DEFAULT 0", {})
    # Old chunked records contain only small hash/size/count metadata; avoid detoasting unchunked payloads.
    query(
        "UPDATE atlas_review_batches SET chunk_count=CASE WHEN pg_column_size(record)<1024 "
        "THEN COALESCE(CAST(record->>'payload_chunks' AS INTEGER),0) ELSE 0 END "
        "WHERE load_id=:load AND pipeline_key=:key AND chunk_count=0 "
        "AND CASE WHEN pg_column_size(record)<1024 THEN record ? 'payload_chunks' ELSE FALSE END",
        {"load": load_id, "key": key},
    )
    return PreparedReviewPanel(
        source_id=source_id, load_id=load_id, key=key, staged=staged, groups=panel_row_groups(staged)
    )


def insert_review_segment(
    query: AppletQuery,
    load_id: str,
    key: str,
    table: str,
    columns: str,
    rows: list[dict],
    progress: Callable[[str, int], None] | None = None,
) -> int:
    """Insert one table's rows with the exact per-batch readback, returning the inserted count."""
    names = ",".join(column.strip().split()[0] for column in columns.split(","))
    count = 0
    for body in batches(rows):
        count += query(
            f"INSERT INTO {table} (load_id,{names}) SELECT :load,{names} "
            f"FROM jsonb_to_recordset(CAST(:rows AS JSONB)) AS r({columns}) ON CONFLICT DO NOTHING",
            {"load": load_id, "rows": body},
        )["row_count"]
        identities = {
            "atlas_populations": ("cohort",),
            "atlas_items": ("task_id",),
            "atlas_raw_reviews": ("task_id",),
            "atlas_review_batches": ("batch_key",),
            "atlas_review_batch_chunks": ("batch_key", "chunk_index"),
            "atlas_review_batch_tasks": ("task_id",),
        }.get(table, ())
        join = " AND ".join(f"stored.{name}=expected.{name}" for name in ("pipeline_key", *identities))
        fields = names.split(",")
        stored_fields = ",".join(f"stored.{name}" for name in fields)
        expected_fields = ",".join(f"expected.{name}" for name in fields)
        try:
            matched = matched_review_rows(
                query,
                f"SELECT COUNT(*) AS matched FROM jsonb_to_recordset(CAST(:rows AS JSONB)) "
                f"AS expected({columns}) JOIN {table} stored ON stored.load_id=:load AND {join} "
                f"WHERE ROW({stored_fields}) IS NOT DISTINCT FROM ROW({expected_fields})",
                load_id,
                json.loads(body),
            )
        except Exception as error:
            raise RuntimeError(f"{table} exact readback failed for {key}, {len(json.loads(body))} rows") from error
        if matched != len(json.loads(body)):
            raise ValueError(f"Stored {table} differs from its sealed panel")
    if progress is not None:
        progress(table, len(rows))
    return count


def finish_content_review(
    query: AppletQuery,
    prepared: PreparedReviewPanel,
    progress: Callable[[str, int], None] | None = None,
) -> None:
    """Materialize the panel's cohort summaries and publish its presentation pointer last."""
    materialize_cohort_summaries(
        lambda sql, params: query(sql, params)["rows"], prepared.load_id, pipeline_key=prepared.key
    )
    query(
        "INSERT INTO atlas_review_current VALUES (:load,:source,:key,CAST(:receipt AS JSONB)) "
        "ON CONFLICT (load_id,source_id) DO UPDATE SET "
        "pipeline_key=EXCLUDED.pipeline_key,receipt=EXCLUDED.receipt "
        "WHERE atlas_review_current.pipeline_key IS DISTINCT FROM EXCLUDED.pipeline_key "
        "OR atlas_review_current.receipt IS DISTINCT FROM EXCLUDED.receipt",
        {
            "load": prepared.load_id,
            "source": prepared.source_id,
            "key": prepared.key,
            "receipt": json.dumps(prepared.staged["receipt"]),
        },
    )
    if query(
        "SELECT pipeline_key,receipt FROM atlas_review_current WHERE load_id=:load AND source_id=:source",
        {"load": prepared.load_id, "source": prepared.source_id},
    )["rows"] != [{"pipeline_key": prepared.key, "receipt": prepared.staged["receipt"]}]:
        raise ValueError("Frozen source already selects another review panel")
    if progress is not None:
        progress("atlas_review_current", 1)


def load_content_reviews(
    staged: dict,
    load_id: str,
    source_id: str,
    query: AppletQuery,
    progress: Callable[[str, int], None] | None = None,
) -> dict[str, int]:
    """Select a verified panel additively, leaving catalog declarations and old review bytes intact."""
    prepared = prepare_content_review(staged, load_id, source_id, query)
    counts: dict[str, int] = {}
    for table, rows, columns in prepared.groups:
        counts[table] = insert_review_segment(query, load_id, prepared.key, table, columns, rows, progress)
    finish_content_review(query, prepared, progress)
    return counts


@click.command(help=__doc__)
@click.option("--artifact", type=(str, str), required=True, metavar="URI SHA256")
@click.option(
    "--mapping", type=(str, str), metavar="URI SHA256", help="Immutable original-batch mapping for a FULL assembly."
)
@click.option("--source-id", required=True)
@click.option("--app-id", required=True)
@click.option("--url", default=DEFAULT_MARINA_URL)
@click.option("--receipt", type=click.Path(path_type=Path), required=True)
def main(
    artifact: tuple[str, str], mapping: tuple[str, str] | None, source_id: str, app_id: str, url: str, receipt: Path
) -> None:
    if receipt.exists():
        raise click.ClickException("Choose a new receipt path")
    configure_coreweave_s3()
    query = applet_query(url, app_id)
    source = query(
        "SELECT s.load_id,s.declaration FROM atlas_sources s JOIN atlas_current c USING(load_id) "
        "WHERE c.name='latest' AND s.id=:source",
        {"source": source_id},
    )["rows"][0]
    staged = stage_content_reviews(
        ManifestReference(uri=artifact[0], sha256=artifact[1]),
        source["declaration"],
        8,
        ManifestReference(uri=mapping[0], sha256=mapping[1]) if mapping else None,
    )
    result = load_content_reviews(staged, source["load_id"], source_id, query)
    receipt.write_text(
        json.dumps(
            {"source_id": source_id, "load_id": source["load_id"], "counts": result, "review": staged["receipt"]},
            indent=2,
        )
        + "\n"
    )
    click.echo(json.dumps(result))


if __name__ == "__main__":
    main()
