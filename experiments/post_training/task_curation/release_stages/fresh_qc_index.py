# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The fresh-QC-index stage: the graph's producer of the release's staged QC panel index.

The release's fresh QC index is a graph output bound to the release's target load and
snapshot (ruling 2026-11-24), not a hand-produced pin. Unchanged panels are exported
from the prior load's review tables with their original receipt. Other panels are
read from digest-bound review artifacts selected by the frozen declarations through
the content-review loader. Both paths preserve the review's task IDs and produce
deterministically compressed staged documents that the catalog load replays.
"""

import argparse
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.release_stages import paced_query
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref
from experiments.post_training.task_curation.release_stages.qc_browse import (
    changed_review_sources,
    pin_currency,
    selected_uri,
)
from infra.marina.apps.rl_data_catalog.load_reviews import stage_content_reviews
from infra.marina.apps.rl_data_catalog.role_projection import ManifestReference, manifest_bytes
from infra.marina.apps.rl_data_catalog.server.restrictions import declaration_restriction

INDEX_KIND_SUFFIX = "fresh-qc-index"
STAGED_DIR = "staged-v1"


@dataclass(frozen=True)
class FreshQcIndexInputs:
    """Typed inputs for the release's fresh QC index; every value is a graph output."""

    release_id: str
    prior_load_id: str
    target_load_id: str
    snapshot_id: str
    source_revision: str
    output_root: str
    service_url: str
    applet_id: str
    declarations: ArtifactRef
    qc_completion: ArtifactRef
    fresh_panels: dict[str, ArtifactRef]
    prior_index: ArtifactRef | None = None
    removed_sources: tuple[str, ...] = ()

    def validate(self) -> None:
        """Refuse malformed inputs before anything runs."""
        if not self.release_id or not self.prior_load_id or not self.target_load_id or not self.snapshot_id:
            raise ValueError("release_id, load ids and snapshot are required")
        if not self.source_revision or not self.output_root:
            raise ValueError("source_revision and output_root are required")
        if not self.service_url or not self.applet_id:
            raise ValueError("the served applet endpoint is required")


def _read_declarations(ref: ArtifactRef) -> dict[str, dict]:
    """Digest-verify the frozen declarations document the panels bind."""
    payload = StoragePath(ref.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref.sha256:
        raise ValueError(f"declarations digest mismatch: {ref.uri}")
    document = json.loads(payload)
    if not isinstance(document, dict) or not document:
        raise ValueError("declarations document is empty or malformed")
    return document


def _completion_rows(ref: ArtifactRef, *, release_id: str, prior_load_id: str, target_load_id: str) -> list[dict]:
    """The browse completion's staged panels."""
    payload = StoragePath(ref.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref.sha256:
        raise ValueError(f"browse completion digest mismatch: {ref.uri}")
    completion = json.loads(payload)
    if completion.get("status") != "PASS":
        raise ValueError(f"the browse completion is {completion.get('status')!r}; the index needs a PASS")
    if (
        completion["release_id"] != release_id
        or completion["source_load_id"] != prior_load_id
        or completion["target_load_id"] != target_load_id
        or completion.get("selected_pointer_changed")
    ):
        raise ValueError("browse completion names different release loads or changed selection")
    return completion.get("results") or []


PAGE_ROWS = 500
PANEL_READ_WORKERS = 4


def _paged(query, sql: str, params: dict, order_columns: tuple[str, ...]) -> list[dict]:
    """Read one panel's table rows through the applet's bounded response size.

    Pages by a keyset over the sort key (composite keys keep every row — a single
    column would skip rows sharing it) and halves the page size when a page exceeds
    the applet's response cap; a single row larger than the cap refuses.
    """
    from marina.client import MarinaRequestError  # noqa: PLC0415 — beside its only use

    rows: list[dict] = []
    after: dict[str, object] | None = None
    while True:
        bound = dict(params)
        predicate = ""
        if after is not None:
            for index, column in enumerate(order_columns):
                bound[f"after_{index}"] = after[column]
            clauses = []
            for index, column in enumerate(order_columns):
                equal = " AND ".join(f"{earlier} = :after_{j}" for j, earlier in enumerate(order_columns[:index]))
                clauses.append(f"({equal} AND {column} > :after_{index})" if equal else f"{column} > :after_{index}")
            predicate = " AND (" + " OR ".join(clauses) + ")"
        page_rows = PAGE_ROWS
        while True:
            try:
                page = query(sql.format(predicate=predicate) + f" LIMIT {page_rows}", bound)["rows"]
                break
            except MarinaRequestError as error:
                if error.status_code != 413 or page_rows == 1:
                    raise
                page_rows //= 2
        if not page:
            return rows
        rows.extend(page)
        after = {column: page[-1][column] for column in order_columns}


def _export_panel(inputs: FreshQcIndexInputs, row: dict, declarations: dict[str, dict], query, *, write: bool) -> dict:
    """Export one panel's replayable staged document from the prior load's review tables."""
    from infra.marina.apps.rl_data_catalog.server.evidence_content import (  # noqa: PLC0415 — the maintained codec
        unpack_content,
    )

    source_id = row["source_id"]
    declaration = declarations.get(source_id)
    if declaration is None:
        raise ValueError(f"no frozen declaration for staged panel source: {source_id}")
    key = row["pipeline_key"]
    bound = {"load": inputs.prior_load_id, "key": key}
    assessments = _paged(
        query,
        "SELECT pipeline_key,source_id,scope,model,rubric,curation,curation_uri,traces"
        " FROM atlas_assessments WHERE load_id=:load AND pipeline_key=:key{predicate}"
        " ORDER BY pipeline_key",
        bound,
        ("pipeline_key",),
    )
    if len(assessments) != 1:
        raise ValueError(f"{len(assessments)} assessments for the staged panel {source_id}")
    populations = _paged(
        query,
        "SELECT pipeline_key,cohort,population,sample_count"
        " FROM atlas_populations WHERE load_id=:load AND pipeline_key=:key{predicate} ORDER BY cohort",
        bound,
        ("cohort",),
    )
    items = _paged(
        query,
        "SELECT pipeline_key,task_id,cohort,status,verdict,explanation,evidence_ids,packet_uri,has_task,"
        "omissions,exclusion_reason,confidence,reference_status,defects,provenance,replaced_task_sha256,"
        "repair_advisories FROM atlas_items WHERE load_id=:load AND pipeline_key=:key{predicate} ORDER BY task_id",
        bound,
        ("task_id",),
    )
    raw_rows = _paged(
        query,
        "SELECT pipeline_key,task_id,record,reviewer"
        " FROM atlas_raw_reviews WHERE load_id=:load AND pipeline_key=:key{predicate} ORDER BY task_id",
        bound,
        # a task may carry several review rows; the composite keyset keeps every row
        ("task_id", "reviewer"),
    )
    batch_rows = _paged(
        query,
        "SELECT pipeline_key,batch_key,record,chunk_count"
        " FROM atlas_review_batches WHERE load_id=:load AND pipeline_key=:key{predicate} ORDER BY batch_key",
        bound,
        ("batch_key",),
    )
    chunk_rows = _paged(
        query,
        "SELECT batch_key,chunk_index,payload"
        " FROM atlas_review_batch_chunks WHERE load_id=:load AND pipeline_key=:key{predicate}"
        " ORDER BY batch_key,chunk_index",
        bound,
        # chunks repeat the batch key; the composite keyset keeps every chunk
        ("batch_key", "chunk_index"),
    )
    links = _paged(
        query,
        "SELECT pipeline_key,task_id,batch_key"
        " FROM atlas_review_batch_tasks WHERE load_id=:load AND pipeline_key=:key{predicate} ORDER BY task_id",
        bound,
        ("task_id", "batch_key"),
    )
    chunks_by_batch: dict[str, list[str]] = {}
    for chunk in chunk_rows:
        chunks_by_batch.setdefault(chunk["batch_key"], []).append(chunk["payload"])
    batches = []
    for batch in batch_rows:
        record = dict(batch["record"])
        if batch["chunk_count"]:
            record["payload"] = "".join(chunks_by_batch.get(batch["batch_key"], []))
            record.pop("payload_chunks", None)
        batches.append(
            {
                "pipeline_key": batch["pipeline_key"],
                "batch_key": batch["batch_key"],
                "record": record,
                "chunk_count": batch["chunk_count"],
            }
        )
    raw = [{**r, "record": unpack_content(r["record"])} for r in raw_rows]
    if len(raw) != len(items):
        raise ValueError(f"the staged panel export for {source_id} covers {len(items)} items and {len(raw)} raw rows")
    staged = {
        "data_restriction": declaration_restriction(declaration).value,
        "assessment": assessments[0],
        "populations": populations,
        "items": items,
        "raw": raw,
        "batches": batches,
        "batch_links": links,
        "receipt": row["original_receipt"],
    }
    return _seal_panel(inputs, source_id, staged, write=write)


def _seal_panel(inputs: FreshQcIndexInputs, source_id: str, staged: dict, *, write: bool) -> dict:
    payload = gzip.compress(
        json.dumps(staged, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(), mtime=0
    )
    identity = hashlib.sha256(json.dumps(staged["receipt"], sort_keys=True).encode()).hexdigest()
    reference = {"uri": staged["receipt"]["artifact"]["uri"], "sha256": identity}
    if write:
        path = StoragePath(f"{inputs.output_root}/{STAGED_DIR}/{identity}.json.gz")
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f"staged panel {identity} exists with different content; use a new output root")
        else:
            path.write_bytes(payload, auto_mkdir=True)
        if path.read_bytes() != payload:
            raise ValueError(f"staged panel readback mismatch: {identity}")
        reference = {"uri": str(path), "sha256": hashlib.sha256(payload).hexdigest()}
    return {
        "source_id": source_id,
        "identity": identity,
        "status": "complete",
        "records": len(staged["items"]),
        "bytes": len(payload),
        "reference": reference,
    }


def _retained_index_rows(inputs: FreshQcIndexInputs, declarations: dict[str, dict]) -> list[dict]:
    """Verify unchanged panel bytes and bindings before retaining their sealed references."""
    assert inputs.prior_index is not None
    prior = read_ref(inputs.prior_index)
    if prior["kind"] != f"atlas-{inputs.release_id}-{INDEX_KIND_SUFFIX}" or prior["release_id"] != inputs.release_id:
        raise ValueError("prior QC index names another release")
    _completion_rows(
        ArtifactRef(**prior["provenance"]["browse_completion"]),
        release_id=inputs.release_id,
        prior_load_id=inputs.prior_load_id,
        target_load_id=prior["load_id"],
    )
    copied = _completion_rows(
        inputs.qc_completion,
        release_id=inputs.release_id,
        prior_load_id=inputs.prior_load_id,
        target_load_id=inputs.target_load_id,
    )
    copied_receipts = {row["source_id"]: row["original_receipt"] for row in copied}
    old_declarations = _read_declarations(ArtifactRef(**prior["provenance"]["declarations"]))
    rows = prior["receipts"]
    if len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("prior QC index repeats a source")
    retained_ids = {row["source_id"] for row in rows} - inputs.fresh_panels.keys() - set(inputs.removed_sources)
    changed = changed_review_sources(
        {key: row for key, row in old_declarations.items() if key in retained_ids},
        {key: row for key, row in declarations.items() if key in retained_ids},
    )
    retained = []
    for row in rows:
        source_id = row["source_id"]
        if source_id in inputs.fresh_panels or source_id in inputs.removed_sources:
            continue
        if source_id in changed or source_id not in declarations or source_id not in old_declarations:
            raise ValueError(f"changed QC binding requires a fresh panel: {source_id}")
        declaration = declarations[source_id]
        if declaration_restriction(old_declarations[source_id]) != declaration_restriction(declaration):
            raise ValueError(f"changed QC restriction requires a fresh panel: {source_id}")
        raw = manifest_bytes(ManifestReference(**row["reference"]))
        staged = json.loads(gzip.decompress(raw))
        receipt = staged["receipt"]
        if source_id in copied_receipts and receipt != copied_receipts[source_id]:
            raise ValueError(f"retained QC panel differs from the current copy receipt: {source_id}")
        if (
            row["status"] != "complete"
            or row["bytes"] != len(raw)
            or row["records"] != len(staged["items"])
            or len(staged["items"]) != len(staged["raw"])
            or row["identity"] != hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
            or staged["assessment"]["source_id"] != source_id
            or receipt["artifact"]["uri"] != selected_uri(declaration)
            or receipt["current_pin"] != pin_currency(declaration, receipt)
            or staged["data_restriction"] != declaration_restriction(declaration).value
        ):
            raise ValueError(f"staged QC panel differs from its retained index or declaration: {source_id}")
        retained.append(row)
    return retained


def run_fresh_qc_index(
    inputs: FreshQcIndexInputs,
    *,
    dry_run: bool,
    feeds: tuple | None = None,
) -> dict:
    """Seal the release's fresh QC index; ``dry_run`` verifies every panel with zero writes."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, "fresh_qc_index", {"mode": mode}, alert_feeds)
    try:
        inputs.validate()
        configure_coreweave_s3()
        declarations = _read_declarations(inputs.declarations)
        still_declared = set(inputs.removed_sources) & (declarations.keys() | inputs.fresh_panels.keys())
        if still_declared:
            raise ValueError(f"removed QC sources remain declared or select fresh panels: {sorted(still_declared)}")
        if inputs.prior_index is not None:
            index_rows = _retained_index_rows(inputs, declarations)
        else:
            rows = _completion_rows(
                inputs.qc_completion,
                release_id=inputs.release_id,
                prior_load_id=inputs.prior_load_id,
                target_load_id=inputs.target_load_id,
            )
            rows = [row for row in rows if row["source_id"] not in inputs.removed_sources]
            carried_ids = {row["source_id"] for row in rows}
            overlap = carried_ids & inputs.fresh_panels.keys()
            if overlap:
                raise ValueError(f"QC sources select both carried and fresh panels: {sorted(overlap)}")
            query = paced_query.open_paced_query(inputs.service_url, inputs.applet_id, paced_query.RetryControl())
            index_rows = [_export_panel(inputs, row, declarations, query, write=not dry_run) for row in rows]
        for source_id, reference in sorted(inputs.fresh_panels.items()):
            staged = stage_content_reviews(
                ManifestReference(uri=reference.uri, sha256=reference.sha256),
                declarations[source_id],
                PANEL_READ_WORKERS,
            )
            index_rows.append(_seal_panel(inputs, source_id, staged, write=not dry_run))
        index_rows.sort(key=lambda row: row["source_id"])
        if dry_run:
            alerts.stage_finished(
                inputs.release_id,
                "fresh_qc_index",
                {"mode": mode, "panels": len(index_rows)},
                alert_feeds,
            )
            return {
                "kind": f"atlas-{inputs.release_id}-{INDEX_KIND_SUFFIX}",
                "release_id": inputs.release_id,
                "load_id": inputs.target_load_id,
                "snapshot_id": inputs.snapshot_id,
                "panels": len(index_rows),
                "dry_run": True,
                "sql_writes": 0,
            }
        document = {
            "kind": f"atlas-{inputs.release_id}-{INDEX_KIND_SUFFIX}",
            "release_id": inputs.release_id,
            "load_id": inputs.target_load_id,
            "snapshot_id": inputs.snapshot_id,
            "source_revision": inputs.source_revision,
            "basis": (
                "The SAMPLE QC panels selected by frozen declarations, combining unchanged prior SQL panels "
                "with source-selected review artifacts, "
                f"staged by this release at load {inputs.target_load_id}"
            ),
            "receipts": index_rows,
            "provenance": {
                "browse_completion": {"uri": inputs.qc_completion.uri, "sha256": inputs.qc_completion.sha256},
                "declarations": {"uri": inputs.declarations.uri, "sha256": inputs.declarations.sha256},
                "fresh_panels": {key: ref.model_dump(mode="json") for key, ref in sorted(inputs.fresh_panels.items())},
                "prior_index": inputs.prior_index.model_dump(mode="json") if inputs.prior_index else None,
                "removed_sources": sorted(inputs.removed_sources),
            },
        }
        raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
        path = StoragePath(f"{inputs.output_root}/fresh-qc-index.json")
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError(f"fresh QC index at {path} exists with different content; use a new output root")
        else:
            path.write_bytes(raw, auto_mkdir=True)
        if path.read_bytes() != raw:
            raise ValueError("fresh QC index readback mismatch")
        reference = {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        alerts.stage_finished(
            inputs.release_id,
            "fresh_qc_index",
            {"mode": mode, "panels": len(index_rows), **reference},
            alert_feeds,
        )
        return {
            "kind": document["kind"],
            "release_id": inputs.release_id,
            "load_id": inputs.target_load_id,
            "snapshot_id": inputs.snapshot_id,
            "panels": len(index_rows),
            "index": reference,
            "sql_writes": 0,
        }
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "fresh_qc_index", {"mode": mode, "error": str(error)[:500]}, alert_feeds)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON FreshQcIndexInputs document")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text())
    for field in ("declarations", "qc_completion"):
        document[field] = ArtifactRef(**document[field])
    if document.get("prior_index") is not None:
        document["prior_index"] = ArtifactRef(**document["prior_index"])
    document["fresh_panels"] = {key: ArtifactRef(**value) for key, value in document["fresh_panels"].items()}
    document["removed_sources"] = tuple(document.get("removed_sources", ()))
    inputs = FreshQcIndexInputs(**document)
    proof = run_fresh_qc_index(inputs, dry_run=args.mode == "dry-run")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
